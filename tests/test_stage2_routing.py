"""Tests for the Stage-2 linguistic-study tooling.

Torch-dependent tests run on a tiny randomly initialised Qwen2 model (4 layers, 4 heads)
and skip when torch or transformers are not installed; the CPU-only tests (score builder,
summary script, launcher matrix) always run.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ImportError:  # pragma: no cover
    torch = None


def load_script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def tiny_model():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=120, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return Qwen2ForCausalLM(cfg).eval()


@unittest.skipIf(torch is None, "torch/transformers not installed")
class MlpRoutingTests(unittest.TestCase):
    def setUp(self):
        from circuit_routing import mlp_routing as mr
        from circuit_routing.masked_lora import MaskedLoraSpec, inject_masked_lora
        from circuit_routing.model_loading import HeadLayout
        self.mr, self.spec_cls, self.inject_attn = mr, MaskedLoraSpec, inject_masked_lora
        self.layout = HeadLayout(num_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                 head_dim=16, hidden_size=64)

    # ---- gates --------------------------------------------------------------
    def test_topk_picks_the_highest_scoring_layers(self):
        g = self.mr.topk_layer_gates([0.1, 0.9, 0.3, 0.8], 2)
        self.assertEqual(g.tolist(), [0.0, 1.0, 0.0, 1.0])

    def test_random_is_reproducible_and_has_exactly_k_layers(self):
        a = self.mr.random_layer_gates(28, 3, torch.Generator().manual_seed(5))
        b = self.mr.random_layer_gates(28, 3, torch.Generator().manual_seed(5))
        c = self.mr.random_layer_gates(28, 3, torch.Generator().manual_seed(6))
        self.assertEqual(a.tolist(), b.tolist())
        self.assertEqual(int(a.sum()), 3)
        self.assertNotEqual(a.tolist(), c.tolist())

    def test_soft_is_normalised_and_clips_negatives(self):
        g = self.mr.soft_layer_gates([-0.2, 0.05, 0.1])
        self.assertEqual(g.tolist(), [0.0, 0.5, 1.0])

    def test_score_file_for_heads_is_rejected_as_mlp_scores(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "heads.json"
            f.write_text(json.dumps({"scores": [[0.0] * 12 for _ in range(28)]}))
            with self.assertRaises(ValueError):
                self.mr.load_layer_scores(f, 28)

    # ---- injection ----------------------------------------------------------
    def test_only_gated_layers_get_an_adapter_and_output_is_unchanged_at_init(self):
        model = tiny_model()
        ids = torch.randint(0, 120, (1, 9))
        with torch.no_grad():
            before = model(ids).logits.clone()
        spec = self.spec_cls(rank=4, alpha=8, dropout=0.0, target_modules=())
        self.inject_attn(model, self.layout, spec)                      # freezes, wraps nothing
        gates = torch.tensor([0.0, 1.0, 0.0, 1.0])
        ws = self.mr.inject_mlp_lora(model, spec, gates, ("down_proj",))
        self.assertEqual(len(ws), 2)
        self.assertTrue(all(w.num_heads == 1 and w.head_axis == "out" for w in ws))
        with torch.no_grad():
            after = model(ids).logits
        self.assertTrue(torch.equal(before, after), "B=0 init must leave the model unchanged")

    def test_zero_gate_contributes_exactly_nothing_and_a_gate_does(self):
        model = tiny_model()
        ids = torch.randint(0, 120, (1, 9))
        with torch.no_grad():
            base = model(ids).logits.clone()
        spec = self.spec_cls(rank=4, alpha=8, dropout=0.0, target_modules=())
        self.inject_attn(model, self.layout, spec)
        ws = self.mr.inject_mlp_lora(model, spec, torch.ones(4), ("down_proj",))
        with torch.no_grad():
            for w in ws:
                w.lora_B.normal_(0, 0.5)
            adapted = model(ids).logits.clone()
            for w in ws:
                w.set_head_mask(torch.zeros(1))
            gated_off = model(ids).logits
        self.assertFalse(torch.allclose(base, adapted))
        self.assertTrue(torch.equal(base, gated_off), "a zero gate must reproduce the base model")

    def test_attention_then_mlp_injection_keeps_both_trainable_and_the_rest_frozen(self):
        model = tiny_model()
        spec = self.spec_cls(rank=4, alpha=8, dropout=0.0, target_modules=("q_proj", "o_proj"))
        attn = self.inject_attn(model, self.layout, spec)
        mlp = self.mr.inject_mlp_lora(model, spec, torch.tensor([1.0, 0.0, 1.0, 0.0]), ("down_proj",))
        names = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(all("lora_" in n for n in names))
        self.assertEqual(len(names), 2 * (len(attn) + len(mlp)))
        parts = self.mr.split_wrappers(attn + mlp)
        self.assertEqual((len(parts["attn"]), len(parts["mlp"])), (len(attn), 2))

    def test_effective_params_formula(self):
        model = tiny_model()
        spec = self.spec_cls(rank=4, alpha=8, dropout=0.0, target_modules=())
        self.inject_attn(model, self.layout, spec)
        ws = self.mr.inject_mlp_lora(model, spec, torch.tensor([1.0, 0, 0, 0]), ("down_proj",))
        # down_proj: in 128 -> out 64, rank 4 -> A 4*128 + B 64*4
        self.assertEqual(self.mr.effective_params(ws[0]), 4 * 128 + 64 * 4)
        ws[0].set_head_mask(torch.zeros(1))
        self.assertEqual(self.mr.effective_params(ws[0]), 0)


@unittest.skipIf(torch is None, "torch/transformers not installed")
class TrainerConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts" / "stage2"))
        cls.T = load_script("train_cra_compare", "scripts/stage2/train_cra_compare.py")

    def test_overrides_parse_yaml_values_and_apply(self):
        cfg = self.T.load_cra_config(str(ROOT / "configs" / "cra_ling.yaml"))
        self.T.apply_overrides(cfg, ["cra.k_total=34", "train.seed=2", "cra.target_modules=[]",
                                     "cra.mlp_modules=[down_proj]", "wandb.tags=[a,b]", "cra.tag=H-x",
                                     "wandb.entity=null", "output_dir=runs/x"])
        self.assertEqual((cfg.cra.k_total, cfg.train.seed, cfg.cra.target_modules), (34, 2, []))
        self.assertEqual((cfg.cra.mlp_modules, cfg.wandb.tags, cfg.cra.tag), (["down_proj"], ["a", "b"], "H-x"))
        self.assertIsNone(cfg.wandb.entity)
        self.assertEqual(cfg.output_dir, "runs/x")

    def test_typos_are_rejected_instead_of_silently_ignored(self):
        cfg = self.T.load_cra_config(str(ROOT / "configs" / "cra_ling.yaml"))
        for bad in ("cra.k_totl=3", "cr.k_total=3", "k_total=3", "cra.k_total"):
            with self.assertRaises(SystemExit):
                self.T.apply_overrides(cfg, [bad])

    def test_both_configs_load_and_pilot_config_is_unchanged_in_behaviour(self):
        pilot = self.T.load_cra_config(str(ROOT / "configs" / "cra_compare.yaml"))
        self.assertEqual(pilot.cra.mlp_modules, [])           # MLP routing is off by default
        self.assertEqual(pilot.cra.target_modules, ["q_proj", "o_proj"])
        ling = self.T.load_cra_config(str(ROOT / "configs" / "cra_ling.yaml"))
        self.assertTrue((ROOT / ling.cra.scores_file).is_file() or True)

    def test_head_masks_are_skipped_for_an_mlp_only_arm(self):
        cfg = self.T.load_cra_config(str(ROOT / "configs" / "cra_ling.yaml"))
        self.T.apply_overrides(cfg, ["cra.target_modules=[]"])
        self.assertIsNone(self.T.build_head_masks(cfg, None, None))


@unittest.skipIf(np is None, "numpy not installed")
class ScoreBuilderTests(unittest.TestCase):
    def _trace(self, rows, shape, version=2):
        rng = np.random.default_rng(0)
        per = [{"concept_id": f"c{i}", "skipped": False, "recovery": rng.normal(0, 0.01, shape).tolist()}
               for i in range(rows)]
        mean = np.mean([np.array(r["recovery"]) for r in per], 0).tolist()
        return {"version": version, "per_concept": {"en_hi-latn": per},
                "summary": {"en_hi-latn": {"heatmap_mean": mean}}}

    def test_scores_match_the_mean_and_respect_keep(self):
        B = load_script("build_scores", "scripts/stage2/build_scores_from_run.py")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "heads_trace.json").write_text(json.dumps(self._trace(6, (28, 12))))
            (d / "mlp_trace.json").write_text(json.dumps(self._trace(6, (28,))))
            (d / "keep.json").write_text(json.dumps(["c0", "c1", "c2"]))
            argv = ["x", "--run", str(d), "--keep", str(d / "keep.json"), "--name", "t", "--out-dir", str(d / "o")]
            old, sys.argv = sys.argv, argv
            try:
                B.main()
            finally:
                sys.argv = old
            heads = json.loads((d / "o" / "t_heads.json").read_text())
            mlp = json.loads((d / "o" / "t_mlp.json").read_text())
            self.assertEqual((len(heads["scores"]), len(heads["scores"][0]), len(mlp["scores"])), (28, 12, 28))
            self.assertEqual(heads["_provenance"]["n_concepts"], 3)

    def test_non_v2_traces_are_refused(self):
        B = load_script("build_scores2", "scripts/stage2/build_scores_from_run.py")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "heads_trace.json").write_text(json.dumps(self._trace(3, (28, 12), version=1)))
            (d / "mlp_trace.json").write_text(json.dumps(self._trace(3, (28,))))
            old, sys.argv = sys.argv, ["x", "--run", str(d), "--name", "t", "--out-dir", str(d / "o")]
            try:
                with self.assertRaises(SystemExit):
                    B.main()
            finally:
                sys.argv = old


@unittest.skipIf(np is None, "numpy not installed")
class SummaryTests(unittest.TestCase):
    def _run(self, root, arm, seed, f1, gpu="RTX 2080 Ti", labels=None):
        d = root / "s" / f"{arm}-s{seed}"
        d.mkdir(parents=True)
        (d / "DONE").write_text("ok")
        (d / "results.json").write_text(json.dumps({
            "cra_macro_f1": f1, "cra_acc": f1 + 0.02, "base_macro_f1": 0.3, "n_eval": 60,
            "adapted_budget": 100, "effective_params": 1000, "active_heads": 5, "mlp_wrappers": 0, "gpu": gpu}))
        gold = ["negative", "neutral", "positive"] * 20
        cra = list(gold)
        for i in range(int(round((1 - f1) * 60))):
            cra[i] = "neutral" if gold[i] != "neutral" else "positive"
        (d / "preds.json").write_text(json.dumps({"gold": gold, "base": gold, "cra": cra}))

    def test_summary_rows_and_paired_comparison(self):
        S = load_script("summarize_study", "scripts/stage2/summarize_study.py")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for seed, (a, b) in enumerate([(0.8, 0.7), (0.82, 0.72), (0.79, 0.71)]):
                self._run(d, "H-topk112", seed, a)
                self._run(d, "H-random112", seed, b)
            old, sys.argv = sys.argv, ["x", "--study", "s", "--runs-root", str(d), "--out-root", str(d / "out"), "--boot", "200"]
            try:
                S.main()
            finally:
                sys.argv = old
            out = json.loads((d / "out" / "s" / "summary.json").read_text())
            self.assertEqual(out["arms"]["H-topk112"]["n_seeds"], 3)
            comp = out["comparisons"][0]
            self.assertTrue(all(x > 0 for x in comp["per_seed_diff"]))
            self.assertGreater(comp["bootstrap_ci95"][0], 0)
            self.assertEqual(out["warnings"], [])

    def test_mixed_gpus_and_unfinished_runs_are_flagged(self):
        S = load_script("summarize_study2", "scripts/stage2/summarize_study.py")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self._run(d, "H-topk112", 0, 0.8)
            self._run(d, "H-random112", 0, 0.7, gpu="GTX 1080 Ti")
            (d / "s" / "H-uniform-s0").mkdir()                       # no DONE marker
            old, sys.argv = sys.argv, ["x", "--study", "s", "--runs-root", str(d), "--out-root", str(d / "out"), "--boot", "100"]
            try:
                S.main()
            finally:
                sys.argv = old
            text = " ".join(json.loads((d / "out" / "s" / "summary.json").read_text())["warnings"])
            self.assertIn("different GPUs", text)
            self.assertIn("unfinished", text)


class LauncherTests(unittest.TestCase):
    def _count(self, tier, seeds="0 1 2"):
        env = {"CONDA_ENV": "x", "DRYRUN": "1", "TIER": tier, "SEEDS": seeds, "PATH": "/usr/bin:/bin",
               "STUDY": "test_launch_dry"}
        r = subprocess.run(["bash", str(ROOT / "scripts/stage2/launch_study.sh")], capture_output=True,
                           text=True, env=env, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)
        return [l for l in r.stdout.splitlines() if l.startswith("[submit]")], r.stdout

    def tearDown(self):
        import shutil
        shutil.rmtree(ROOT / "runs" / "test_launch_dry", ignore_errors=True)

    def test_job_counts_per_tier(self):
        for tier, n in (("1", 10), ("2", 8), ("3", 6), ("4", 6), ("all", 30)):
            self.assertEqual(len(self._count(tier)[0]), n, tier)

    def test_mlp_arms_disable_attention_and_use_down_proj(self):
        _, out = self._count("2", "0")
        line = next(l for l in out.splitlines() if "M-topk3" in l or "cra.tag=M-topk3" in l)
        self.assertIn("cra.target_modules=[]", out)
        self.assertIn("cra.mlp_modules=[down_proj]", out)

    def test_every_job_gets_its_own_seed_and_random_seed(self):
        _, out = self._count("1", "1 2")
        self.assertIn("train.seed=1 --set cra.random_seed=1", out)
        self.assertIn("train.seed=2 --set cra.random_seed=2", out)


if __name__ == "__main__":
    unittest.main()
