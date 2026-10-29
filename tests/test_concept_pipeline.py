"""CPU-only tests for scripts/pipeline/run_concepts.py.

The GPU scripts are replaced by stand-ins that write outputs of the right shape,
so these tests exercise the orchestration (order, skipping, guards, freezing,
validation handling) and the real CPU analysis and report code.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_concepts", ROOT / "scripts" / "pipeline" / "run_concepts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RC = _load()
PAIRS = ("en_hi-latn", "hi-deva_hi-latn", "en_hi-deva")


def _concepts(n=8):
    cats = ["feature_a", "feature_b"]
    return [{"concept_id": f"c{i}", "category": cats[i % 2],
             "en_prompt": f"english prompt {i} ", "en_target": f"word{i}",
             "hi_deva_prompt": f"deva prompt {i} ", "hi_deva_target": f"deva{i}",
             "hi_latn_prompt": f"latn prompt {i} ", "hi_latn_target": f"latn{i}"}
            for i in range(n)]


class FakeScripts:
    """Stand-in for Run.script: writes outputs of the right shape."""

    def __init__(self):
        self.calls = []
        self.unusable = set()

    def __call__(self, run, step, args, ok_codes=(0,)):
        args = [str(a) for a in args]
        self.calls.append(step)
        out = Path(args[args.index("--out") + 1])
        data = json.loads(Path(args[args.index("--data") + 1]).read_text()) if "--data" in args else []
        rng = np.random.default_rng(len(self.calls))
        if args[0].endswith("validate_tracing_set_v1.py"):
            rows = [{"concept_id": c["concept_id"], "category": c["category"],
                     "usable": c["concept_id"] not in self.unusable} for c in data]
            summ = {"n_items": len(rows), "n_usable": sum(r["usable"] for r in rows),
                    "mean_contrast": 10.0, "min_contrast": 2.0, "max_contrast": 15.0,
                    "mean_cross_penalty": 1.0}
            obj = {"summary": {"pairs": {k: summ for k in ("en<->hi_deva", "en<->hi_latn", "hi_deva<->hi_latn")}},
                   "pairs": {k: rows for k in ("en<->hi_deva", "en<->hi_latn", "hi_deva<->hi_latn")}}
        elif args[0].endswith("full_trace.py"):
            mlp = args[args.index("--component") + 1] == "mlp"
            shape = (28,) if mlp else (28, 12)
            per, summ = {}, {}
            for p in PAIRS:
                recs = []
                for c in data:
                    r = rng.normal(0, 0.003, shape)
                    if mlp:
                        r[22] += 0.08                 # planted MLP layer
                    else:
                        r[21, 2] += 0.03              # planted head
                    recs.append({"concept_id": c["concept_id"], "category": c["category"],
                                 "skipped": False, "recovery": r.tolist()})
                per[p] = recs
                summ[p] = {"contrast_mean": 12.0, "n_used": len(recs),
                           "heatmap_mean": np.mean([x["recovery"] for x in recs], 0).flatten().tolist()}
            obj = {"version": 2, "component": "mlp" if mlp else "heads", "summary": summ, "per_concept": per}
        else:
            raise AssertionError(f"unexpected script in test: {args[0]}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(obj))
        return 0


@unittest.skipIf(np is None, "numpy not installed")
class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "set.json"
        self.src.write_text(json.dumps(_concepts()))
        self.fake = FakeScripts()
        self._orig_script, self._orig_gpu = RC.Run.script, RC.gpu_name
        fake = self.fake
        RC.Run.script = lambda run, step, args, ok_codes=(0,): fake(run, step, args, ok_codes)
        RC.gpu_name = lambda: "NVIDIA GeForce RTX 2080 Ti"

    def tearDown(self):
        RC.Run.script, RC.gpu_name = self._orig_script, self._orig_gpu
        self.tmp.cleanup()

    def main(self, *extra, name="t"):
        argv = ["--concepts", str(self.src), "--name", name,
                "--results-root", str(self.dir / "results"),
                "--figures-root", str(self.dir / "figures"), "--boot", "200"] + list(extra)
        with redirect_stdout(io.StringIO()) as buf:
            RC.main(argv)
        return buf.getvalue()

    def run_dir(self, name="t"):
        return self.dir / "results" / name

    # ---- checks ----------------------------------------------------------------
    def test_v1_tracing_set_passes_check(self):
        v1 = json.loads((ROOT / "data" / "tracing" / "tracing_set_v1.json").read_text(encoding="utf-8"))
        errors, _ = RC.check_concepts(v1)
        self.assertEqual(errors, [])

    def test_check_flags_duplicates_identical_targets_and_missing_fields(self):
        bad = _concepts(3)
        bad[1]["concept_id"] = "c0"
        bad[2]["hi_latn_target"] = bad[2]["en_target"]
        del bad[0]["hi_deva_prompt"]
        errors, _ = RC.check_concepts(bad)
        text = " ".join(errors)
        self.assertIn("appears 2 times", text)
        self.assertIn("identical", text)
        self.assertIn("hi_deva_prompt", text)

    def test_parse_steps(self):
        self.assertEqual(RC.parse_steps("all"), RC.STEP_NAMES)
        self.assertTrue(all(RC.DEVICE[s] == "gpu" for s in RC.parse_steps("gpu")))
        self.assertEqual(RC.parse_steps("report,heads"), ["heads", "report"])
        with self.assertRaises(SystemExit):
            RC.parse_steps("heads,nonsense")

    def test_estimate_matches_the_v2_trace_size(self):
        est = RC.estimate_passes("heads", 38, 50, 1)
        self.assertLess(abs(est - 76608) / 76608, 0.02)

    # ---- orchestration ---------------------------------------------------------
    def test_dry_run_writes_nothing(self):
        out = self.main("--dry-run")
        self.assertIn("total still to run", out)
        self.assertFalse(self.run_dir().exists())

    def test_end_to_end_cpu_with_stand_in_gpu_scripts(self):
        self.main("--steps", "check,validate,heads,mlp,mlp_analysis,report")
        rd = self.run_dir()
        man = json.loads((rd / "manifest.json").read_text())
        for s in ("check", "validate", "heads", "mlp", "mlp_analysis", "report"):
            self.assertEqual(man["steps"][s]["status"], "ok", s)
        self.assertEqual(man["device"], "NVIDIA GeForce RTX 2080 Ti")
        mlp = json.loads((rd / "mlp_analysis.json").read_text())
        self.assertIn(22, mlp["pairs"]["en_hi-latn"]["significant_t"])
        report = (rd / "REPORT.md").read_text()
        self.assertIn("## 5. MLP layers", report)
        self.assertIn("L22", report)
        self.assertIn("feature_a", report)                  # per-category table
        n_calls = len(self.fake.calls)
        self.main("--steps", "check,validate,heads,mlp,mlp_analysis,report")
        self.assertEqual(len(self.fake.calls), n_calls)     # nothing re-run

    def test_force_reruns_one_step(self):
        self.main("--steps", "check,validate,heads")
        n = len(self.fake.calls)
        self.main("--steps", "check,validate,heads", "--force", "heads")
        self.assertEqual(self.fake.calls[n:], ["heads"])

    def test_missing_input_is_reported(self):
        with self.assertRaises(SystemExit) as cm:
            self.main("--steps", "heads")
        self.assertIn("validate", str(cm.exception.code))

    # ---- guards ----------------------------------------------------------------
    def test_wrong_gpu_is_refused(self):
        RC.gpu_name = lambda: "NVIDIA GeForce GTX 1080 Ti"
        with self.assertRaises(SystemExit) as cm:
            self.main("--steps", "check,validate")
        self.assertIn("2080 Ti", str(cm.exception.code))

    def test_gpu_change_within_a_run_is_refused(self):
        self.main("--steps", "check,validate")
        RC.gpu_name = lambda: "NVIDIA A100"
        with self.assertRaises(SystemExit):
            self.main("--steps", "heads", "--require-gpu", "")
        self.main("--steps", "heads", "--require-gpu", "", "--allow-gpu-change")

    def test_edited_concept_file_is_detected(self):
        self.main("--steps", "check")
        data = _concepts()
        data[0]["en_target"] = "changed"
        self.src.write_text(json.dumps(data))
        with self.assertRaises(SystemExit):
            self.main("--steps", "check")
        self.main("--steps", "check", "--force", "check")
        frozen = json.loads((self.run_dir() / "concepts.json").read_text())
        self.assertEqual(frozen[0]["en_target"], "changed")

    def test_unusable_concepts_stop_or_are_dropped(self):
        self.fake.unusable = {"c3"}
        with self.assertRaises(SystemExit) as cm:
            self.main("--steps", "check,validate")
        self.assertIn("c3", str(cm.exception.code))
        self.main("--steps", "check,validate", "--drop-unusable", "--force", "validate")
        used = json.loads((self.run_dir() / "concepts_used.json").read_text())
        self.assertNotIn("c3", [c["concept_id"] for c in used])
        self.assertEqual(len(used), 7)

    def test_smoke_runs_use_their_own_folder(self):
        self.main("--steps", "check", "--limit", "3")
        frozen = json.loads((self.dir / "results" / "t__smoke3" / "concepts.json").read_text())
        self.assertEqual(len(frozen), 3)
        self.assertFalse(self.run_dir().exists())


if __name__ == "__main__":
    unittest.main()
