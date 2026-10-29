"""CPU-only tests for scripts/stage1/entangle.py (no torch, no GPU).

Plants one synergistic pair (each head ~0 alone, 0.05 together) and one
redundant pair (each 0.03 alone, 0.03 together) in synthetic data, then
checks that `shortlist` catches the synergy heads through the necessity
sweep and that `analyze` flags exactly the two planted pairs.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _load_entangle():
    stage1 = ROOT / "scripts" / "stage1"
    if str(stage1) not in sys.path:
        sys.path.insert(0, str(stage1))
    spec = importlib.util.spec_from_file_location("entangle", stage1 / "entangle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(np is None, "numpy not installed")
class EntangleAnalysisTests(unittest.TestCase):
    PAIR = "en_hi-latn"
    N = 38
    SYN = (21 * 12 + 5, 9 * 12 + 3)     # L21H5, L9H3
    RED = (22 * 12 + 6, 27 * 12 + 6)    # L22H6, L27H6

    @classmethod
    def setUpClass(cls):
        cls.E = _load_entangle()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        rng = np.random.default_rng(1)
        ids = [f"c{i}" for i in range(cls.N)]

        singles = rng.normal(0, 0.004, (cls.N, 336))
        singles[:, cls.RED[0]] += 0.03
        singles[:, cls.RED[1]] += 0.03
        singles[:, 21 * 12 + 2] += 0.025
        cls.v2 = cls.dir / "v2.json"
        cls.v2.write_text(json.dumps({
            "version": 2,
            "summary": {cls.PAIR: {"heatmap_mean": singles.mean(0).reshape(28, 12).tolist()}},
            "per_concept": {cls.PAIR: [
                {"concept_id": ids[c], "skipped": False,
                 "recovery": singles[c].reshape(28, 12).tolist()} for c in range(cls.N)]},
        }))

        nec = rng.normal(0, 0.004, (cls.N, 336))
        nec[:, cls.SYN[0]] += 0.04
        nec[:, cls.SYN[1]] += 0.04
        cls.noise = cls.dir / "noise.json"
        cls.noise.write_text(json.dumps({"command": "noise", "pair": cls.PAIR, "per_concept": [
            {"concept_id": ids[c], "necessity": nec[c].reshape(28, 12).tolist(),
             "necessity_all": 0.7} for c in range(cls.N)]}))

        cls.shortlist = cls.dir / "shortlist.json"
        with redirect_stdout(io.StringIO()):
            cls.E.cmd_shortlist(argparse.Namespace(
                pair=cls.PAIR, scores=cls.v2, noise=cls.noise, top=10, exclude=[],
                out=cls.shortlist, seed=0))
        sl = json.loads(cls.shortlist.read_text())

        flat, lh = cls.E.flat, cls.E.lh
        heads = [flat(n) for n in sl["heads"]]
        pairs = [tuple(sorted(p)) for p in combinations(heads, 2)]
        taken, rand_pairs = set(pairs), []
        while len(rand_pairs) < 190:
            p = tuple(sorted(rng.choice(336, 2, replace=False).tolist()))
            if p not in taken:
                taken.add(p)
                rand_pairs.append(p)
        syn, red = tuple(sorted(cls.SYN)), tuple(sorted(cls.RED))

        def joint(c, x, y):
            v = singles[c, x] + singles[c, y] + rng.normal(0, 0.002)
            if (x, y) == syn:
                v += 0.05
            if (x, y) == red:
                v -= 0.03
            return float(v)

        cls.pairs = cls.dir / "pairs.json"
        cls.pairs.write_text(json.dumps({"command": "pairs", "pair": cls.PAIR, "per_concept": [
            {"concept_id": ids[c], "gap": 13.0,
             "single": {lh(i): float(singles[c, i]) for i in heads},
             "pair": {f"{lh(x)}+{lh(y)}": joint(c, x, y) for x, y in pairs},
             "random_pair": {f"{lh(x)}+{lh(y)}": joint(c, x, y) for x, y in rand_pairs}}
            for c in range(cls.N)]}))

        cls.analysis = cls.dir / "analysis.json"
        with redirect_stdout(io.StringIO()):
            cls.E.cmd_analyze(argparse.Namespace(
                pair=cls.PAIR, scores=cls.v2, curve=cls.dir / "absent.json",
                noise=cls.noise, pairs_file=cls.pairs, boot=300, seed=0,
                singles="v2", out=cls.analysis))
        cls.result = json.loads(cls.analysis.read_text())
        cls.analysis_same = cls.dir / "analysis_same.json"
        with redirect_stdout(io.StringIO()):
            cls.E.cmd_analyze(argparse.Namespace(
                pair=cls.PAIR, scores=cls.v2, curve=cls.dir / "absent.json",
                noise=cls.noise, pairs_file=cls.pairs, boot=300, seed=0,
                singles="same-run", out=cls.analysis_same))
        cls.result_same = json.loads(cls.analysis_same.read_text())
        cls.analysis_t = cls.dir / "analysis_t.json"
        with redirect_stdout(io.StringIO()):
            cls.E.cmd_analyze(argparse.Namespace(
                pair=cls.PAIR, scores=cls.v2, curve=cls.dir / "absent.json",
                noise=cls.noise, pairs_file=cls.pairs, boot=300, seed=0,
                singles="v2", stat="t", out=cls.analysis_t))
        cls.result_t = json.loads(cls.analysis_t.read_text())
        cls.sl = sl

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_shortlist_finds_synergy_heads_via_necessity(self):
        for name in ("L21H5", "L9H3"):
            self.assertIn(name, self.sl["heads"])
            self.assertIn(name, self.sl["necessary_not_sufficient"])

    def test_shortlist_records_rule_and_hashes(self):
        self.assertEqual(self.sl["n_pairs"], len(self.sl["heads"]) * (len(self.sl["heads"]) - 1) // 2)
        self.assertIn("scores", self.sl["inputs_sha256"])

    def test_analysis_flags_exactly_the_planted_pairs(self):
        found = {s["pair"]: s["kind"] for s in self.result["pairs"]["significant"]}
        self.assertEqual(found, {"L9H3+L21H5": "synergy", "L22H6+L27H6": "redundancy"})

    def test_both_baselines_reported_and_agree_when_singles_match(self):
        # In the synthetic data the re-measured singles equal the v2 singles,
        # so the two baselines must agree.
        p = self.result["pairs"]
        self.assertEqual(set(p["by_baseline"]), {"v2", "same-run"})
        self.assertAlmostEqual(p["mean_shift_same_run_minus_v2"], 0.0, places=9)
        self.assertEqual(p["by_baseline"]["v2"]["significant"],
                         p["by_baseline"]["same-run"]["significant"])
        self.assertEqual(p["baseline"], "v2")
        self.assertEqual(self.result_same["pairs"]["baseline"], "same-run")

    def test_same_run_primary_finds_the_planted_pairs(self):
        found = {s["pair"] for s in self.result_same["pairs"]["significant"]}
        self.assertEqual(found, {"L9H3+L21H5", "L22H6+L27H6"})

    def test_max_t_is_reported_and_finds_the_planted_pairs(self):
        p = self.result["pairs"]
        for source in ("v2", "same-run"):
            b = p["by_baseline"][source]
            self.assertIn("tau_t", b)
            self.assertEqual(set(b["significant_t"]), {"L9H3+L21H5", "L22H6+L27H6"})
        self.assertIn("t", p["all"][0])
        self.assertEqual(self.result_t["pairs"]["stat"], "t")
        self.assertEqual({s["pair"] for s in self.result_t["pairs"]["significant"]},
                         {"L9H3+L21H5", "L22H6+L27H6"})

    def test_max_t_survives_noisy_pairs_that_inflate_max_mean(self):
        # A few very noisy pairs raise the single max-|mean| threshold for every
        # pair; standardising each pair by its own spread does not.
        rng = np.random.default_rng(0)
        n = 38
        X = np.hstack([rng.normal(0, 0.002, (n, 2000)),     # quiet null pairs
                       rng.normal(0, 0.10, (n, 30)),        # noisy null pairs
                       rng.normal(0.004, 0.002, (n, 1))])   # a real, quiet interaction
        tau = self.E.signflip_maxabs(X, 400, rng)
        tau_t = self.E.signflip_maxt(X, 400, rng)
        self.assertLess(abs(X.mean(0)[-1]), tau)            # max |mean| misses it
        self.assertGreater(abs(self.E.tstat(X)[-1]), tau_t)  # max |t| finds it

    def test_missing_curve_is_skipped_not_fatal(self):
        self.assertNotIn("curve", self.result)
        self.assertIn("noise", self.result)


if __name__ == "__main__":
    unittest.main()
