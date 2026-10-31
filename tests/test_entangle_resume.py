"""CPU-only tests for the resumable `pairs` run and `shortlist --size`.

A stand-in replaces the GPU tracer, so these run anywhere with numpy.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
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
    spec = importlib.util.spec_from_file_location("entangle_resume_mod", stage1 / "entangle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONCEPTS = [{"concept_id": f"c{i}", "category": "x"} for i in range(6)]


class FakeTracer:
    """Deterministic stand-in for the GPU tracer; can fail on a chosen concept."""
    fail_on = None
    recovery_calls = 0
    ft = types.SimpleNamespace(MIN_CONTRAST=0.5)
    env = {"device": "fake"}

    def __init__(self, _data):
        pass

    def selftest(self, pair, v2):
        return {"identity_diff": 0.0, "v2_max_diff": 0.0}

    def items(self, limit):
        return CONCEPTS[:limit] if limit else CONCEPTS

    def prepare(self, item, pair):
        if FakeTracer.fail_on == item["concept_id"]:
            raise RuntimeError("simulated kill")
        return {"gap": 10.0, "s_a": None, "cid": int(item["concept_id"][1:])}

    def capture(self, enc):
        return None

    def recovery(self, P, heads, stored):
        FakeTracer.recovery_calls += 1
        return 1e-3 * P["cid"] + 1e-5 * sum(h + 1 for h in heads)


@unittest.skipIf(np is None, "numpy not installed")
class ResumeTests(unittest.TestCase):
    PAIR = "en_hi-latn"

    @classmethod
    def setUpClass(cls):
        cls.E = _load_entangle()
        cls.E.new_tracer = lambda data: FakeTracer(data)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        rng = np.random.default_rng(0)
        suff = rng.normal(0, 0.004, 336)
        suff[21 * 12 + 2] = 0.03
        suff[0 * 12 + 10] = 0.02                      # the head we exclude
        cls.v2 = cls.dir / "v2.json"
        cls.v2.write_text(json.dumps(
            {"summary": {cls.PAIR: {"heatmap_mean": suff.reshape(28, 12).tolist()}}}))
        nec = rng.normal(0, 0.004, (3, 336))
        nec[:, 21 * 12 + 2] += 0.02                   # same head high in both rankings
        cls.noise = cls.dir / "noise.json"
        cls.noise.write_text(json.dumps({"pair": cls.PAIR, "per_concept": [
            {"necessity": nec[c].reshape(28, 12).tolist()} for c in range(3)]}))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def shortlist(self, **kw):
        out = self.dir / f"sl_{kw.get('size') or kw.get('top')}.json"
        args = dict(pair=self.PAIR, scores=self.v2, noise=self.noise, top=10,
                    exclude=["L0H10"], out=out, seed=0, size=None)
        args.update(kw)
        with redirect_stdout(io.StringIO()):
            self.E.cmd_shortlist(argparse.Namespace(**args))
        return json.loads(out.read_text()), out

    def pairs(self, sl_path, out, **kw):
        args = dict(pair=self.PAIR, scores=self.v2, shortlist=sl_path, out=out, seed=0,
                    data=None, limit=None, random=5, fresh=False)
        args.update(kw)
        with redirect_stdout(io.StringIO()):
            self.E.cmd_pairs(argparse.Namespace(**args))

    # ---- shortlist --size ----------------------------------------------------
    def test_size_gives_exactly_that_many_distinct_heads_without_excluded(self):
        sl, _ = self.shortlist(size=40)
        self.assertEqual(len(sl["heads"]), 40)
        self.assertEqual(len(set(sl["heads"])), 40)
        self.assertNotIn("L0H10", sl["heads"])
        self.assertEqual(sl["n_pairs"], 40 * 39 // 2)

    def test_small_shortlist_is_a_subset_of_the_larger_one(self):
        small, _ = self.shortlist(top=10)
        big, _ = self.shortlist(size=40)
        self.assertTrue(set(small["heads"]) <= set(big["heads"]))

    # ---- resume --------------------------------------------------------------
    def test_interrupted_run_resumes_and_matches_a_clean_run(self):
        _, sl_path = self.shortlist(size=6)
        clean = self.dir / "clean.json"
        self.pairs(sl_path, clean)
        clean_rows = json.loads(clean.read_text())["per_concept"]

        out = self.dir / "resumed.json"
        partial = out.with_name("resumed.partial.jsonl")
        FakeTracer.fail_on = "c3"
        with self.assertRaises(RuntimeError):
            self.pairs(sl_path, out)
        FakeTracer.fail_on = None
        self.assertTrue(partial.is_file())
        self.assertEqual(len(partial.read_text().splitlines()) - 1, 3)   # header + 3 concepts
        self.assertFalse(out.exists())

        FakeTracer.recovery_calls = 0
        self.pairs(sl_path, out)
        n_heads, n_pairs, n_rand = 6, 15, 5
        per_concept_calls = n_heads + n_pairs + n_rand
        self.assertEqual(FakeTracer.recovery_calls, 3 * per_concept_calls)   # only c3..c5 recomputed
        self.assertFalse(partial.exists())
        got = json.loads(out.read_text())["per_concept"]
        strip = lambda rows: [{k: v for k, v in r.items() if k != "seconds"} for r in rows]
        self.assertEqual(strip(got), strip(clean_rows))

    def test_truncated_last_line_is_ignored(self):
        _, sl_path = self.shortlist(size=6)
        out = self.dir / "trunc.json"
        partial = out.with_name("trunc.partial.jsonl")
        FakeTracer.fail_on = "c2"
        with self.assertRaises(RuntimeError):
            self.pairs(sl_path, out)
        FakeTracer.fail_on = None
        with partial.open("a") as fh:
            fh.write('{"concept_id": "c2", "gap": 10.0, "sing')      # a kill mid-write
        self.pairs(sl_path, out)
        self.assertEqual(len(json.loads(out.read_text())["per_concept"]), len(CONCEPTS))

    def test_mismatched_partial_is_refused_and_fresh_overrides(self):
        _, sl_path = self.shortlist(size=6)
        out = self.dir / "mismatch.json"
        FakeTracer.fail_on = "c1"
        with self.assertRaises(RuntimeError):
            self.pairs(sl_path, out)
        FakeTracer.fail_on = None
        with self.assertRaises(SystemExit):
            self.pairs(sl_path, out, seed=1)            # different seed -> different run
        self.pairs(sl_path, out, seed=1, fresh=True)
        self.assertEqual(len(json.loads(out.read_text())["per_concept"]), len(CONCEPTS))


if __name__ == "__main__":
    unittest.main()