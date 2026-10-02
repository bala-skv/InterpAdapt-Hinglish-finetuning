"""Fast, dependency-free checks for the refactored repository contract."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_tracing_set_contract(self) -> None:
        path = ROOT / "data" / "tracing" / "tracing_set_v1.json"
        items = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(items), 38)
        required = {
            "concept_id", "category",
            "en_prompt", "en_target",
            "hi_latn_prompt", "hi_latn_target",
            "hi_deva_prompt", "hi_deva_target",
        }
        ids = set()
        for item in items:
            self.assertTrue(required <= item.keys())
            self.assertNotIn(item["concept_id"], ids)
            ids.add(item["concept_id"])
            for cond in ("en", "hi_latn", "hi_deva"):
                self.assertTrue(item[f"{cond}_prompt"].endswith(" "))
                self.assertFalse(item[f"{cond}_target"].startswith(" "))

    def test_stage1_trace_contract(self) -> None:
        path = ROOT / "results" / "stage1" / "full_trace_v2.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["version"], 2)
        for pair in ("en_hi-latn", "hi-deva_hi-latn", "en_hi-deva"):
            heatmap = data["summary"][pair]["heatmap_mean"]
            self.assertEqual(len(heatmap), 28)
            self.assertTrue(all(len(row) == 12 for row in heatmap))

    def test_stage1_scores_contract(self) -> None:
        path = ROOT / "results" / "stage1" / "stage1_head_scores.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        scores = data["scores"]
        self.assertEqual(len(scores), 28)
        self.assertTrue(all(len(row) == 12 for row in scores))

    def test_canonical_entrypoints_exist(self) -> None:
        expected = [
            "scripts/phase0/run_phase0.py",
            "scripts/phase0/verify_model.py",
            "scripts/phase0/hook_smoke_test.py",
            "scripts/phase0/memory_pilot.py",
            "scripts/stage1/build_tracing_set_v1.py",
            "scripts/stage1/validate_tracing_set_v1.py",
            "scripts/stage1/full_trace.py",
            "scripts/stage1/analyze_trace.py",
            "scripts/stage2/train_lora_baseline.py",
            "scripts/stage2/train_cra_compare.py",
            "scripts/stage2/build_stage1_scores.py",
            "cluster/memory_pilot.slurm",
            "cluster/train_lora_baseline.slurm",
            "cluster/train_cra_compare.slurm",
        ]
        for rel in expected:
            self.assertTrue((ROOT / rel).is_file(), rel)


if __name__ == "__main__":
    unittest.main()
