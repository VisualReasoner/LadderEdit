import unittest

import torch

from easyeditor.models.hopedit.diagnostics import summarize_conflicts, summarize_route_diagnostics


class HopEditDiagnosticsV2Tests(unittest.TestCase):
    def test_within_cell_conflict_is_reported(self):
        entries = [
            {
                "edit_id": "e0",
                "cell_id": "c0",
                "prompt": "p0",
                "raw_semantic_key": torch.tensor([1.0, 0.0]),
                "raw_activation_key": torch.tensor([1.0, 0.0]),
                "semantic_key": torch.tensor([1.0, 0.0]),
                "activation_key": torch.tensor([1.0, 0.0]),
            },
            {
                "edit_id": "e1",
                "cell_id": "c0",
                "prompt": "p1",
                "raw_semantic_key": torch.tensor([0.9, 0.1]),
                "raw_activation_key": torch.tensor([0.9, 0.1]),
                "semantic_key": torch.tensor([0.9, 0.1]),
                "activation_key": torch.tensor([0.9, 0.1]),
            },
            {
                "edit_id": "e2",
                "cell_id": "c1",
                "prompt": "p2",
                "raw_semantic_key": torch.tensor([0.0, 1.0]),
                "raw_activation_key": torch.tensor([0.0, 1.0]),
                "semantic_key": torch.tensor([0.0, 1.0]),
                "activation_key": torch.tensor([0.0, 1.0]),
            },
        ]
        summary = summarize_conflicts(entries, semantic_weight=0.7, activation_weight=0.3)
        self.assertEqual(summary["num_cells"], 2)
        self.assertIsNotNone(summary["within_cell_conflict_mean"])
        self.assertEqual(len(summary["cell_summary"]), 2)

    def test_cross_view_route_gap_is_reported(self):
        annotated_logs = [
            {"event_type": "rewrite", "chosen_memory_id": "c0", "correct_route": True, "top1_prob": 0.9, "route_margin": 0.8, "route_stage": "ambiguity_rerank"},
            {"event_type": "rewrite", "chosen_memory_id": "c1", "correct_route": True, "top1_prob": 0.8, "route_margin": 0.7, "route_stage": "ambiguity_rerank"},
            {"event_type": "rephrase", "chosen_memory_id": "c0", "correct_route": False, "top1_prob": 0.5, "route_margin": 0.1, "route_stage": "ambiguity_rerank"},
            {"event_type": "rephrase", "chosen_memory_id": None, "correct_route": False, "top1_prob": 0.2, "route_margin": 0.0, "route_stage": "ambiguity_rerank"},
        ]
        metrics = [
            {"case_id": 0, "requested_rewrite": {"subject": "a", "prompt": "p"}, "pre": {}, "post": {}},
            {"case_id": 1, "requested_rewrite": {"subject": "b", "prompt": "q"}, "pre": {}, "post": {}},
        ]
        summary = summarize_route_diagnostics(annotated_logs, metrics)
        self.assertIn("cross_view", summary)
        self.assertAlmostEqual(summary["cross_view"]["cross_view_route_gap"], 1.0, places=4)
        self.assertEqual(summary["summary"]["route_stage_counts"]["ambiguity_rerank"], 4)


if __name__ == "__main__":
    unittest.main()
