import json
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from easyeditor.models.hopedit.hopedit_main import HopEditController


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self._vocab = {"<pad>": self.pad_token_id, "<eos>": self.eos_token_id}
        self._inverse_vocab = {self.pad_token_id: "<pad>", self.eos_token_id: "<eos>"}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for token in str(text).split():
            if token not in self._vocab:
                token_id = len(self._vocab)
                self._vocab[token] = token_id
                self._inverse_vocab[token_id] = token
            ids.append(self._vocab[token])
        if add_special_tokens:
            ids = ids + [self.eos_token_id]
        return ids

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=32, add_special_tokens=True):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            token_ids = self.encode(text, add_special_tokens=add_special_tokens)
            if not token_ids:
                token_ids = [self.eos_token_id]
            rows.append(torch.tensor(token_ids[:max_length], dtype=torch.long))
        max_len = max(row.numel() for row in rows)
        input_ids = torch.full((len(rows), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        for idx, row in enumerate(rows):
            input_ids[idx, : row.numel()] = row
            attention_mask[idx, : row.numel()] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, input_ids, skip_special_tokens=True):
        rows = input_ids if not torch.is_tensor(input_ids) else input_ids.detach().cpu().tolist()
        decoded = []
        for row in rows:
            tokens = []
            for token_id in row:
                if token_id == self.pad_token_id:
                    continue
                if skip_special_tokens and token_id == self.eos_token_id:
                    continue
                tokens.append(self._inverse_vocab.get(int(token_id), "unk"))
            decoded.append(" ".join(tokens))
        return decoded


class LeftPadDummyTokenizer(DummyTokenizer):
    def __call__(
        self,
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
        add_special_tokens=True,
        return_special_tokens_mask=False,
    ):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            token_ids = self.encode(text, add_special_tokens=add_special_tokens)
            if not token_ids:
                token_ids = [self.eos_token_id]
            rows.append(torch.tensor(token_ids[:max_length], dtype=torch.long))
        max_len = max(row.numel() for row in rows)
        input_ids = torch.full((len(rows), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        special_tokens_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        for idx, row in enumerate(rows):
            start = max_len - row.numel()
            input_ids[idx, start:] = row
            attention_mask[idx, start:] = 1
            if add_special_tokens and row.numel() > 0 and int(row[-1].item()) == self.eos_token_id:
                special_tokens_mask[idx, max_len - 1] = 1
        result = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_special_tokens_mask:
            result["special_tokens_mask"] = special_tokens_mask
        return result


def make_hparams():
    return SimpleNamespace(
        rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        target_modules=["c_attn", "c_proj"],
        num_steps=1,
        lr=1.0e-3,
        weight_decay=0.0,
        top_k=2,
        beta=1.0,
        semantic_weight=0.7,
        activation_weight=0.3,
        probe_layer=0,
        route_log_dir="",
        device=0,
        alg_name="HOPEDIT",
        model_name="dummy-gpt2",
        probe_layers=[0],
        route_strategy="multiview",
        min_route_prob=0.0,
        min_route_margin=0.0,
        fallback_min_route_prob=0.0,
        fallback_min_route_margin=0.0,
        activation_center=True,
        activation_whiten=True,
        key_l2_normalize=True,
        use_rephrase_prompt=True,
        use_subject_prompt=True,
        negative_top_k=0,
        negative_weight=0.0,
        negative_margin=0.2,
        negative_min_conflict=0.0,
        log_training_loss=False,
        loss_log_every=1,
        batch_size=1,
        max_length=32,
        model_parallel=False,
        use_chat_template=False,
        fp16=False,
        bf16=False,
        save_path=None,
        hopedit_mode="v2_cell_bank",
        cell_budget=4,
        cell_assignment_policy="conflict_aware",
        cell_conflict_threshold=0.65,
        cell_merge_rule="weighted_delta_average",
        cell_router_topk=2,
        cell_min_assignment_margin=0.0,
        cell_metadata_backend="inline_json",
        cell_prototype_strategy="multiview",
        cell_prototype_rerank_topk=2,
        cell_rerank_dispersion_penalty=0.05,
        cell_direct_accept_min_prob=0.52,
        cell_direct_accept_min_margin=0.05,
        state_memory_impl="cell_bank",
        state_slot_capacity=8,
        slot_rank=2,
        state_basis_rank=4,
        slot_realization_impl="concatenated_lora",
        slot_topk=2,
        atom_sparse_topk=2,
        atom_write_topk=2,
        atom_coherence_penalty=0.25,
        atom_support_exclusivity_penalty=0.35,
        atom_sparse_ridge=1.0e-4,
        atom_sparse_min_abs_affinity=1.0e-6,
        slot_activation_min_weight=0.15,
        state_finalist_margin=0.05,
        locality_risk_threshold=0.35,
        memory_tier_warmup_edits=4,
        stability_min_observations=2,
        stability_max_cross_view_gap=0.10,
        stability_max_prototype_dispersion=0.30,
        stability_max_within_state_conflict=0.30,
        stability_min_locality=0.95,
        consolidation_interval_edits=2,
        consolidation_max_pairs_per_pass=4,
        hierarchy_enable=True,
        hierarchy_start_edit=4,
        bucket_topk=2,
        bucket_max_size=2,
        bucket_split_dispersion=0.35,
        atoms_per_shard=8,
        atom_rank=2,
        read_support_topk=2,
        write_support_topk=2,
        max_new_atoms_per_edit=1,
        module_grouping="attention",
        side_memory_shard_shortlist=2,
        side_memory_energy_margin=0.05,
        support_usage_penalty=0.35,
        support_overlap_veto=0.75,
        new_shard_residual_threshold=1.0,
        side_memory_write_steps=1,
        side_memory_loss_threshold=2.5,
        base_only_fallback_threshold=0.0,
        shard_budget_microbench=8,
        address_encoder_impl="deterministic_topk",
        address_num_atoms=16,
        address_code_topk=2,
        address_candidate_budget=8,
        address_atom_merge_threshold=0.92,
        address_code_min_affinity=1.0e-6,
        address_coherence_penalty=0.05,
        trace_energy_beta=12.0,
        trace_energy_temperature=1.0,
        trace_abstain_margin=0.03,
        trace_abstain_min_energy=0.10,
        trace_family_budget=4,
        trace_family_boost=0.35,
        trace_anchor_weight=0.25,
        trace_anchor_energy_floor=0.05,
        trace_family_negative_topk=4,
        trace_irrelevant_negative_topk=2,
        trace_family_margin_scale=0.60,
        trace_anchor_margin_scale=0.60,
        trace_value_cache_size=4,
        factored_address_layer=None,
        factored_subject_weight=0.5,
        factored_relation_weight=0.5,
        factored_subject_margin_threshold=0.03,
        factored_relation_margin_threshold=0.03,
        factored_subject_energy_threshold=0.0,
        factored_relation_energy_threshold=0.0,
        factored_use_subject_metadata=True,
        factored_subject_resolution="metadata_or_substring",
        factored_subject_pooling="last",
        log_full_factor_scores=False,
        factored_relation_encoder_impl="identity",
        factored_relation_encoder_hidden_dim=8,
        factored_relation_encoder_steps=0,
        factored_relation_encoder_lr=1.0e-3,
        factored_relation_encoder_temperature=0.05,
        factored_relation_encoder_relation_weight=0.25,
        factored_relation_encoder_min_examples=4,
        factored_relation_encoder_rebuild_on_train=True,
        factored_relation_encoder_checkpoint=None,
        factored_relation_encoder_freeze_checkpoint=True,
        factored_relation_match_rule="top1_same_trace",
        factored_relation_exclude_same_relation_id_from_margin=True,
        state_gate_enable=False,
        state_gate_model="logistic",
        state_gate_warm_start_from_replay=False,
        state_gate_replay_buffer_size=128,
        state_gate_online_update_interval=2,
        state_gate_lr=1.0e-2,
        state_gate_batch_size=8,
        state_gate_warm_start_epochs=1,
        state_gate_direct_threshold=0.70,
        state_gate_promotion_threshold=0.80,
        state_gate_merge_threshold=0.75,
    )


class HopEditV2CheckpointTests(unittest.TestCase):
    def test_v41_sparse_trace_bank_routes_by_sparse_address(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_sparse_address_trace_bank"
        hparams.route_strategy = "multiview"
        hparams.trace_abstain_margin = 0.0
        hparams.trace_abstain_min_energy = 0.0
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller.memory_entries = [
            {
                "edit_id": "hopedit_00000",
                "prompt": "p0",
                "raw_semantic_key": torch.tensor([1.0, 0.0]),
                "raw_activation_key": torch.tensor([1.0, 0.0]),
                "semantic_key": torch.tensor([1.0, 0.0]),
                "activation_key": torch.tensor([1.0, 0.0]),
                "trace_id": "hopedit_00000",
                "trace_value_impl": "exact_lora",
                "value_adapter_name": "hopedit_00000",
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": "p0",
                        "raw_semantic_key": torch.tensor([1.0, 0.0]),
                        "raw_activation_key": torch.tensor([1.0, 0.0]),
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                    },
                    {
                        "view_name": "rephrase",
                        "text": "p0-r",
                        "raw_semantic_key": torch.tensor([0.9, 0.1]),
                        "raw_activation_key": torch.tensor([0.9, 0.1]),
                        "semantic_key": torch.tensor([0.9, 0.1]),
                        "activation_key": torch.tensor([0.9, 0.1]),
                    },
                ],
            },
            {
                "edit_id": "hopedit_00001",
                "prompt": "p1",
                "raw_semantic_key": torch.tensor([0.0, 1.0]),
                "raw_activation_key": torch.tensor([0.0, 1.0]),
                "semantic_key": torch.tensor([0.0, 1.0]),
                "activation_key": torch.tensor([0.0, 1.0]),
                "trace_id": "hopedit_00001",
                "trace_value_impl": "exact_lora",
                "value_adapter_name": "hopedit_00001",
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": "p1",
                        "raw_semantic_key": torch.tensor([0.0, 1.0]),
                        "raw_activation_key": torch.tensor([0.0, 1.0]),
                        "semantic_key": torch.tensor([0.0, 1.0]),
                        "activation_key": torch.tensor([0.0, 1.0]),
                    }
                ],
            },
        ]
        controller.edit_registry = {entry["edit_id"]: entry for entry in controller.memory_entries}
        controller._refresh_processed_memory_keys()
        self.assertIsNotNone(controller.memory_entries[0].get("trace_min_activation_energy"))
        self.assertIsNotNone(controller.memory_entries[0].get("trace_min_activation_margin"))
        decision = controller._route_from_keys_v4(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        self.assertEqual(decision["chosen_memory_id"], "hopedit_00000")
        self.assertFalse(decision["address_abstained"])
        self.assertGreaterEqual(decision["candidate_set_size"], 1)
        self.assertGreater(decision["address_support_size"], 0)

    def test_v41_sparse_trace_bank_abstains_on_ambiguous_address(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_sparse_address_trace_bank"
        hparams.route_strategy = "multiview"
        hparams.trace_abstain_margin = 0.20
        hparams.trace_abstain_min_energy = 0.0
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller.memory_entries = [
            {
                "edit_id": "hopedit_00000",
                "prompt": "p0",
                "raw_semantic_key": torch.tensor([1.0, 0.0]),
                "raw_activation_key": torch.tensor([1.0, 0.0]),
                "semantic_key": torch.tensor([1.0, 0.0]),
                "activation_key": torch.tensor([1.0, 0.0]),
                "trace_id": "hopedit_00000",
                "trace_value_impl": "exact_lora",
                "value_adapter_name": "hopedit_00000",
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": "p0",
                        "raw_semantic_key": torch.tensor([1.0, 0.0]),
                        "raw_activation_key": torch.tensor([1.0, 0.0]),
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                    }
                ],
            },
            {
                "edit_id": "hopedit_00001",
                "prompt": "p1",
                "raw_semantic_key": torch.tensor([1.0, 0.0]),
                "raw_activation_key": torch.tensor([1.0, 0.0]),
                "semantic_key": torch.tensor([1.0, 0.0]),
                "activation_key": torch.tensor([1.0, 0.0]),
                "trace_id": "hopedit_00001",
                "trace_value_impl": "exact_lora",
                "value_adapter_name": "hopedit_00001",
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": "p1",
                        "raw_semantic_key": torch.tensor([1.0, 0.0]),
                        "raw_activation_key": torch.tensor([1.0, 0.0]),
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                    }
                ],
            },
        ]
        controller.edit_registry = {entry["edit_id"]: entry for entry in controller.memory_entries}
        controller._refresh_processed_memory_keys()
        decision = controller._route_from_keys_v4(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        self.assertIsNone(decision["chosen_memory_id"])
        self.assertTrue(decision["address_abstained"])
        self.assertTrue(decision["base_only_fallback"])

    def test_v41_sparse_trace_bank_checkpoint_round_trip(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_sparse_address_trace_bank"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller._ensure_adapter("hopedit_00000")
        controller.memory_entries = [
            {
                "edit_id": "hopedit_00000",
                "prompt": "p0",
                "raw_semantic_key": torch.tensor([1.0, 0.0]),
                "raw_activation_key": torch.tensor([1.0, 0.0]),
                "semantic_key": torch.tensor([1.0, 0.0]),
                "activation_key": torch.tensor([1.0, 0.0]),
                "trace_id": "hopedit_00000",
                "trace_value_impl": "exact_lora",
                "value_adapter_name": "hopedit_00000",
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": "p0",
                        "raw_semantic_key": torch.tensor([1.0, 0.0]),
                        "raw_activation_key": torch.tensor([1.0, 0.0]),
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                    }
                ],
            }
        ]
        controller.edit_registry = {"hopedit_00000": controller.memory_entries[0]}
        controller._refresh_processed_memory_keys()
        with tempfile.TemporaryDirectory() as tmpdir:
            controller.save_runtime_checkpoint(tmpdir)
            restored = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=make_hparams())
            restored.hparams.hopedit_mode = "v4_sparse_address_trace_bank"
            restored.load_runtime_checkpoint(tmpdir, is_trainable=True)
            self.assertEqual(restored.hparams.hopedit_mode, "v4_sparse_address_trace_bank")
            self.assertTrue(len(restored.address_dictionary.get("atoms") or []) >= 1)
            self.assertIn(0, restored.address_postings)
            self.assertEqual(restored.memory_entries[0]["trace_address_impl"], "sparse_dictionary")
            self.assertTrue(len(restored.memory_entries[0]["trace_address_support"]) >= 1)
            self.assertIsNotNone(restored.memory_entries[0].get("trace_min_activation_energy"))
            self.assertIsNotNone(restored.memory_entries[0].get("trace_min_activation_margin"))

    def test_v41_sparse_trace_bank_builds_exclusion_support_from_fresh_dictionary(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_sparse_address_trace_bank"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller.memory_entries = [
            {
                "edit_id": "hopedit_00000",
                "prompt": "p0",
                "raw_semantic_key": torch.tensor([1.0, 0.0]),
                "raw_activation_key": torch.tensor([1.0, 0.0]),
                "semantic_key": torch.tensor([1.0, 0.0]),
                "activation_key": torch.tensor([1.0, 0.0]),
                "trace_id": "hopedit_00000",
                "trace_value_impl": "exact_lora",
                "value_adapter_name": "hopedit_00000",
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": "p0",
                        "raw_semantic_key": torch.tensor([1.0, 0.0]),
                        "raw_activation_key": torch.tensor([1.0, 0.0]),
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                    }
                ],
            },
            {
                "edit_id": "hopedit_00001",
                "prompt": "p1",
                "raw_semantic_key": torch.tensor([0.0, 1.0]),
                "raw_activation_key": torch.tensor([0.0, 1.0]),
                "semantic_key": torch.tensor([0.0, 1.0]),
                "activation_key": torch.tensor([0.0, 1.0]),
                "trace_id": "hopedit_00001",
                "trace_value_impl": "exact_lora",
                "value_adapter_name": "hopedit_00001",
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": "p1",
                        "raw_semantic_key": torch.tensor([0.0, 1.0]),
                        "raw_activation_key": torch.tensor([0.0, 1.0]),
                        "semantic_key": torch.tensor([0.0, 1.0]),
                        "activation_key": torch.tensor([0.0, 1.0]),
                    }
                ],
            },
        ]
        controller.edit_registry = {entry["edit_id"]: entry for entry in controller.memory_entries}
        controller.address_dictionary = {"atoms": []}
        controller._refresh_processed_memory_keys()
        for entry in controller.memory_entries:
            self.assertTrue(len(entry.get("trace_exclusion_support") or []) >= 1)
            self.assertIsNotNone(entry.get("trace_negative_exclusion_score_floor"))

    def test_v41_sparse_trace_bank_locality_veto_abstains(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_sparse_address_trace_bank"
        hparams.trace_abstain_margin = 0.0
        hparams.trace_abstain_min_energy = 0.0
        hparams.trace_use_locality_veto = True
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        ranking = [
            {
                "entry": {
                    "edit_id": "hopedit_00000",
                    "trace_min_activation_energy": 0.0,
                    "trace_min_activation_margin": 0.0,
                    "trace_max_exclusion_score": 0.10,
                    "trace_address_agreement": 1.0,
                },
                "trace_energy": 0.95,
                "trace_exclusion_score": 0.15,
                "best_view_name": "prompt",
                "address_support_overlap": 1.0,
                "cross_view_code_agreement": 1.0,
            }
        ]
        decision = controller._sparse_trace_decision_from_ranking(
            ranking,
            torch.zeros(4, dtype=torch.float32),
            [0, 1],
            1,
            "trace_sparse_address_multiview",
        )
        self.assertIsNone(decision["chosen_memory_id"])
        self.assertTrue(decision["address_abstained"])
        self.assertTrue(decision["address_locality_vetoed"])
        self.assertAlmostEqual(decision["trace_locality_margin_threshold"], 0.10)
        self.assertAlmostEqual(decision["trace_exclusion_score"], 0.15)

    def test_v42_overlap_trace_bank_builds_family_and_anchor_state(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_overlap_aware_anchor_trace_bank"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller.memory_entries = [
            {
                "edit_id": "hopedit_00000",
                "prompt": "What is the twin city of Alpha? It is",
                "subject": "Alpha",
                "target_new": "Beta",
                "raw_semantic_key": torch.tensor([1.0, 0.0]),
                "raw_activation_key": torch.tensor([1.0, 0.0]),
                "semantic_key": torch.tensor([1.0, 0.0]),
                "activation_key": torch.tensor([1.0, 0.0]),
                "trace_id": "hopedit_00000",
                "trace_value_impl": "exact_lora_cold_store",
                "value_ref": "hopedit_00000",
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": "What is the twin city of Alpha? It is",
                        "raw_semantic_key": torch.tensor([1.0, 0.0]),
                        "raw_activation_key": torch.tensor([1.0, 0.0]),
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                    },
                    {
                        "view_name": "rephrase",
                        "text": "Alpha has a twin city named",
                        "raw_semantic_key": torch.tensor([0.9, 0.1]),
                        "raw_activation_key": torch.tensor([0.9, 0.1]),
                        "semantic_key": torch.tensor([0.9, 0.1]),
                        "activation_key": torch.tensor([0.9, 0.1]),
                    },
                ],
            },
            {
                "edit_id": "hopedit_00001",
                "prompt": "What is the twin city of Gamma? It is",
                "subject": "Gamma",
                "target_new": "Delta",
                "raw_semantic_key": torch.tensor([0.0, 1.0]),
                "raw_activation_key": torch.tensor([0.0, 1.0]),
                "semantic_key": torch.tensor([0.0, 1.0]),
                "activation_key": torch.tensor([0.0, 1.0]),
                "trace_id": "hopedit_00001",
                "trace_value_impl": "exact_lora_cold_store",
                "value_ref": "hopedit_00001",
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": "What is the twin city of Gamma? It is",
                        "raw_semantic_key": torch.tensor([0.0, 1.0]),
                        "raw_activation_key": torch.tensor([0.0, 1.0]),
                        "semantic_key": torch.tensor([0.0, 1.0]),
                        "activation_key": torch.tensor([0.0, 1.0]),
                    }
                ],
            },
        ]
        controller.edit_registry = {entry["edit_id"]: entry for entry in controller.memory_entries}
        controller._refresh_processed_memory_keys()
        self.assertTrue(controller.family_postings)
        self.assertIn("template::what is the twin city of <subj>? it is", controller.family_postings)
        self.assertTrue(len(controller.memory_entries[0].get("trace_anchor_views") or []) >= 1)
        self.assertTrue(len(controller.memory_entries[0].get("trace_family_ids") or []) >= 1)
        self.assertTrue(len(controller.memory_entries[0].get("trace_family_negative_trace_ids") or []) >= 1)

    def test_v42_overlap_trace_bank_two_pass_codes_align_dimensions(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_overlap_aware_anchor_trace_bank"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        entry = {
            "edit_id": "hopedit_00000",
            "prompt": "Prompt Alpha",
            "subject": "Alpha",
            "target_new": "Beta",
            "view_keys": [
                {
                    "view_name": "prompt",
                    "text": "Prompt Alpha",
                    "semantic_key": torch.tensor([1.0, 0.0]),
                    "activation_key": torch.tensor([1.0, 0.0]),
                },
                {
                    "view_name": "rephrase",
                    "text": "Paraphrase Alpha",
                    "semantic_key": torch.tensor([0.0, 1.0]),
                    "activation_key": torch.tensor([0.0, 1.0]),
                },
            ],
        }
        atoms = []
        usage_counts = []
        controller._assign_overlap_trace_address_state(entry, atoms, usage_counts, update_atoms=True, address_version=1)
        self.assertEqual(len(atoms), 2)
        code_sizes = [int(view["address_code"].numel()) for view in entry["view_keys"]]
        self.assertEqual(code_sizes, [2, 2])
        self.assertEqual(sorted(entry["trace_address_support"]), [0, 1])
        self.assertIsNotNone(entry.get("trace_address_agreement"))

    def test_v42_overlap_trace_bank_refreshes_non_family_hard_negatives(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_overlap_aware_anchor_trace_bank"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)

        def make_entry(edit_id, prompt, subject, target_new, key):
            return {
                "edit_id": edit_id,
                "prompt": prompt,
                "subject": subject,
                "target_new": target_new,
                "raw_semantic_key": key.clone(),
                "raw_activation_key": key.clone(),
                "semantic_key": key.clone(),
                "activation_key": key.clone(),
                "trace_id": edit_id,
                "trace_value_impl": "exact_lora_cold_store",
                "value_ref": edit_id,
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": prompt,
                        "raw_semantic_key": key.clone(),
                        "raw_activation_key": key.clone(),
                        "semantic_key": key.clone(),
                        "activation_key": key.clone(),
                    }
                ],
            }

        entry_a = make_entry("hopedit_00000", "What relation does Alpha have?", "Alpha", "Beta", torch.tensor([1.0, 0.0]))
        controller.edit_registry[entry_a["edit_id"]] = entry_a
        controller.memory_entries.append(entry_a)
        controller._refresh_single_trace_entry_keys(entry_a)
        controller._index_overlap_anchor_trace_entry(entry_a)

        entry_b = make_entry("hopedit_00001", "Who discovered Gamma?", "Gamma", "Delta", torch.tensor([0.98, 0.02]))
        controller.edit_registry[entry_b["edit_id"]] = entry_b
        controller.memory_entries.append(entry_b)
        controller._refresh_single_trace_entry_keys(entry_b)
        controller._index_overlap_anchor_trace_entry(entry_b)

        refreshed_a = controller.edit_registry["hopedit_00000"]
        negative_ids = set(refreshed_a.get("trace_family_negative_trace_ids") or []) | set(
            refreshed_a.get("trace_irrelevant_negative_trace_ids") or []
        )
        self.assertIn("hopedit_00001", negative_ids)
        self.assertIn(
            "hopedit_00001",
            [row.get("edit_id") for row in refreshed_a.get("conflict_neighbors") or []],
        )

    def test_v42_overlap_trace_bank_family_margin_can_abstain(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_overlap_aware_anchor_trace_bank"
        hparams.trace_abstain_margin = 0.0
        hparams.trace_abstain_min_energy = 0.0
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        ranking = [
            {
                "entry": {
                    "edit_id": "hopedit_00000",
                    "trace_min_activation_energy": 0.0,
                    "trace_min_activation_margin": 0.0,
                    "trace_min_anchor_energy": 0.0,
                    "trace_min_family_margin": 0.20,
                    "trace_address_agreement": 1.0,
                },
                "combined_conflict": 1.00,
                "trace_energy": 0.95,
                "trace_anchor_energy": 0.80,
                "trace_family_margin": 0.10,
                "best_view_name": "prompt",
                "address_support_overlap": 1.0,
                "cross_view_code_agreement": 1.0,
                "family_shortlist_size": 1,
            }
        ]
        decision = controller._sparse_trace_decision_from_ranking(
            ranking,
            torch.zeros(4, dtype=torch.float32),
            [0, 1],
            1,
            "trace_overlap_anchor_multiview",
        )
        self.assertIsNone(decision["chosen_memory_id"])
        self.assertTrue(decision["address_abstained"])
        self.assertTrue(decision["address_locality_vetoed"])
        self.assertAlmostEqual(decision["trace_family_margin_threshold"], 0.20)

    def test_v42_overlap_trace_bank_anchor_route_uses_anchor_views_only(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_overlap_aware_anchor_trace_bank"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        entry = {
            "edit_id": "hopedit_00000",
            "prompt": "Prompt view",
            "subject": "Alpha",
            "target_new": "Beta",
            "raw_semantic_key": torch.tensor([1.0, 0.0]),
            "raw_activation_key": torch.tensor([1.0, 0.0]),
            "semantic_key": torch.tensor([1.0, 0.0]),
            "activation_key": torch.tensor([1.0, 0.0]),
            "trace_id": "hopedit_00000",
            "trace_value_impl": "exact_lora_cold_store",
            "value_ref": "hopedit_00000",
            "view_keys": [
                {
                    "view_name": "prompt",
                    "text": "Prompt view",
                    "raw_semantic_key": torch.tensor([1.0, 0.0]),
                    "raw_activation_key": torch.tensor([1.0, 0.0]),
                    "semantic_key": torch.tensor([1.0, 0.0]),
                    "activation_key": torch.tensor([1.0, 0.0]),
                },
                {
                    "view_name": "rephrase",
                    "text": "Rephrase view",
                    "raw_semantic_key": torch.tensor([0.0, 1.0]),
                    "raw_activation_key": torch.tensor([0.0, 1.0]),
                    "semantic_key": torch.tensor([0.0, 1.0]),
                    "activation_key": torch.tensor([0.0, 1.0]),
                },
                {
                    "view_name": "subject",
                    "text": "Subject view",
                    "raw_semantic_key": torch.tensor([0.0, 1.0]),
                    "raw_activation_key": torch.tensor([0.0, 1.0]),
                    "semantic_key": torch.tensor([0.0, 1.0]),
                    "activation_key": torch.tensor([0.0, 1.0]),
                },
            ],
        }
        controller.edit_registry = {entry["edit_id"]: entry}
        controller.memory_entries = [entry]
        controller._refresh_processed_memory_keys()

        full_ranking, _, _, _ = controller._rank_sparse_trace_addresses(
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 1.0]),
        )
        anchor_ranking, _, _, _ = controller._rank_sparse_trace_addresses(
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 1.0]),
            allowed_view_names=controller._anchor_view_names(),
        )
        self.assertTrue(full_ranking)
        self.assertTrue(anchor_ranking)
        self.assertEqual(full_ranking[0]["best_view_name"], "rephrase")
        self.assertIn(anchor_ranking[0]["best_view_name"], {"prompt", "rephrase"})
        self.assertNotEqual(anchor_ranking[0]["best_view_name"], "subject")

    def test_v42_overlap_trace_bank_value_cache_checkpoint_round_trip(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_overlap_aware_anchor_trace_bank"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller._ensure_adapter("hopedit_00000")
        controller._store_trace_value_reference("hopedit_00000", "hopedit_00000")
        controller._drop_runtime_adapter_without_disabling("hopedit_00000")
        controller.memory_entries = [
            {
                "edit_id": "hopedit_00000",
                "prompt": "p0",
                "subject": "s0",
                "target_new": "t0",
                "raw_semantic_key": torch.tensor([1.0, 0.0]),
                "raw_activation_key": torch.tensor([1.0, 0.0]),
                "semantic_key": torch.tensor([1.0, 0.0]),
                "activation_key": torch.tensor([1.0, 0.0]),
                "trace_id": "hopedit_00000",
                "trace_value_impl": "exact_lora_cold_store",
                "value_ref": "hopedit_00000",
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": "p0",
                        "raw_semantic_key": torch.tensor([1.0, 0.0]),
                        "raw_activation_key": torch.tensor([1.0, 0.0]),
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                    }
                ],
            }
        ]
        controller.edit_registry = {"hopedit_00000": controller.memory_entries[0]}
        controller._refresh_processed_memory_keys()
        adapter_name, cache_hit = controller._materialize_trace_value("hopedit_00000")
        self.assertEqual(adapter_name, "hopedit_00000")
        self.assertFalse(cache_hit)
        adapter_name, cache_hit = controller._materialize_trace_value("hopedit_00000")
        self.assertTrue(cache_hit)
        with tempfile.TemporaryDirectory() as tmpdir:
            controller.save_runtime_checkpoint(tmpdir)
            restored = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=make_hparams())
            restored.hparams.hopedit_mode = "v4_overlap_aware_anchor_trace_bank"
            with mock.patch.object(restored, "_refresh_processed_memory_keys", side_effect=AssertionError("unexpected full rebuild")):
                restored.load_runtime_checkpoint(tmpdir, is_trainable=True)
            self.assertIn("hopedit_00000", restored.trace_value_store)
            self.assertEqual(len(restored.trace_value_cache), 0)
            adapter_name, cache_hit = restored._materialize_trace_value("hopedit_00000")
            self.assertEqual(adapter_name, "hopedit_00000")
            self.assertFalse(cache_hit)

    def test_v42_overlap_trace_bank_extract_keys_after_dropping_last_adapter(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_overlap_aware_anchor_trace_bank"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller._ensure_adapter("hopedit_00000")
        controller._store_trace_value_reference("hopedit_00000", "hopedit_00000")
        controller._drop_runtime_adapter_without_disabling("hopedit_00000")
        self.assertEqual(getattr(controller.model, "active_adapter", None), None)
        semantic_key, activation_key = controller._extract_keys(["alpha prompt"], ["alpha prompt"])
        self.assertIsInstance(semantic_key, torch.Tensor)
        self.assertIsInstance(activation_key, torch.Tensor)
        self.assertEqual(semantic_key.ndim, 1)
        self.assertEqual(activation_key.ndim, 1)

    def test_v42_overlap_trace_bank_memory_snapshot_is_json_serializable(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_overlap_aware_anchor_trace_bank"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        entry = {
            "edit_id": "hopedit_00000",
            "prompt": "What is the twin city of Alpha? It is",
            "subject": "Alpha",
            "rephrase_prompt": "Alpha has a twin city named",
            "target_new": "Beta",
            "raw_semantic_key": torch.tensor([1.0, 0.0]),
            "raw_activation_key": torch.tensor([1.0, 0.0]),
            "semantic_key": torch.tensor([1.0, 0.0]),
            "activation_key": torch.tensor([1.0, 0.0]),
            "trace_id": "hopedit_00000",
            "trace_value_impl": "exact_lora_cold_store",
            "value_ref": "hopedit_00000",
            "view_keys": [
                {
                    "view_name": "prompt",
                    "text": "What is the twin city of Alpha? It is",
                    "raw_semantic_key": torch.tensor([1.0, 0.0]),
                    "raw_activation_key": torch.tensor([1.0, 0.0]),
                    "semantic_key": torch.tensor([1.0, 0.0]),
                    "activation_key": torch.tensor([1.0, 0.0]),
                },
                {
                    "view_name": "rephrase",
                    "text": "Alpha has a twin city named",
                    "raw_semantic_key": torch.tensor([0.9, 0.1]),
                    "raw_activation_key": torch.tensor([0.9, 0.1]),
                    "semantic_key": torch.tensor([0.9, 0.1]),
                    "activation_key": torch.tensor([0.9, 0.1]),
                },
            ],
        }
        controller.edit_registry = {entry["edit_id"]: entry}
        controller.memory_entries = [entry]
        controller._refresh_processed_memory_keys()
        snapshot = controller.export_memory_snapshot(include_keys=False)
        json.dumps(snapshot)
        self.assertEqual(snapshot[0]["trace_value_impl"], "exact_lora_cold_store")
        self.assertIsInstance(snapshot[0]["trace_anchor_views"], list)

    def test_v43_factored_trace_bank_builds_subject_and_relation_factors(self):
        config = GPT2Config(
            vocab_size=128,
            n_positions=64,
            n_ctx=64,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_factored_address_trace_bank"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        request = {
            "prompt": "Where was Alpha founded?",
            "subject": "Alpha",
            "target_new": "Paris",
            "rephrase_prompt": "Alpha was founded in",
        }
        views = controller._build_view_key_records(request, request["prompt"], request["prompt"])
        self.assertEqual([view["view_name"] for view in views], ["prompt", "rephrase"])
        self.assertTrue(all(isinstance(view.get("subject_factor"), torch.Tensor) for view in views))
        self.assertTrue(all(isinstance(view.get("relation_factor"), torch.Tensor) for view in views))
        self.assertTrue(all(bool(view.get("subject_found")) for view in views))
        self.assertTrue(all(int(view.get("relation_token_count") or 0) >= 1 for view in views))

    def test_v43_factored_key_extraction_supports_subject_layer_and_pooling_overrides(self):
        config = GPT2Config(
            vocab_size=128,
            n_positions=64,
            n_ctx=64,
            n_embd=16,
            n_layer=2,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_factored_address_trace_bank"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        rows = controller._extract_batched_factored_address_keys(
            ["Where was Alpha founded?"],
            ["Alpha"],
            ["Paris"],
            subject_layer_override=0,
            relation_layer_override=1,
            subject_pooling_override="mean",
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["subject_layer_index"], 0)
        self.assertEqual(row["relation_layer_index"], 1)
        self.assertEqual(row["subject_pooling_mode"], "mean")
        self.assertIsInstance(row["subject_factor"], torch.Tensor)
        self.assertIsInstance(row["relation_factor"], torch.Tensor)

    def test_v43_factored_key_extraction_handles_left_padded_batches(self):
        config = GPT2Config(
            vocab_size=128,
            n_positions=64,
            n_ctx=64,
            n_embd=16,
            n_layer=2,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_factored_address_trace_bank"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=LeftPadDummyTokenizer(), hparams=hparams)
        rows = controller._extract_batched_factored_address_keys(
            [
                "Alpha founded",
                "The organization Alpha was founded in Paris after the war",
            ],
            ["Alpha", "Alpha"],
            [None, "Paris"],
            subject_layer_override=0,
            relation_layer_override=1,
            subject_pooling_override="last",
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["subject_found"])
        self.assertTrue(rows[1]["subject_found"])
        self.assertEqual(rows[0]["relation_token_count"], 1)
        self.assertEqual(rows[1]["relation_token_count"], 8)
        self.assertIsInstance(rows[0]["subject_factor"], torch.Tensor)
        self.assertIsInstance(rows[0]["relation_factor"], torch.Tensor)

    def test_v43_factored_trace_bank_hard_and_routes_common_trace(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_factored_address_trace_bank"
        hparams.factored_subject_margin_threshold = 0.0
        hparams.factored_relation_margin_threshold = 0.0
        hparams.factored_subject_energy_threshold = 0.0
        hparams.factored_relation_energy_threshold = 0.0
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)

        def make_entry(edit_id, prompt, subject, target_new, subj_factor, rel_factor):
            return {
                "edit_id": edit_id,
                "prompt": prompt,
                "subject": subject,
                "target_new": target_new,
                "raw_semantic_key": subj_factor.clone(),
                "raw_activation_key": rel_factor.clone(),
                "semantic_key": subj_factor.clone(),
                "activation_key": rel_factor.clone(),
                "trace_id": edit_id,
                "trace_value_impl": "exact_lora_cold_store",
                "value_ref": edit_id,
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": prompt,
                        "raw_semantic_key": subj_factor.clone(),
                        "raw_activation_key": rel_factor.clone(),
                        "semantic_key": subj_factor.clone(),
                        "activation_key": rel_factor.clone(),
                        "subject_factor": subj_factor.clone(),
                        "relation_factor": rel_factor.clone(),
                        "subject_found": True,
                        "relation_token_count": 2,
                    }
                ],
            }

        entry_a = make_entry("hopedit_00000", "Where was Alpha founded?", "Alpha", "Paris", torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        entry_b = make_entry("hopedit_00001", "Where was Beta founded?", "Beta", "Rome", torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
        controller.edit_registry = {entry_a["edit_id"]: entry_a, entry_b["edit_id"]: entry_b}
        controller.memory_entries = [entry_a, entry_b]
        controller._rebuild_factored_trace_address_state()

        query = {
            "subject_factor": torch.tensor([1.0, 0.0]),
            "relation_factor": torch.tensor([1.0, 0.0]),
        }
        query["combined_vector"] = controller._factored_combined_address_vector(query["subject_factor"], query["relation_factor"])
        query["query_code"] = controller._encode_sparse_address(query["combined_vector"], controller.address_dictionary["atoms"])
        query["query_support"] = controller._address_support_from_code(query["query_code"])
        query["resolved_subject"] = "Alpha"
        query["subject_found"] = True
        query["relation_token_count"] = 2

        decision = controller._route_from_factored_query(query)
        self.assertEqual(decision["chosen_memory_id"], "hopedit_00000")
        self.assertTrue(decision["factor_same_trace"])
        self.assertTrue(decision["factor_subject_pass"])
        self.assertTrue(decision["factor_relation_pass"])
        self.assertEqual(decision["factor_failure_partition"], "none")

    def test_v43_factored_trace_bank_honors_zero_margin_threshold(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_factored_address_trace_bank"
        hparams.factored_subject_margin_threshold = 0.0
        hparams.factored_relation_margin_threshold = 0.0
        hparams.factored_subject_energy_threshold = 0.0
        hparams.factored_relation_energy_threshold = 0.0
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)

        def make_entry(edit_id, subject, subj_factor, rel_factor):
            return {
                "edit_id": edit_id,
                "prompt": f"Where was {subject} founded?",
                "subject": subject,
                "target_new": "Paris",
                "raw_semantic_key": subj_factor.clone(),
                "raw_activation_key": rel_factor.clone(),
                "semantic_key": subj_factor.clone(),
                "activation_key": rel_factor.clone(),
                "trace_id": edit_id,
                "trace_value_impl": "exact_lora_cold_store",
                "value_ref": edit_id,
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": f"Where was {subject} founded?",
                        "raw_semantic_key": subj_factor.clone(),
                        "raw_activation_key": rel_factor.clone(),
                        "semantic_key": subj_factor.clone(),
                        "activation_key": rel_factor.clone(),
                        "subject_factor": subj_factor.clone(),
                        "relation_factor": rel_factor.clone(),
                        "subject_found": True,
                        "relation_token_count": 2,
                    }
                ],
            }

        entry_a = make_entry("hopedit_00000", "Alpha", torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        entry_b = make_entry("hopedit_00001", "Beta", torch.tensor([0.0, 1.0]), torch.tensor([0.99, 0.14106736]))
        controller.edit_registry = {entry_a["edit_id"]: entry_a, entry_b["edit_id"]: entry_b}
        controller.memory_entries = [entry_a, entry_b]
        controller._rebuild_factored_trace_address_state()

        query = {
            "subject_factor": torch.tensor([1.0, 0.0]),
            "relation_factor": torch.tensor([1.0, 0.0]),
            "combined_vector": controller._factored_combined_address_vector(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])),
            "query_code": torch.zeros(0),
            "query_support": [],
            "resolved_subject": "Alpha",
            "subject_found": True,
            "relation_token_count": 2,
        }
        decision = controller._route_from_factored_query(query)
        self.assertLess(decision["factor_relation_margin"], 0.03)
        self.assertEqual(decision["chosen_memory_id"], "hopedit_00000")
        self.assertTrue(decision["factor_relation_pass"])

    def test_v43_factored_trace_bank_can_log_full_factor_scores(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_factored_address_trace_bank"
        hparams.factored_subject_margin_threshold = 0.0
        hparams.factored_relation_margin_threshold = 0.0
        hparams.factored_subject_energy_threshold = 0.0
        hparams.factored_relation_energy_threshold = 0.0
        hparams.log_full_factor_scores = True
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)

        def make_entry(edit_id, subject, subj_factor, rel_factor):
            return {
                "edit_id": edit_id,
                "prompt": f"Where was {subject} founded?",
                "subject": subject,
                "target_new": "Paris",
                "raw_semantic_key": subj_factor.clone(),
                "raw_activation_key": rel_factor.clone(),
                "semantic_key": subj_factor.clone(),
                "activation_key": rel_factor.clone(),
                "trace_id": edit_id,
                "trace_value_impl": "exact_lora_cold_store",
                "value_ref": edit_id,
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": f"Where was {subject} founded?",
                        "raw_semantic_key": subj_factor.clone(),
                        "raw_activation_key": rel_factor.clone(),
                        "semantic_key": subj_factor.clone(),
                        "activation_key": rel_factor.clone(),
                        "subject_factor": subj_factor.clone(),
                        "relation_factor": rel_factor.clone(),
                        "subject_found": True,
                        "relation_token_count": 2,
                    }
                ],
            }

        entry_a = make_entry("hopedit_00000", "Alpha", torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        entry_b = make_entry("hopedit_00001", "Beta", torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
        controller.edit_registry = {entry_a["edit_id"]: entry_a, entry_b["edit_id"]: entry_b}
        controller.memory_entries = [entry_a, entry_b]
        controller._rebuild_factored_trace_address_state()

        query = {
            "subject_factor": torch.tensor([1.0, 0.0]),
            "relation_factor": torch.tensor([1.0, 0.0]),
            "combined_vector": controller._factored_combined_address_vector(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])),
            "query_code": torch.zeros(0),
            "query_support": [],
            "resolved_subject": "Alpha",
            "subject_found": True,
            "relation_token_count": 2,
        }
        decision = controller._route_from_factored_query(query)
        self.assertEqual(decision["factor_score_trace_ids"], ["hopedit_00000", "hopedit_00001"])
        self.assertEqual(decision["factor_subject_scores"], [1.0, 0.0])
        self.assertEqual(decision["factor_relation_scores"], [1.0, 0.0])
        self.assertEqual(decision["factor_subject_margin_threshold"], 0.0)

    def test_v44_relation_encoder_starts_as_identity(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_factored_address_trace_bank"
        hparams.factored_relation_encoder_impl = "residual_mlp"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        raw_factor = torch.tensor([1.0, 2.0, 0.0, 0.0])

        encoded = controller._encode_factored_relation_factor(raw_factor)

        self.assertTrue(torch.allclose(encoded, raw_factor / raw_factor.norm(), atol=1.0e-6))

    def test_v44_relation_encoder_training_refreshes_view_factors(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_factored_address_trace_bank"
        hparams.factored_relation_encoder_impl = "residual_mlp"
        hparams.factored_relation_encoder_steps = 2
        hparams.factored_relation_encoder_min_examples = 4
        hparams.factored_relation_encoder_hidden_dim = 4
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller.memory_entries = [
            {
                "edit_id": "hopedit_00000",
                "relation_id": "P19",
                "view_keys": [
                    {"view_name": "prompt", "relation_raw_factor": torch.tensor([1.0, 0.0, 0.0, 0.0])},
                    {"view_name": "rephrase", "relation_raw_factor": torch.tensor([0.9, 0.1, 0.0, 0.0])},
                ],
            },
            {
                "edit_id": "hopedit_00001",
                "relation_id": "P20",
                "view_keys": [
                    {"view_name": "prompt", "relation_raw_factor": torch.tensor([0.0, 1.0, 0.0, 0.0])},
                    {"view_name": "rephrase", "relation_raw_factor": torch.tensor([0.1, 0.9, 0.0, 0.0])},
                ],
            },
        ]

        trained = controller._maybe_train_factored_relation_encoder()

        self.assertTrue(trained)
        self.assertEqual(controller.factored_relation_encoder_updates, 1)
        self.assertIsNotNone(controller.factored_relation_encoder_last_loss)
        for entry in controller.memory_entries:
            for view in entry["view_keys"]:
                self.assertIn("relation_factor", view)
                self.assertAlmostEqual(float(view["relation_factor"].norm().item()), 1.0, places=5)

    def test_v44_relation_encoder_checkpoint_round_trip(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_factored_address_trace_bank"
        hparams.factored_relation_encoder_impl = "residual_mlp"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        encoder = controller._ensure_factored_relation_encoder(4)
        self.assertIsNotNone(encoder)
        with torch.no_grad():
            encoder.net[-1].bias.fill_(0.125)
        controller.factored_relation_encoder_updates = 3
        controller.factored_relation_encoder_last_loss = 0.5

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            controller.save_runtime_checkpoint(checkpoint_dir)
            restored = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
            restored.load_runtime_checkpoint(checkpoint_dir, is_trainable=False)

        restored_encoder = restored.factored_relation_encoder
        self.assertIsNotNone(restored_encoder)
        self.assertEqual(restored.factored_relation_encoder_updates, 3)
        self.assertEqual(restored.factored_relation_encoder_last_loss, 0.5)
        self.assertTrue(torch.allclose(restored_encoder.net[-1].bias, torch.full_like(restored_encoder.net[-1].bias, 0.125)))

    def test_v45_subject_candidate_relation_rule_allows_relation_clusters(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_factored_address_trace_bank"
        hparams.factored_relation_match_rule = "subject_candidate"
        hparams.factored_subject_margin_threshold = 0.0
        hparams.factored_relation_margin_threshold = 0.0
        hparams.factored_subject_energy_threshold = 0.0
        hparams.factored_relation_energy_threshold = 0.0
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)

        def make_entry(edit_id, subject, relation_id, subj_factor, rel_factor):
            return {
                "edit_id": edit_id,
                "prompt": f"Where was {subject} founded?",
                "subject": subject,
                "relation_id": relation_id,
                "target_new": "Paris",
                "raw_semantic_key": subj_factor.clone(),
                "raw_activation_key": rel_factor.clone(),
                "semantic_key": subj_factor.clone(),
                "activation_key": rel_factor.clone(),
                "trace_id": edit_id,
                "trace_value_impl": "exact_lora_cold_store",
                "value_ref": edit_id,
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": f"Where was {subject} founded?",
                        "raw_semantic_key": subj_factor.clone(),
                        "raw_activation_key": rel_factor.clone(),
                        "semantic_key": subj_factor.clone(),
                        "activation_key": rel_factor.clone(),
                        "subject_factor": subj_factor.clone(),
                        "relation_factor": rel_factor.clone(),
                        "subject_found": True,
                        "relation_token_count": 2,
                    }
                ],
            }

        target = make_entry("hopedit_00000", "Alpha", "R1", torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        same_relation_neighbor = make_entry("hopedit_00001", "Beta", "R1", torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]))
        other_relation = make_entry("hopedit_00002", "Gamma", "R2", torch.tensor([0.0, -1.0]), torch.tensor([0.0, 1.0]))
        controller.edit_registry = {entry["edit_id"]: entry for entry in [same_relation_neighbor, target, other_relation]}
        controller.memory_entries = [same_relation_neighbor, target, other_relation]
        controller._rebuild_factored_trace_address_state()

        query = {
            "subject_factor": torch.tensor([1.0, 0.0]),
            "relation_factor": torch.tensor([1.0, 0.0]),
            "combined_vector": controller._factored_combined_address_vector(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])),
            "query_code": torch.zeros(0),
            "query_support": [],
            "resolved_subject": "Alpha",
            "subject_found": True,
            "relation_token_count": 2,
        }
        decision = controller._route_from_factored_query(query)
        self.assertEqual(decision["factor_relation_top_edit_id"], "hopedit_00001")
        self.assertFalse(decision["factor_top1_same_trace"])
        self.assertEqual(decision["chosen_memory_id"], "hopedit_00000")
        self.assertEqual(decision["factor_relation_match_rule"], "subject_candidate")

    def test_v45_relation_encoder_loads_frozen_checkpoint(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        source_encoder = HopEditController(
            model=GPT2LMHeadModel(config),
            tok=DummyTokenizer(),
            hparams=make_hparams(),
        )
        encoder = source_encoder._ensure_factored_relation_encoder(4)
        self.assertIsNone(encoder)

        hparams = make_hparams()
        hparams.hopedit_mode = "v4_factored_address_trace_bank"
        hparams.factored_relation_encoder_impl = "residual_mlp"
        checkpoint_encoder = HopEditController(
            model=GPT2LMHeadModel(config),
            tok=DummyTokenizer(),
            hparams=hparams,
        )._ensure_factored_relation_encoder(4)
        with torch.no_grad():
            checkpoint_encoder.net[-1].bias.fill_(0.25)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = f"{tmpdir}/relation_encoder.pt"
            torch.save(
                {
                    "encoder_state_dict": checkpoint_encoder.state_dict(),
                    "input_dim": 4,
                    "hidden_dim": 8,
                    "best_dev_q_r": 0.75,
                    "train_count": 500,
                    "dev_count": 50,
                },
                checkpoint_path,
            )
            hparams.factored_relation_encoder_checkpoint = checkpoint_path
            controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
            encoded = controller._encode_factored_relation_factor(torch.tensor([1.0, 0.0, 0.0, 0.0]))

        self.assertEqual(controller.factored_relation_encoder_checkpoint_loaded, checkpoint_path)
        self.assertEqual(controller.factored_relation_encoder_checkpoint_metadata["best_dev_q_r"], 0.75)
        self.assertAlmostEqual(float(encoded.norm().item()), 1.0, places=5)

    def test_v43_factored_trace_bank_abstains_on_factor_disagreement(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_factored_address_trace_bank"
        hparams.factored_subject_margin_threshold = 0.0
        hparams.factored_relation_margin_threshold = 0.0
        hparams.factored_subject_energy_threshold = 0.0
        hparams.factored_relation_energy_threshold = 0.0
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)

        entry_a = {
            "edit_id": "hopedit_00000",
            "prompt": "Where was Alpha founded?",
            "subject": "Alpha",
            "target_new": "Paris",
            "raw_semantic_key": torch.tensor([1.0, 0.0]),
            "raw_activation_key": torch.tensor([1.0, 0.0]),
            "semantic_key": torch.tensor([1.0, 0.0]),
            "activation_key": torch.tensor([1.0, 0.0]),
            "trace_id": "hopedit_00000",
            "trace_value_impl": "exact_lora_cold_store",
            "value_ref": "hopedit_00000",
            "view_keys": [
                {
                    "view_name": "prompt",
                    "text": "Where was Alpha founded?",
                    "raw_semantic_key": torch.tensor([1.0, 0.0]),
                    "raw_activation_key": torch.tensor([1.0, 0.0]),
                    "semantic_key": torch.tensor([1.0, 0.0]),
                    "activation_key": torch.tensor([1.0, 0.0]),
                    "subject_factor": torch.tensor([1.0, 0.0]),
                    "relation_factor": torch.tensor([0.0, 1.0]),
                    "subject_found": True,
                    "relation_token_count": 2,
                }
            ],
        }
        entry_b = {
            "edit_id": "hopedit_00001",
            "prompt": "Where was Beta founded?",
            "subject": "Beta",
            "target_new": "Rome",
            "raw_semantic_key": torch.tensor([0.0, 1.0]),
            "raw_activation_key": torch.tensor([0.0, 1.0]),
            "semantic_key": torch.tensor([0.0, 1.0]),
            "activation_key": torch.tensor([0.0, 1.0]),
            "trace_id": "hopedit_00001",
            "trace_value_impl": "exact_lora_cold_store",
            "value_ref": "hopedit_00001",
            "view_keys": [
                {
                    "view_name": "prompt",
                    "text": "Where was Beta founded?",
                    "raw_semantic_key": torch.tensor([0.0, 1.0]),
                    "raw_activation_key": torch.tensor([0.0, 1.0]),
                    "semantic_key": torch.tensor([0.0, 1.0]),
                    "activation_key": torch.tensor([0.0, 1.0]),
                    "subject_factor": torch.tensor([0.0, 1.0]),
                    "relation_factor": torch.tensor([1.0, 0.0]),
                    "subject_found": True,
                    "relation_token_count": 2,
                }
            ],
        }
        controller.edit_registry = {entry_a["edit_id"]: entry_a, entry_b["edit_id"]: entry_b}
        controller.memory_entries = [entry_a, entry_b]
        controller._rebuild_factored_trace_address_state()

        query = {
            "subject_factor": torch.tensor([1.0, 0.0]),
            "relation_factor": torch.tensor([1.0, 0.0]),
        }
        query["combined_vector"] = controller._factored_combined_address_vector(query["subject_factor"], query["relation_factor"])
        query["query_code"] = controller._encode_sparse_address(query["combined_vector"], controller.address_dictionary["atoms"])
        query["query_support"] = controller._address_support_from_code(query["query_code"])
        query["resolved_subject"] = "Alpha"
        query["subject_found"] = True
        query["relation_token_count"] = 2

        decision = controller._route_from_factored_query(query)
        self.assertIsNone(decision["chosen_memory_id"])
        self.assertTrue(decision["address_abstained"])
        self.assertFalse(decision["factor_same_trace"])
        self.assertEqual(decision["factor_failure_partition"], "both")

    def test_v43_factored_trace_bank_does_not_depend_on_flat_shortlist(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_factored_address_trace_bank"
        hparams.factored_subject_margin_threshold = 0.0
        hparams.factored_relation_margin_threshold = 0.0
        hparams.factored_subject_energy_threshold = 0.0
        hparams.factored_relation_energy_threshold = 0.0
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)

        def make_entry(edit_id, subject, subj_factor, rel_factor):
            return {
                "edit_id": edit_id,
                "prompt": f"Where was {subject} founded?",
                "subject": subject,
                "target_new": "Paris",
                "raw_semantic_key": subj_factor.clone(),
                "raw_activation_key": rel_factor.clone(),
                "semantic_key": subj_factor.clone(),
                "activation_key": rel_factor.clone(),
                "trace_id": edit_id,
                "trace_value_impl": "exact_lora_cold_store",
                "value_ref": edit_id,
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": f"Where was {subject} founded?",
                        "raw_semantic_key": subj_factor.clone(),
                        "raw_activation_key": rel_factor.clone(),
                        "semantic_key": subj_factor.clone(),
                        "activation_key": rel_factor.clone(),
                        "subject_factor": subj_factor.clone(),
                        "relation_factor": rel_factor.clone(),
                        "subject_found": True,
                        "relation_token_count": 2,
                    }
                ],
            }

        entry_a = make_entry("hopedit_00000", "Alpha", torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        entry_b = make_entry("hopedit_00001", "Beta", torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
        controller.edit_registry = {entry_a["edit_id"]: entry_a, entry_b["edit_id"]: entry_b}
        controller.memory_entries = [entry_a, entry_b]
        controller._rebuild_factored_trace_address_state()

        def fail_if_called(*args, **kwargs):
            raise AssertionError("flat shortlist should not gate v4.3 factored routing")

        controller._candidate_trace_ids_from_query_code = fail_if_called

        query = {
            "subject_factor": torch.tensor([1.0, 0.0]),
            "relation_factor": torch.tensor([1.0, 0.0]),
            "combined_vector": controller._factored_combined_address_vector(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])),
            "query_code": torch.zeros(0),
            "query_support": [],
            "resolved_subject": "Alpha",
            "subject_found": True,
            "relation_token_count": 2,
        }
        decision = controller._route_from_factored_query(query)
        self.assertEqual(decision["chosen_memory_id"], "hopedit_00000")

    def test_v43_forward_propagates_metadata_to_router(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_factored_address_trace_bank"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller.memory_entries = [{"edit_id": "hopedit_00000"}]
        captured = {}

        def capture_select(input_ids, attention_mask=None, metadata=None):
            captured["metadata"] = metadata
            return {"adapter_name": None}

        controller._select_adapter_for_inputs = capture_select
        batch = controller._tokenize(["Where was Alpha founded?"])
        metadata = {"prompt": "Where was Alpha founded?", "subject": "Alpha"}
        controller(**batch, metadata=metadata)
        self.assertEqual(captured["metadata"], metadata)

    def test_discrete_support_masks_apply_write_time_exclusivity(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        model = GPT2LMHeadModel(config)
        hparams = make_hparams()
        hparams.state_memory_impl = "sparse_slots"
        hparams.slot_realization_impl = "discrete_support_masks"
        hparams.atom_write_topk = 1
        hparams.atom_sparse_topk = 1
        hparams.atom_support_exclusivity_penalty = 0.5
        controller = HopEditController(model=model, tok=DummyTokenizer(), hparams=hparams)

        state_basis = {
            "test.attn": {
                "canonical_lora_A": "test.attn.lora_A.__ADAPTER__.weight",
                "canonical_lora_B": "test.attn.lora_B.__ADAPTER__.weight",
                "left_basis": torch.eye(2, dtype=torch.float32),
                "right_basis": torch.eye(2, dtype=torch.float32),
                "singular_values": torch.ones(2, dtype=torch.float32),
                "basis_rank": 2,
            }
        }
        shared_weights = {
            "test.attn.lora_A.__ADAPTER__.weight": torch.tensor([[1.0, 0.9]], dtype=torch.float32),
            "test.attn.lora_B.__ADAPTER__.weight": torch.tensor([[1.0], [1.0]], dtype=torch.float32),
        }
        slots = [
            {
                "slot_id": "s0",
                "slot_rank": 1,
                "slot_alpha": 1.0,
                "slot_weights": {k: v.clone() for k, v in shared_weights.items()},
            },
            {
                "slot_id": "s1",
                "slot_rank": 1,
                "slot_alpha": 1.0,
                "slot_weights": {k: v.clone() for k, v in shared_weights.items()},
            },
        ]

        controller._fit_slot_latent_supports(slots, state_basis)

        support0 = slots[0]["slot_codes"]["test.attn"]
        support1 = slots[1]["slot_codes"]["test.attn"]
        self.assertTrue(torch.equal(support0, torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(support1, torch.tensor([0.0, 1.0])))

    def test_discrete_support_overlap_uses_atom_strengths(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        model = GPT2LMHeadModel(config)
        hparams = make_hparams()
        hparams.state_memory_impl = "sparse_slots"
        hparams.slot_realization_impl = "discrete_support_masks"
        controller = HopEditController(model=model, tok=DummyTokenizer(), hparams=hparams)

        cell = {
            "state_shared_basis": {
                "test.attn": {
                    "left_basis": torch.eye(2, dtype=torch.float32),
                    "right_basis": torch.eye(2, dtype=torch.float32),
                    "singular_values": torch.tensor([2.0, 1.0], dtype=torch.float32),
                    "basis_rank": 2,
                }
            }
        }
        overlap = controller._state_self_overlap(cell, {"test.attn": torch.tensor([1.0, 0.0])})
        self.assertAlmostEqual(overlap, 4.0, places=5)

    def test_sparse_slot_runtime_checkpoint_round_trip(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        model = GPT2LMHeadModel(config)
        hparams = make_hparams()
        hparams.state_memory_impl = "sparse_slots"
        controller = HopEditController(model=model, tok=DummyTokenizer(), hparams=hparams)
        controller.memory_entries = [
            {
                "edit_id": "hopedit_00000",
                "cell_id": "hopedit_cell_00000",
                "raw_semantic_key": torch.tensor([1.0, 0.0]),
                "raw_activation_key": torch.tensor([1.0, 0.0]),
                "semantic_key": torch.tensor([1.0, 0.0]),
                "activation_key": torch.tensor([1.0, 0.0]),
                "prompt": "p0",
                "view_keys": [{"view_name": "prompt", "text": "p0", "raw_semantic_key": torch.tensor([1.0, 0.0]), "raw_activation_key": torch.tensor([1.0, 0.0]), "semantic_key": torch.tensor([1.0, 0.0]), "activation_key": torch.tensor([1.0, 0.0])}],
            }
        ]
        controller.cell_registry = {
            "hopedit_cell_00000": {
                "cell_id": "hopedit_cell_00000",
                "adapter_name": None,
                "member_edit_ids": ["hopedit_00000"],
                "tier": "active",
                "bucket_id": None,
                "state_stability_score": 0.75,
                "cross_view_route_gap": 0.02,
                "locality_fragility": 0.01,
                "is_stable": False,
                "state_support_observations": 3,
                "slots": [
                    {
                        "slot_id": "hopedit_cell_00000_slot_000",
                        "source_edit_id": "hopedit_00000",
                        "slot_prototypes": [
                            {
                                "view_name": "prompt",
                                "text": "p0",
                                "semantic_key": torch.tensor([1.0, 0.0]),
                                "activation_key": torch.tensor([1.0, 0.0]),
                                "prototype_dispersion": 0.0,
                            }
                        ],
                        "slot_dispersion": 0.0,
                        "slot_usage_count": 2,
                        "slot_weights": {
                            "base_model.model.transformer.h.0.attn.c_attn.lora_A.__ADAPTER__.weight": torch.ones(2, 16),
                            "base_model.model.transformer.h.0.attn.c_attn.lora_B.__ADAPTER__.weight": torch.ones(48, 2),
                        },
                    }
                ],
                "state_summary_prototypes": [
                    {
                        "view_name": "prompt",
                        "text": "p0",
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                        "prototype_dispersion": 0.0,
                    }
                ],
                "cell_prototypes": [],
                "prototype_stats": {"prototype_count_by_view": {"prompt": 1}},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            controller.save_runtime_checkpoint(tmpdir)
            restored = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=make_hparams())
            restored.hparams.state_memory_impl = "sparse_slots"
            restored.load_runtime_checkpoint(tmpdir, is_trainable=True)
            restored_slot = restored.cell_registry["hopedit_cell_00000"]["slots"][0]
            self.assertEqual(restored_slot["slot_id"], "hopedit_cell_00000_slot_000")
            self.assertEqual(restored_slot["slot_usage_count"], 2)
            self.assertTrue(torch.allclose(restored_slot["slot_weights"]["base_model.model.transformer.h.0.attn.c_attn.lora_A.__ADAPTER__.weight"], torch.ones(2, 16)))

    def test_v2_runtime_checkpoint_round_trip(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        model = GPT2LMHeadModel(config)
        tok = DummyTokenizer()
        hparams = make_hparams()
        hparams.state_gate_enable = True
        controller = HopEditController(model=model, tok=tok, hparams=hparams)
        controller.state_gate_module.weight.data.fill_(0.25)
        controller.state_gate_module.bias.data.fill_(0.1)
        controller.state_gate_feature_stats = {
            "mean": torch.zeros_like(controller.state_gate_feature_stats["mean"]),
            "std": torch.ones_like(controller.state_gate_feature_stats["std"]),
        }
        controller.state_gate_replay_buffer = [
            {
                "features": torch.ones(len(controller.state_gate_feature_stats["mean"]), dtype=torch.float32),
                "label": 1.0,
                "source": "unit",
            }
        ]
        controller.state_gate_seen_examples = 3
        controller.state_gate_online_updates = 2
        controller.state_gate_warm_start_source = ["unit"]
        controller.state_gate_runtime_counts["direct_accepted_count"] = 5
        controller.state_gate_ready = True

        controller._ensure_adapter("hopedit_cell_00000")
        controller._ensure_adapter("hopedit_cell_00001")
        controller.cell_registry = {
            "hopedit_cell_00000": {
                "cell_id": "hopedit_cell_00000",
                "adapter_name": "hopedit_cell_00000",
                "member_edit_ids": ["hopedit_00000"],
                "tier": "consolidated",
                "bucket_id": "hopedit_bucket_00000",
                "state_stability_score": 0.97,
                "cross_view_route_gap": 0.02,
                "locality_fragility": 0.01,
                "is_stable": True,
                "state_support_observations": 8,
                "state_gate_recent_scores": [0.91, 0.93],
                "state_gate_score_mean": 0.92,
                "cell_prototypes": [
                    {
                        "view_name": "prompt",
                        "text": "p0",
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                        "prototype_dispersion": 0.1,
                    }
                ],
                "prototype_stats": {"prototype_count_by_view": {"prompt": 1}},
            },
            "hopedit_cell_00001": {
                "cell_id": "hopedit_cell_00001",
                "adapter_name": "hopedit_cell_00001",
                "member_edit_ids": ["hopedit_00001"],
                "tier": "active",
                "bucket_id": None,
                "state_stability_score": 0.45,
                "cross_view_route_gap": 0.40,
                "locality_fragility": 0.03,
                "is_stable": False,
                "state_support_observations": 2,
                "state_gate_recent_scores": [0.44, 0.48],
                "state_gate_score_mean": 0.46,
                "cell_prototypes": [
                    {
                        "view_name": "rephrase",
                        "text": "p1",
                        "semantic_key": torch.tensor([0.0, 1.0]),
                        "activation_key": torch.tensor([0.0, 1.0]),
                        "prototype_dispersion": 0.2,
                    }
                ],
                "prototype_stats": {"prototype_count_by_view": {"rephrase": 1}},
            },
        }
        controller.bucket_registry = {
            "hopedit_bucket_00000": {
                "bucket_id": "hopedit_bucket_00000",
                "state_ids": ["hopedit_cell_00000"],
                "bucket_prototypes": [
                    {
                        "view_name": "prompt",
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                        "prototype_dispersion": 0.1,
                    }
                ],
                "bucket_dispersion": 0.1,
            }
        }
        controller.edit_index = 2
        controller.cell_index = 2

        weights_before = controller.runtime_state_dict()["cell_adapter_weights"]
        with tempfile.TemporaryDirectory() as tmpdir:
            controller.save_runtime_checkpoint(tmpdir)
            restored = HopEditController(model=GPT2LMHeadModel(config), tok=tok, hparams=make_hparams())
            restored.load_runtime_checkpoint(tmpdir, is_trainable=True)
            self.assertEqual(restored.hparams.hopedit_mode, "v2_cell_bank")
            self.assertEqual(set(restored.cell_registry.keys()), {"hopedit_cell_00000", "hopedit_cell_00001"})
            self.assertEqual(
                restored.cell_registry["hopedit_cell_00000"]["prototype_stats"]["prototype_count_by_view"]["prompt"],
                1,
            )
            self.assertEqual(
                restored.cell_registry["hopedit_cell_00001"]["cell_prototypes"][0]["view_name"],
                "rephrase",
            )
            self.assertEqual(restored.cell_registry["hopedit_cell_00000"]["tier"], "consolidated")
            self.assertEqual(restored.cell_registry["hopedit_cell_00000"]["bucket_id"], "hopedit_bucket_00000")
            self.assertEqual(restored.cell_registry["hopedit_cell_00000"]["state_gate_score_mean"], 0.92)
            self.assertEqual(restored.bucket_registry["hopedit_bucket_00000"]["state_ids"], ["hopedit_cell_00000"])
            self.assertTrue(restored.state_gate_ready)
            self.assertEqual(restored.state_gate_seen_examples, 3)
            self.assertEqual(restored.state_gate_online_updates, 2)
            self.assertEqual(restored.state_gate_warm_start_source, ["unit"])
            self.assertEqual(restored.state_gate_runtime_counts["direct_accepted_count"], 5)
            self.assertTrue(torch.allclose(restored.state_gate_module.weight, controller.state_gate_module.weight))
            self.assertTrue(torch.allclose(restored.state_gate_module.bias, controller.state_gate_module.bias))
            weights_after = restored.runtime_state_dict()["cell_adapter_weights"]
            self.assertEqual(set(weights_before.keys()), set(weights_after.keys()))
            for adapter_name in weights_before:
                for tensor_name in weights_before[adapter_name]:
                    self.assertTrue(torch.allclose(weights_before[adapter_name][tensor_name], weights_after[adapter_name][tensor_name]))

    def test_v22_multiview_routes_prompt_and_rephrase_to_same_cell(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=make_hparams())
        controller.cell_registry = {
            "hopedit_cell_00000": {
                "cell_id": "hopedit_cell_00000",
                "adapter_name": None,
                "member_edit_ids": ["hopedit_00000"],
                "cell_prototypes": [
                    {
                        "view_name": "prompt",
                        "text": "prompt-view",
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                        "prototype_dispersion": 0.1,
                    },
                    {
                        "view_name": "rephrase",
                        "text": "rephrase-view",
                        "semantic_key": torch.tensor([0.0, 1.0]),
                        "activation_key": torch.tensor([0.0, 1.0]),
                        "prototype_dispersion": 0.1,
                    },
                ],
            },
            "hopedit_cell_00001": {
                "cell_id": "hopedit_cell_00001",
                "adapter_name": None,
                "member_edit_ids": ["hopedit_00001"],
                "cell_prototypes": [
                    {
                        "view_name": "prompt",
                        "text": "other-prompt",
                        "semantic_key": torch.tensor([-1.0, 0.0]),
                        "activation_key": torch.tensor([-1.0, 0.0]),
                        "prototype_dispersion": 0.1,
                    }
                ],
            },
        }
        prompt_decision = controller._route_from_keys_v2(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        rephrase_decision = controller._route_from_keys_v2(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
        self.assertEqual(prompt_decision["chosen_cell_id"], "hopedit_cell_00000")
        self.assertEqual(rephrase_decision["chosen_cell_id"], "hopedit_cell_00000")

    def test_v23_ambiguity_triggers_rerank(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.cell_prototype_rerank_topk = 4
        hparams.state_gate_enable = True
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller.state_gate_ready = True
        controller.state_gate_module.weight.data.zero_()
        controller.state_gate_module.bias.data.fill_(-10.0)
        controller.cell_registry = {
            "hopedit_cell_00000": {
                "cell_id": "hopedit_cell_00000",
                "adapter_name": None,
                "member_edit_ids": ["hopedit_00000"],
                "cell_prototypes": [
                    {
                        "view_name": "prompt",
                        "text": "prompt-a",
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                        "prototype_dispersion": 0.02,
                    }
                ],
                "prototype_dispersion": 0.02,
                "tier": "active",
            },
            "hopedit_cell_00001": {
                "cell_id": "hopedit_cell_00001",
                "adapter_name": None,
                "member_edit_ids": ["hopedit_00001"],
                "cell_prototypes": [
                    {
                        "view_name": "prompt",
                        "text": "prompt-b",
                        "semantic_key": torch.tensor([0.94, 0.06]),
                        "activation_key": torch.tensor([0.94, 0.06]),
                        "prototype_dispersion": 0.35,
                    }
                ],
                "prototype_dispersion": 0.35,
                "tier": "active",
            },
        }
        decision = controller._route_from_keys_v2(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        self.assertEqual(decision["route_stage"], "ambiguity_rerank")
        self.assertEqual(decision["chosen_cell_id"], "hopedit_cell_00000")
        self.assertEqual(decision["gate_decision"], "rerank")

    def test_v23_gate_can_force_direct_multiview(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.state_gate_enable = True
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller.state_gate_ready = True
        controller.state_gate_module.weight.data.zero_()
        controller.state_gate_module.bias.data.fill_(10.0)
        controller.cell_registry = {
            "hopedit_cell_00000": {
                "cell_id": "hopedit_cell_00000",
                "adapter_name": None,
                "member_edit_ids": ["hopedit_00000"],
                "cell_prototypes": [
                    {
                        "view_name": "prompt",
                        "text": "prompt-a",
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                        "prototype_dispersion": 0.02,
                    }
                ],
                "prototype_dispersion": 0.02,
                "tier": "active",
            },
            "hopedit_cell_00001": {
                "cell_id": "hopedit_cell_00001",
                "adapter_name": None,
                "member_edit_ids": ["hopedit_00001"],
                "cell_prototypes": [
                    {
                        "view_name": "prompt",
                        "text": "prompt-b",
                        "semantic_key": torch.tensor([0.9, 0.1]),
                        "activation_key": torch.tensor([0.9, 0.1]),
                        "prototype_dispersion": 0.15,
                    }
                ],
                "prototype_dispersion": 0.15,
                "tier": "active",
            },
        }
        decision = controller._route_from_keys_v2(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        self.assertEqual(decision["route_stage"], "direct_multiview")
        self.assertEqual(decision["gate_decision"], "direct_accept")
        self.assertEqual(decision["chosen_cell_id"], "hopedit_cell_00000")

    def test_trial_merge_rejects_high_dispersion_merge(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=make_hparams())
        controller.memory_entries = [
            {
                "edit_id": "hopedit_00000",
                "cell_id": "hopedit_cell_00000",
                "semantic_key": torch.tensor([1.0, 0.0]),
                "activation_key": torch.tensor([1.0, 0.0]),
                "prompt": "p0",
                "view_keys": [{"view_name": "prompt", "text": "p0", "semantic_key": torch.tensor([1.0, 0.0]), "activation_key": torch.tensor([1.0, 0.0])}],
            },
            {
                "edit_id": "hopedit_00001",
                "cell_id": "hopedit_cell_00001",
                "semantic_key": torch.tensor([-1.0, 0.0]),
                "activation_key": torch.tensor([-1.0, 0.0]),
                "prompt": "p1",
                "view_keys": [{"view_name": "prompt", "text": "p1", "semantic_key": torch.tensor([-1.0, 0.0]), "activation_key": torch.tensor([-1.0, 0.0])}],
            },
        ]
        target_cell = {"cell_id": "hopedit_cell_00000"}
        source_cell = {"cell_id": "hopedit_cell_00001"}
        trial = controller._trial_merge_state_score(target_cell, source_cell)
        self.assertFalse(trial["accepted"])

    def test_trial_merge_rejects_low_gate_score(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.state_gate_enable = True
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller.memory_entries = [
            {
                "edit_id": "hopedit_00000",
                "cell_id": "hopedit_cell_00000",
                "semantic_key": torch.tensor([1.0, 0.0]),
                "activation_key": torch.tensor([1.0, 0.0]),
                "prompt": "p0",
                "view_keys": [{"view_name": "prompt", "text": "p0", "semantic_key": torch.tensor([1.0, 0.0]), "activation_key": torch.tensor([1.0, 0.0])}],
            },
            {
                "edit_id": "hopedit_00001",
                "cell_id": "hopedit_cell_00001",
                "semantic_key": torch.tensor([0.98, 0.02]),
                "activation_key": torch.tensor([0.98, 0.02]),
                "prompt": "p1",
                "view_keys": [{"view_name": "prompt", "text": "p1", "semantic_key": torch.tensor([0.98, 0.02]), "activation_key": torch.tensor([0.98, 0.02])}],
            },
        ]
        target_cell = {"cell_id": "hopedit_cell_00000", "state_gate_score_mean": 0.90}
        source_cell = {"cell_id": "hopedit_cell_00001", "state_gate_score_mean": 0.40}
        trial = controller._trial_merge_state_score(target_cell, source_cell)
        self.assertFalse(trial["accepted"])
        self.assertIn("gate_score", trial["reject_reasons"])

    def test_bucket_registry_splits_when_size_threshold_exceeded(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.bucket_max_size = 1
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller.memory_entries = [object(), object(), object(), object()]
        controller.cell_registry = {
            "hopedit_cell_00000": {
                "cell_id": "hopedit_cell_00000",
                "semantic_key": torch.tensor([1.0, 0.0]),
                "activation_key": torch.tensor([1.0, 0.0]),
                "cell_prototypes": [{"view_name": "prompt", "semantic_key": torch.tensor([1.0, 0.0]), "activation_key": torch.tensor([1.0, 0.0])}],
                "tier": "consolidated",
                "is_stable": True,
            },
            "hopedit_cell_00001": {
                "cell_id": "hopedit_cell_00001",
                "semantic_key": torch.tensor([0.9, 0.1]),
                "activation_key": torch.tensor([0.9, 0.1]),
                "cell_prototypes": [{"view_name": "prompt", "semantic_key": torch.tensor([0.9, 0.1]), "activation_key": torch.tensor([0.9, 0.1])}],
                "tier": "consolidated",
                "is_stable": True,
            },
        }
        controller._rebuild_bucket_registry()
        self.assertEqual(len(controller.bucket_registry), 2)
        for cell in controller.cell_registry.values():
            self.assertIsNotNone(cell.get("bucket_id"))

    def test_sparse_slot_routing_keeps_prompt_and_rephrase_in_same_state_and_slot_set(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.state_memory_impl = "sparse_slots"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        shared_slot = {
            "slot_id": "slot_shared",
            "slot_prototypes": [
                {"view_name": "prompt", "text": "prompt", "semantic_key": torch.tensor([1.0, 0.0]), "activation_key": torch.tensor([1.0, 0.0]), "prototype_dispersion": 0.0},
                {"view_name": "rephrase", "text": "rephrase", "semantic_key": torch.tensor([0.0, 1.0]), "activation_key": torch.tensor([0.0, 1.0]), "prototype_dispersion": 0.0},
            ],
            "slot_dispersion": 0.0,
            "slot_weights": {},
        }
        controller.cell_registry = {
            "hopedit_cell_00000": {
                "cell_id": "hopedit_cell_00000",
                "adapter_name": None,
                "slots": [shared_slot],
                "state_summary_prototypes": shared_slot["slot_prototypes"],
                "tier": "active",
                "within_cell_conflict_mean": 0.0,
            },
            "hopedit_cell_00001": {
                "cell_id": "hopedit_cell_00001",
                "adapter_name": None,
                "slots": [
                    {
                        "slot_id": "slot_other",
                        "slot_prototypes": [{"view_name": "prompt", "text": "other", "semantic_key": torch.tensor([-1.0, 0.0]), "activation_key": torch.tensor([-1.0, 0.0]), "prototype_dispersion": 0.0}],
                        "slot_dispersion": 0.0,
                        "slot_weights": {},
                    }
                ],
                "state_summary_prototypes": [{"view_name": "prompt", "text": "other", "semantic_key": torch.tensor([-1.0, 0.0]), "activation_key": torch.tensor([-1.0, 0.0]), "prototype_dispersion": 0.0}],
                "tier": "active",
                "within_cell_conflict_mean": 0.0,
            },
        }
        prompt_decision = controller._route_from_keys_v2(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        rephrase_decision = controller._route_from_keys_v2(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
        self.assertEqual(prompt_decision["chosen_cell_id"], "hopedit_cell_00000")
        self.assertEqual(rephrase_decision["chosen_cell_id"], "hopedit_cell_00000")
        self.assertEqual(prompt_decision["selected_slot_ids"], ["slot_shared"])
        self.assertEqual(rephrase_decision["selected_slot_ids"], ["slot_shared"])

    def test_sparse_locality_risk_can_reduce_activation_to_top1(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.state_memory_impl = "sparse_slots"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller.cell_registry = {
            "hopedit_cell_00000": {
                "cell_id": "hopedit_cell_00000",
                "adapter_name": None,
                "slots": [
                    {"slot_id": "slot_a", "slot_prototypes": [{"view_name": "prompt", "text": "a", "semantic_key": torch.tensor([1.0, 0.0]), "activation_key": torch.tensor([1.0, 0.0]), "prototype_dispersion": 0.1}], "slot_dispersion": 0.1, "slot_weights": {}},
                    {"slot_id": "slot_b", "slot_prototypes": [{"view_name": "prompt", "text": "b", "semantic_key": torch.tensor([0.96, 0.04]), "activation_key": torch.tensor([0.96, 0.04]), "prototype_dispersion": 0.1}], "slot_dispersion": 0.1, "slot_weights": {}},
                ],
                "state_summary_prototypes": [{"view_name": "prompt", "text": "sum", "semantic_key": torch.tensor([1.0, 0.0]), "activation_key": torch.tensor([1.0, 0.0]), "prototype_dispersion": 0.1}],
                "tier": "active",
                "within_cell_conflict_mean": 1.0,
            },
            "hopedit_cell_00001": {
                "cell_id": "hopedit_cell_00001",
                "adapter_name": None,
                "slots": [
                    {"slot_id": "slot_c", "slot_prototypes": [{"view_name": "prompt", "text": "c", "semantic_key": torch.tensor([0.94, 0.06]), "activation_key": torch.tensor([0.94, 0.06]), "prototype_dispersion": 0.1}], "slot_dispersion": 0.1, "slot_weights": {}},
                ],
                "state_summary_prototypes": [{"view_name": "prompt", "text": "sum", "semantic_key": torch.tensor([0.94, 0.06]), "activation_key": torch.tensor([0.94, 0.06]), "prototype_dispersion": 0.1}],
                "tier": "active",
                "within_cell_conflict_mean": 0.0,
            },
        }
        decision = controller._route_from_keys_v2(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        self.assertEqual(len(decision["selected_slot_ids"]), 1)

    def test_sparse_slot_transfer_rejects_incompatible_slot(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.state_memory_impl = "sparse_slots"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        target_cell = {"cell_id": "hopedit_cell_00000", "within_cell_conflict_mean": 0.6, "locality_proxy": 0.8}
        source_cell = {"cell_id": "hopedit_cell_00001", "cross_view_route_gap": 0.4, "locality_proxy": 0.8}
        slot = {"slot_id": "slot_bad", "slot_dispersion": 0.4}
        trial = controller._trial_slot_transfer_score(target_cell, source_cell, slot)
        self.assertFalse(trial["accepted"])
        self.assertIn("cross_view_route_gap", trial["reject_reasons"])

    def test_sparse_slot_composition_populates_runtime_adapter(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.state_memory_impl = "sparse_slots"
        hparams.slot_realization_impl = "shared_basis_codes"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        adapter_name = controller._ensure_runtime_slot_adapter()
        refs = controller._adapter_parameter_refs(adapter_name)
        state_shared_basis = {}
        for canonical_name, parameter in refs.items():
            if "lora_A" not in canonical_name:
                continue
            stem = canonical_name.replace(".lora_A.__ADAPTER__.weight", "")
            canonical_lora_b = stem + ".lora_B.__ADAPTER__.weight"
            if canonical_lora_b not in refs:
                continue
            realized_rank = controller._shared_basis_rank()
            left_basis = torch.zeros(refs[canonical_lora_b].shape[0], realized_rank)
            right_basis = torch.zeros(realized_rank, parameter.shape[1])
            for idx in range(realized_rank):
                left_basis[idx, idx] = 1.0
                right_basis[idx, idx] = 1.0
            state_shared_basis[stem] = {
                "canonical_lora_A": canonical_name,
                "canonical_lora_B": canonical_lora_b,
                "left_basis": left_basis,
                "right_basis": right_basis,
                "singular_values": torch.ones(realized_rank),
                "basis_rank": realized_rank,
            }
        cell = {"cell_id": "hopedit_cell_00000", "state_shared_basis": state_shared_basis}
        controller._load_composed_slot_adapter(
            [
                {
                    "cell": cell,
                    "slot": {
                        "slot_id": "slot_a",
                        "slot_codes": {stem: torch.tensor([1.0, 0.0, 0.0, 0.0]) for stem in state_shared_basis},
                    },
                    "slot_weight": 0.75,
                },
                {
                    "cell": cell,
                    "slot": {
                        "slot_id": "slot_b",
                        "slot_codes": {stem: torch.tensor([0.0, 2.0, 0.0, 0.0]) for stem in state_shared_basis},
                    },
                    "slot_weight": 0.25,
                },
            ]
        )
        weights = controller._capture_adapter_parameters(adapter_name)
        a_tensor = next(value for name, value in weights.items() if "lora_A" in name)
        self.assertGreater(float(a_tensor[0].abs().sum().item()), 0.0)
        self.assertGreater(float(a_tensor[1].abs().sum().item()), 0.0)

    def test_sparse_slot_delete_repairs_active_adapter_reference(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.state_memory_impl = "sparse_slots"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller._ensure_adapter("hopedit_00000", rank=controller._slot_rank(), alpha=float(controller._slot_rank()))
        controller._set_runtime_adapter("hopedit_00000")
        controller._delete_adapter("hopedit_00000")
        with controller._adapter_disabled():
            pass

    def test_v3_side_memory_route_selects_best_shard_support(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v3_side_memory"
        hparams.state_memory_impl = "side_memory_shards"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller._ensure_runtime_slot_adapter()
        refs = controller._adapter_parameter_refs(controller._runtime_slot_adapter_name())
        atom_weights = {}
        for canonical_name, parameter in refs.items():
            if "lora_A" in canonical_name:
                atom_weights[canonical_name] = torch.ones(2, parameter.shape[1], dtype=torch.float32)
            elif "lora_B" in canonical_name:
                atom_weights[canonical_name] = torch.ones(parameter.shape[0], 2, dtype=torch.float32)
        controller.shard_registry = {
            "hopedit_shard_00000": {
                "shard_id": "hopedit_shard_00000",
                "member_edit_ids": ["hopedit_00000"],
                "atoms": [
                    {
                        "atom_id": "hopedit_atom_00000",
                        "atom_weights": atom_weights,
                        "usage_count": 0,
                        "view_keys": [],
                        "atom_prototypes": [
                            {
                                "view_name": "prompt",
                                "text": "p0",
                                "semantic_key": torch.tensor([1.0, 0.0]),
                                "activation_key": torch.tensor([1.0, 0.0]),
                                "prototype_dispersion": 0.0,
                            }
                        ],
                    }
                ],
                "shard_prototypes": [
                    {
                        "view_name": "prompt",
                        "text": "p0",
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                        "prototype_dispersion": 0.0,
                    }
                ],
                "prototype_dispersion": 0.0,
                "prototype_margin_history": [],
            }
        }
        decision = controller._route_from_keys(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        self.assertEqual(decision["memory_unit"], "shard")
        self.assertEqual(decision["chosen_memory_id"], "hopedit_shard_00000")
        self.assertEqual(decision["selected_slot_ids"], ["hopedit_atom_00000"])
        self.assertIsNotNone(decision["adapter_name"])

    def test_v3_support_overlap_veto_opens_new_shard(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v3_side_memory"
        hparams.state_memory_impl = "side_memory_shards"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller.shard_registry = {
            "hopedit_shard_00000": {
                "shard_id": "hopedit_shard_00000",
                "member_edit_ids": ["hopedit_00000"],
                "atoms": [],
                "shard_prototypes": [],
                "prototype_dispersion": 0.0,
                "prototype_margin_history": [],
            }
        }
        controller.memory_entries = [
            {"edit_id": "hopedit_00000", "shard_id": "hopedit_shard_00000", "support_atom_ids": ["hopedit_atom_00000"]}
        ]
        overlap = controller._support_overlap_with_entries("hopedit_shard_00000", ["hopedit_atom_00000"])
        self.assertAlmostEqual(overlap, 1.0, places=5)
        controller.support_exclusivity_failures = 1
        diagnostics = controller.export_support_diagnostics()
        self.assertEqual(diagnostics["support_exclusivity_failures"], 1)

    def test_v3_runtime_checkpoint_round_trip(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v3_side_memory"
        hparams.state_memory_impl = "side_memory_shards"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller.memory_entries = [
            {
                "edit_id": "hopedit_00000",
                "prompt": "p0",
                "target_new": "t0",
                "raw_semantic_key": torch.tensor([1.0, 0.0]),
                "raw_activation_key": torch.tensor([1.0, 0.0]),
                "semantic_key": torch.tensor([1.0, 0.0]),
                "activation_key": torch.tensor([1.0, 0.0]),
                "view_keys": [{"view_name": "prompt", "text": "p0", "raw_semantic_key": torch.tensor([1.0, 0.0]), "raw_activation_key": torch.tensor([1.0, 0.0]), "semantic_key": torch.tensor([1.0, 0.0]), "activation_key": torch.tensor([1.0, 0.0])}],
                "shard_id": "hopedit_shard_00000",
                "support_atom_ids": ["hopedit_atom_00000"],
                "support_amplitudes": [1.0],
            }
        ]
        controller.shard_registry = {
            "hopedit_shard_00000": {
                "shard_id": "hopedit_shard_00000",
                "member_edit_ids": ["hopedit_00000"],
                "atoms": [
                    {
                        "atom_id": "hopedit_atom_00000",
                        "group_name": "attention",
                        "atom_rank": 2,
                        "atom_weights": {},
                        "usage_count": 1,
                        "member_edit_ids": ["hopedit_00000"],
                        "view_keys": [{"view_name": "prompt", "text": "p0", "raw_semantic_key": torch.tensor([1.0, 0.0]), "raw_activation_key": torch.tensor([1.0, 0.0]), "semantic_key": torch.tensor([1.0, 0.0]), "activation_key": torch.tensor([1.0, 0.0])}],
                        "atom_prototypes": [{"view_name": "prompt", "text": "p0", "semantic_key": torch.tensor([1.0, 0.0]), "activation_key": torch.tensor([1.0, 0.0]), "prototype_dispersion": 0.0}],
                        "atom_dispersion": 0.0,
                    }
                ],
                "shard_view_keys": [],
                "shard_prototypes": [{"view_name": "prompt", "text": "p0", "semantic_key": torch.tensor([1.0, 0.0]), "activation_key": torch.tensor([1.0, 0.0]), "prototype_dispersion": 0.0}],
                "prototype_dispersion": 0.0,
                "prototype_margin_history": [],
            }
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            controller.save_runtime_checkpoint(checkpoint_dir)
            restored = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
            restored.load_runtime_checkpoint(checkpoint_dir, is_trainable=False)
            self.assertEqual(restored.memory_unit, "shard")
            self.assertIn("hopedit_shard_00000", restored.shard_registry)
            self.assertEqual(restored.memory_entries[0]["support_atom_ids"], ["hopedit_atom_00000"])

    def test_v4_trace_route_selects_best_trace(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_trace_bank"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller.memory_entries = [
            {
                "edit_id": "hopedit_00000",
                "trace_id": "hopedit_00000",
                "raw_semantic_key": torch.tensor([1.0, 0.0]),
                "raw_activation_key": torch.tensor([1.0, 0.0]),
                "semantic_key": torch.tensor([1.0, 0.0]),
                "activation_key": torch.tensor([1.0, 0.0]),
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": "p0",
                        "raw_semantic_key": torch.tensor([1.0, 0.0]),
                        "raw_activation_key": torch.tensor([1.0, 0.0]),
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                    }
                ],
            },
            {
                "edit_id": "hopedit_00001",
                "trace_id": "hopedit_00001",
                "raw_semantic_key": torch.tensor([0.0, 1.0]),
                "raw_activation_key": torch.tensor([0.0, 1.0]),
                "semantic_key": torch.tensor([0.0, 1.0]),
                "activation_key": torch.tensor([0.0, 1.0]),
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": "p1",
                        "raw_semantic_key": torch.tensor([0.0, 1.0]),
                        "raw_activation_key": torch.tensor([0.0, 1.0]),
                        "semantic_key": torch.tensor([0.0, 1.0]),
                        "activation_key": torch.tensor([0.0, 1.0]),
                    }
                ],
            },
        ]
        decision = controller._route_from_keys(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        self.assertEqual(decision["memory_unit"], "trace")
        self.assertEqual(decision["chosen_memory_id"], "hopedit_00000")
        self.assertEqual(decision["chosen_trace_id"], "hopedit_00000")
        self.assertEqual(decision["adapter_name"], "hopedit_00000")
        self.assertEqual(decision["top_memory_ids"][0], "hopedit_00000")

    def test_v4_runtime_checkpoint_round_trip(self):
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
        hparams = make_hparams()
        hparams.hopedit_mode = "v4_trace_bank"
        controller = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
        controller._ensure_adapter("hopedit_00000")
        controller.memory_entries = [
            {
                "edit_id": "hopedit_00000",
                "trace_id": "hopedit_00000",
                "prompt": "p0",
                "target_new": "t0",
                "raw_semantic_key": torch.tensor([1.0, 0.0]),
                "raw_activation_key": torch.tensor([1.0, 0.0]),
                "semantic_key": torch.tensor([1.0, 0.0]),
                "activation_key": torch.tensor([1.0, 0.0]),
                "view_keys": [
                    {
                        "view_name": "prompt",
                        "text": "p0",
                        "raw_semantic_key": torch.tensor([1.0, 0.0]),
                        "raw_activation_key": torch.tensor([1.0, 0.0]),
                        "semantic_key": torch.tensor([1.0, 0.0]),
                        "activation_key": torch.tensor([1.0, 0.0]),
                    }
                ],
                "trace_address": {"num_views": 1, "view_names": ["prompt"]},
                "trace_value_impl": "exact_lora",
                "value_adapter_name": "hopedit_00000",
            }
        ]
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            controller.save_runtime_checkpoint(checkpoint_dir)
            restored = HopEditController(model=GPT2LMHeadModel(config), tok=DummyTokenizer(), hparams=hparams)
            restored.load_runtime_checkpoint(checkpoint_dir, is_trainable=False)
            self.assertEqual(restored.hparams.hopedit_mode, "v4_trace_bank")
            self.assertEqual(restored.memory_unit, "trace")
            self.assertEqual(restored.memory_entries[0]["trace_id"], "hopedit_00000")
            self.assertEqual(restored.memory_entries[0]["trace_value_impl"], "exact_lora")


if __name__ == "__main__":
    unittest.main()
