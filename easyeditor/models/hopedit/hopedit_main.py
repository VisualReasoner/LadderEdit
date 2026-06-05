from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
from collections import OrderedDict, defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hopedit_hparams import HopEditHyperParams


STATE_GATE_FEATURE_NAMES = [
    "direct_top1_prob",
    "direct_route_margin",
    "direct_top1_score",
    "direct_top2_score",
    "top4_entropy",
    "prototype_dispersion",
    "cross_view_route_gap",
    "state_stability_score",
    "state_member_count",
    "state_tier_consolidated",
    "bucket_dispersion",
]


class FactoredRelationResidualEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = max(1, int(hidden_dim))
        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, raw_factor: torch.Tensor) -> torch.Tensor:
        return raw_factor + self.net(self.norm(raw_factor))


class HopEditController(nn.Module):
    def __init__(self, model: AutoModelForCausalLM, tok: AutoTokenizer, hparams: HopEditHyperParams):
        super().__init__()
        self.model = model
        self.tok = tok
        self.hparams = hparams
        self.memory_entries: list[dict[str, Any]] = []
        self.edit_registry: dict[str, dict[str, Any]] = {}
        self.cell_registry: dict[str, dict[str, Any]] = {}
        self.shard_registry: dict[str, dict[str, Any]] = {}
        self.bucket_registry: dict[str, dict[str, Any]] = {}
        self.address_dictionary: dict[str, Any] = {"atoms": [], "usage_counts": []}
        self.address_postings: dict[int, list[str]] = {}
        self.family_postings: dict[str, list[str]] = {}
        self.address_version = 0
        self.route_logs: list[dict[str, Any]] = []
        self.disabled_adapters: set[str] = set()
        self.cached_activation_stats: dict[str, torch.Tensor] | None = None
        self.trace_value_store: dict[str, dict[str, torch.Tensor]] = {}
        self.trace_value_cache: OrderedDict[str, str] = OrderedDict()
        self.trace_value_cache_hits = 0
        self.trace_value_cache_misses = 0
        self.factored_relation_encoder: FactoredRelationResidualEncoder | None = None
        self.factored_relation_encoder_input_dim: int | None = None
        self.factored_relation_encoder_updates = 0
        self.factored_relation_encoder_last_loss: float | None = None
        self.factored_relation_encoder_checkpoint_loaded: str | None = None
        self.factored_relation_encoder_checkpoint_metadata: dict[str, Any] = {}
        self.factored_relation_score_transform_signature: tuple[Any, ...] | None = None
        self.factored_relation_score_transform_cache: dict[str, Any] = {}
        self.factored_capsule_config: dict[str, Any] | None = None
        self.factored_capsule_config_loaded_from: str | None = None
        self.edit_index = 0
        self.cell_index = 0
        self.shard_index = 0
        self.atom_index = 0
        self.consolidation_attempts = 0
        self.consolidation_accepted = 0
        self.consolidation_rejected_by_reason: dict[str, int] = {}
        self.slot_transfer_attempts = 0
        self.slot_transfer_accepted = 0
        self.slot_transfer_rejected_by_reason: dict[str, int] = {}
        self.support_exclusivity_failures = 0
        self.base_only_fallback_count = 0
        self.runtime_composed_signature: tuple[str, ...] | None = None
        if self.hparams.route_log_dir:
            os.makedirs(self.hparams.route_log_dir, exist_ok=True)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()
        self._init_state_gate()

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            model = super().__getattribute__("_modules").get("model")
            if model is not None:
                return getattr(model, name)
            raise

    @property
    def device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    @property
    def config(self):
        return self.model.config

    @property
    def is_v2(self) -> bool:
        return getattr(self.hparams, "hopedit_mode", "v1_per_edit") == "v2_cell_bank"

    @property
    def is_v3(self) -> bool:
        return getattr(self.hparams, "hopedit_mode", "v1_per_edit") == "v3_side_memory"

    @property
    def is_v4(self) -> bool:
        return getattr(self.hparams, "hopedit_mode", "v1_per_edit") in {
            "v4_trace_bank",
            "v4_sparse_address_trace_bank",
            "v4_overlap_aware_anchor_trace_bank",
            "v4_factored_address_trace_bank",
        }

    @property
    def is_v4_trace_bank(self) -> bool:
        return getattr(self.hparams, "hopedit_mode", "v1_per_edit") == "v4_trace_bank"

    @property
    def is_v4_sparse_address_trace_bank(self) -> bool:
        return getattr(self.hparams, "hopedit_mode", "v1_per_edit") == "v4_sparse_address_trace_bank"

    @property
    def is_v4_overlap_aware_anchor_trace_bank(self) -> bool:
        return getattr(self.hparams, "hopedit_mode", "v1_per_edit") == "v4_overlap_aware_anchor_trace_bank"

    @property
    def is_v4_factored_address_trace_bank(self) -> bool:
        return getattr(self.hparams, "hopedit_mode", "v1_per_edit") == "v4_factored_address_trace_bank"

    @property
    def memory_unit(self) -> str:
        if self._use_side_memory_shards():
            return "shard"
        if self._use_sparse_slots():
            return "state"
        if self.is_v4:
            return "trace"
        return "cell" if self.is_v2 else "edit"

    def _route_top_k(self) -> int:
        if self.is_v3:
            return max(1, int(getattr(self.hparams, "side_memory_shard_shortlist", 2) or 2))
        if self.is_v2:
            return max(1, int(getattr(self.hparams, "cell_router_topk", self.hparams.top_k)))
        return max(1, int(self.hparams.top_k))

    @contextmanager
    def _peft_base_model_disabled(self):
        if not isinstance(self.model, PeftModel):
            yield
            return
        base_model = getattr(self.model, "base_model", None)
        if base_model is None or not hasattr(base_model, "disable_adapter_layers"):
            yield
            return
        enabled_before = None
        if hasattr(self.model, "get_model_status"):
            try:
                enabled_before = self.model.get_model_status().enabled
            except Exception:
                enabled_before = None
        try:
            base_model.disable_adapter_layers()
            self.model._adapters_disabled = True
            yield
        finally:
            if hasattr(base_model, "enable_adapter_layers") and enabled_before is not False:
                try:
                    base_model.enable_adapter_layers()
                except Exception:
                    pass
            self.model._adapters_disabled = False

    def _adapter_disabled(self):
        if isinstance(self.model, PeftModel):
            self._repair_active_adapter_reference()
            active_adapter = getattr(self.model, "active_adapter", None)
            if active_adapter not in getattr(self.model, "peft_config", {}):
                return self._peft_base_model_disabled()
        if hasattr(self.model, "disable_adapter"):
            return self.model.disable_adapter()
        return nullcontext()

    def _repair_active_adapter_reference(self) -> None:
        if not isinstance(self.model, PeftModel):
            return
        peft_config = getattr(self.model, "peft_config", {})
        active_adapter = getattr(self.model, "active_adapter", None)
        if active_adapter in peft_config:
            return
        runtime_adapter = self._runtime_slot_adapter_name() if self._use_sparse_slots() else None
        fallback_adapter = None
        if runtime_adapter is not None and runtime_adapter in peft_config:
            fallback_adapter = runtime_adapter
        elif peft_config:
            fallback_adapter = next(iter(peft_config.keys()))
        if fallback_adapter is not None and hasattr(self.model, "set_adapter"):
            self.model.set_adapter(fallback_adapter)
            return
        if hasattr(self.model, "active_adapter"):
            try:
                self.model.active_adapter = None
            except Exception:
                pass

    def _next_edit_id(self) -> str:
        edit_id = f"hopedit_{self.edit_index:05d}"
        self.edit_index += 1
        return edit_id

    def _next_cell_id(self) -> str:
        cell_id = f"hopedit_cell_{self.cell_index:05d}"
        self.cell_index += 1
        return cell_id

    def _next_shard_id(self) -> str:
        shard_id = f"hopedit_shard_{self.shard_index:05d}"
        self.shard_index += 1
        return shard_id

    def _next_atom_id(self) -> str:
        atom_id = f"hopedit_atom_{self.atom_index:05d}"
        self.atom_index += 1
        return atom_id

    @staticmethod
    def _clone_to_cpu(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: HopEditController._clone_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [HopEditController._clone_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(HopEditController._clone_to_cpu(item) for item in value)
        return value

    @staticmethod
    def _clone_to_json(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {key: HopEditController._clone_to_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [HopEditController._clone_to_json(item) for item in value]
        if isinstance(value, tuple):
            return [HopEditController._clone_to_json(item) for item in value]
        if isinstance(value, (np.generic,)):
            return value.item()
        return value

    def _format_prompt(self, request: dict[str, Any]) -> str:
        prompt = request["prompt"]
        if "{}" in prompt and request.get("subject") is not None:
            return prompt.format(request["subject"])
        return prompt

    def _subject_conditioned_prompt(self, request: dict[str, Any], prompt: str) -> str:
        subject = request.get("subject")
        if not subject or subject in prompt:
            return prompt
        return f"{subject}. {prompt}"

    def _make_lora_config(self, *, rank: int | None = None, alpha: float | None = None) -> LoraConfig:
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.hparams.rank if rank is None else int(rank),
            lora_alpha=self.hparams.lora_alpha if alpha is None else float(alpha),
            lora_dropout=self.hparams.lora_dropout,
            target_modules=self.hparams.target_modules,
        )

    def _ensure_adapter(self, adapter_name: str, *, rank: int | None = None, alpha: float | None = None) -> None:
        self.model.config.use_cache = False
        if hasattr(self.model, "supports_gradient_checkpointing"):
            self.model.supports_gradient_checkpointing = True
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        if isinstance(self.model, PeftModel):
            if adapter_name not in getattr(self.model, "peft_config", {}):
                self.model.add_adapter(adapter_name, self._make_lora_config(rank=rank, alpha=alpha))
        else:
            self.model = get_peft_model(self.model, self._make_lora_config(rank=rank, alpha=alpha), adapter_name=adapter_name)
            self.model.is_parallelizable = True
            self.model.model_parallel = True

    def _set_runtime_adapter(self, adapter_name: str) -> None:
        if adapter_name is not None and hasattr(self.model, "set_adapter"):
            self.model.set_adapter(adapter_name)

    def _model_for_direct_call(self):
        if isinstance(self.model, PeftModel):
            self._repair_active_adapter_reference()
            active_adapter = getattr(self.model, "active_adapter", None)
            if active_adapter not in getattr(self.model, "peft_config", {}):
                base_model = getattr(self.model, "base_model", None)
                if base_model is not None:
                    return getattr(base_model, "model", base_model)
        return self.model

    def _configure_trainable_adapter(self, adapter_name: str) -> list[torch.nn.Parameter]:
        self._set_runtime_adapter(adapter_name)
        trainable = []
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = False
            if "lora_" in name and adapter_name in name:
                parameter.requires_grad = True
                trainable.append(parameter)
        return trainable

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        tokens = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.hparams.max_length,
        )
        return {key: value.to(self.device) for key, value in tokens.items()}

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    @staticmethod
    def _masked_mean(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.numel() == 0 or int(mask.sum().item()) <= 0:
            return hidden_states.mean(dim=0)
        mask = mask.to(hidden_states.device, dtype=hidden_states.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (hidden_states * mask.unsqueeze(-1)).sum(dim=0) / denom

    @staticmethod
    def _masked_last(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.numel() == 0 or int(mask.sum().item()) <= 0:
            return hidden_states[-1]
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if indices.numel() == 0:
            return hidden_states[-1]
        return hidden_states[int(indices[-1].item())]

    def _token_ids_without_special(self, text: str) -> list[int]:
        text = str(text or "")
        if not text:
            return []
        if hasattr(self.tok, "encode"):
            try:
                return list(self.tok.encode(text, add_special_tokens=False))
            except TypeError:
                pass
        try:
            tokens = self.tok(
                [text],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.hparams.max_length,
                add_special_tokens=False,
            )
            input_ids = tokens.get("input_ids")
            if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2:
                return input_ids[0].detach().cpu().tolist()
        except TypeError:
            pass
        words = [token for token in text.split() if token]
        return list(range(1, len(words) + 1))

    @staticmethod
    def _find_subsequence_span(sequence: list[int], subsequence: list[int]) -> tuple[int, int] | None:
        if not sequence or not subsequence or len(subsequence) > len(sequence):
            return None
        for start in range(len(sequence) - len(subsequence) + 1):
            if sequence[start : start + len(subsequence)] == subsequence:
                return start, start + len(subsequence)
        return None

    @staticmethod
    def _find_word_span(text: str, needle: str) -> tuple[int, int] | None:
        text_words = [token for token in str(text or "").split() if token]
        needle_words = [token for token in str(needle or "").split() if token]
        if not text_words or not needle_words or len(needle_words) > len(text_words):
            return None
        def normalize_word(token: str) -> str:
            normalized = re.sub(r"^\W+|\W+$", "", str(token or "").lower())
            normalized = re.sub(r"(?:'s|’s)$", "", normalized)
            return normalized

        lowered_text = [normalize_word(token) for token in text_words]
        lowered_needle = [normalize_word(token) for token in needle_words]
        for start in range(len(lowered_text) - len(lowered_needle) + 1):
            if lowered_text[start : start + len(lowered_needle)] == lowered_needle:
                return start, start + len(lowered_needle)
        return None

    def _token_position_mask(
        self,
        text: str,
        needle: str | None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        valid_length = int(attention_mask.sum().item()) if isinstance(attention_mask, torch.Tensor) else int(input_ids.numel())
        mask = torch.zeros(valid_length, dtype=torch.bool)
        if not needle:
            return mask
        full_ids = input_ids[:valid_length].detach().cpu().tolist()
        needle_ids = self._token_ids_without_special(str(needle))
        span = self._find_subsequence_span(full_ids, needle_ids)
        if span is not None:
            mask[span[0] : span[1]] = True
            return mask
        word_span = self._find_word_span(text, str(needle))
        if word_span is None:
            return mask
        text_word_count = max(1, len([token for token in str(text or "").split() if token]))
        offset = max(0, valid_length - text_word_count)
        start = min(valid_length, offset + word_span[0])
        end = min(valid_length, offset + word_span[1])
        if end > start:
            mask[start:end] = True
        return mask

    def _extract_batched_factored_address_keys(
        self,
        texts: list[str],
        subject_texts: list[str | None],
        object_texts: list[str | None],
        *,
        subject_layer_override: int | None = None,
        relation_layer_override: int | None = None,
        subject_pooling_override: str | None = None,
    ) -> list[dict[str, torch.Tensor | None]]:
        if not texts:
            return []
        try:
            tokens = self.tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.hparams.max_length,
                return_special_tokens_mask=True,
            )
        except TypeError:
            tokens = self.tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.hparams.max_length,
            )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.no_grad():
            with self._adapter_disabled():
                outputs = self._model_for_direct_call()(**tokens, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        subject_layer_idx = self._factored_subject_layer(len(hidden_states))
        if subject_layer_override is not None:
            subject_layer_idx = self._resolve_factored_layer(subject_layer_override, len(hidden_states))
        relation_layer_idx = self._factored_relation_layer(len(hidden_states))
        if relation_layer_override is not None:
            relation_layer_idx = self._resolve_factored_layer(relation_layer_override, len(hidden_states))
        subject_hidden = hidden_states[subject_layer_idx].detach().float().cpu()
        relation_hidden = hidden_states[relation_layer_idx].detach().float().cpu()
        attention_mask = tokens.get("attention_mask")
        input_ids = tokens["input_ids"]
        special_tokens_mask = tokens.get("special_tokens_mask")
        results: list[dict[str, torch.Tensor | None]] = []
        for batch_index, text in enumerate(texts):
            row_input_ids = input_ids[batch_index].detach().cpu()
            row_attention = None if attention_mask is None else attention_mask[batch_index].detach().cpu()
            row_special = None if special_tokens_mask is None else special_tokens_mask[batch_index].detach().cpu()
            if row_attention is not None:
                valid_positions = torch.nonzero(row_attention.to(dtype=torch.bool), as_tuple=False).flatten()
                if valid_positions.numel() == 0:
                    valid_positions = torch.arange(row_input_ids.numel(), dtype=torch.long)
            else:
                valid_positions = torch.arange(row_input_ids.numel(), dtype=torch.long)
            row_subject_hidden = subject_hidden[batch_index].index_select(0, valid_positions)
            row_relation_hidden = relation_hidden[batch_index].index_select(0, valid_positions)
            row_input_ids = row_input_ids.index_select(0, valid_positions)
            row_special = None if row_special is None else row_special.index_select(0, valid_positions)
            valid_length = int(row_input_ids.numel())
            subject_mask = self._token_position_mask(text, subject_texts[batch_index], row_input_ids, None)
            object_mask = self._token_position_mask(text, object_texts[batch_index], row_input_ids, None)
            valid_mask = torch.ones(valid_length, dtype=torch.bool)
            if row_special is not None:
                valid_mask = valid_mask & ~row_special[:valid_length].to(dtype=torch.bool)
            relation_mask = valid_mask & ~subject_mask & ~object_mask
            if int(subject_mask.sum().item()) <= 0:
                subject_factor = None
            elif str(subject_pooling_override or self._factored_subject_pooling()).strip().lower() == "mean":
                subject_factor = self._masked_mean(row_subject_hidden, subject_mask)
            else:
                subject_factor = self._masked_last(row_subject_hidden, subject_mask)
            relation_raw_factor = self._masked_mean(row_relation_hidden, relation_mask)
            relation_factor = self._encode_factored_relation_factor(relation_raw_factor)
            results.append(
                {
                    "subject_factor": None if subject_factor is None else self._l2_normalize(subject_factor.detach().float().cpu()),
                    "relation_factor": relation_factor,
                    "relation_raw_factor": relation_raw_factor.detach().float().cpu(),
                    "subject_found": bool(subject_mask.any().item()),
                    "relation_token_count": int(relation_mask.sum().item()),
                    "layer_index": int(subject_layer_idx),
                    "subject_layer_index": int(subject_layer_idx),
                    "relation_layer_index": int(relation_layer_idx),
                    "subject_pooling_mode": str(subject_pooling_override or self._factored_subject_pooling()).strip().lower(),
                }
            )
        return results

    def _resolve_probe_layers(self, hidden_count: int) -> list[int]:
        configured_layers = self.hparams.probe_layers if self.hparams.probe_layers else [self.hparams.probe_layer]
        resolved_layers = []
        for layer in configured_layers:
            resolved_layer = hidden_count + layer if layer < 0 else layer
            resolved_layer = max(0, min(resolved_layer, hidden_count - 1))
            resolved_layers.append(resolved_layer)
        return sorted(set(resolved_layers))

    @staticmethod
    def _l2_normalize(key: torch.Tensor) -> torch.Tensor:
        norm = key.norm(p=2).clamp_min(1e-12)
        return key / norm

    def _factored_relation_encoder_enabled(self) -> bool:
        return str(getattr(self.hparams, "factored_relation_encoder_impl", "identity") or "identity").strip().lower() == "residual_mlp"

    def _factored_relation_encoder_checkpoint_path(self) -> str | None:
        checkpoint_path = getattr(self.hparams, "factored_relation_encoder_checkpoint", None)
        if checkpoint_path is None:
            return None
        checkpoint_path = str(checkpoint_path).strip()
        return checkpoint_path or None

    @staticmethod
    def _relation_encoder_state_from_payload(payload: Any) -> dict[str, torch.Tensor]:
        if isinstance(payload, dict):
            for key in ("encoder_state_dict", "state_dict", "model_state_dict"):
                state = payload.get(key)
                if isinstance(state, dict):
                    return state
            if payload and all(isinstance(key, str) for key in payload.keys()):
                if any(key.startswith(("norm.", "net.")) for key in payload.keys()):
                    return payload
        raise ValueError("Relation encoder checkpoint must contain encoder_state_dict or a raw state_dict.")

    def _load_factored_relation_encoder_checkpoint(self, checkpoint_path: str, input_dim: int) -> FactoredRelationResidualEncoder:
        payload = torch.load(checkpoint_path, map_location="cpu")
        metadata = payload if isinstance(payload, dict) else {}
        checkpoint_input_dim = int(metadata.get("input_dim", input_dim) or input_dim)
        if checkpoint_input_dim != int(input_dim):
            raise ValueError(
                f"Relation encoder checkpoint input_dim={checkpoint_input_dim} does not match runtime input_dim={input_dim}."
            )
        hidden_dim = int(metadata.get("hidden_dim", getattr(self.hparams, "factored_relation_encoder_hidden_dim", 512)) or 512)
        encoder = FactoredRelationResidualEncoder(int(input_dim), hidden_dim)
        encoder.load_state_dict(self._relation_encoder_state_from_payload(payload))
        encoder.eval()
        self.factored_relation_encoder = encoder
        self.factored_relation_encoder_input_dim = int(input_dim)
        self.factored_relation_encoder_checkpoint_loaded = checkpoint_path
        self.factored_relation_encoder_checkpoint_metadata = {
            "input_dim": checkpoint_input_dim,
            "hidden_dim": hidden_dim,
            "train_count": metadata.get("train_count"),
            "dev_count": metadata.get("dev_count"),
            "best_dev_q_r": metadata.get("best_dev_q_r"),
            "epoch": metadata.get("epoch"),
        }
        return encoder

    def _ensure_factored_relation_encoder(self, input_dim: int) -> FactoredRelationResidualEncoder | None:
        if not self._factored_relation_encoder_enabled():
            return None
        input_dim = int(input_dim)
        if input_dim <= 0:
            return None
        checkpoint_path = self._factored_relation_encoder_checkpoint_path()
        if (
            checkpoint_path is not None
            and self.factored_relation_encoder_checkpoint_loaded != checkpoint_path
        ):
            return self._load_factored_relation_encoder_checkpoint(checkpoint_path, input_dim)
        if self.factored_relation_encoder is None or self.factored_relation_encoder_input_dim != input_dim:
            hidden_dim = int(getattr(self.hparams, "factored_relation_encoder_hidden_dim", 512) or 512)
            self.factored_relation_encoder = FactoredRelationResidualEncoder(input_dim, hidden_dim)
            self.factored_relation_encoder_input_dim = input_dim
        return self.factored_relation_encoder

    def _encode_factored_relation_factor(self, raw_factor: torch.Tensor) -> torch.Tensor:
        raw_factor = raw_factor.detach().float().cpu()
        encoder = self._ensure_factored_relation_encoder(int(raw_factor.numel()))
        if encoder is None:
            return self._l2_normalize(raw_factor)
        encoder.eval()
        with torch.no_grad():
            encoded = encoder(raw_factor.unsqueeze(0)).squeeze(0).detach().float().cpu()
        return self._l2_normalize(encoded)

    def _normalize_semantic_key(self, raw_key: torch.Tensor) -> torch.Tensor:
        key = raw_key.detach().float().cpu()
        if self.hparams.key_l2_normalize:
            key = self._l2_normalize(key)
        return key

    def _activation_stats_from_raw_keys(self, raw_keys: list[torch.Tensor]) -> dict[str, torch.Tensor] | None:
        if len(raw_keys) < 2 or not self.hparams.activation_whiten:
            return None
        stacked = torch.stack([key.detach().float().cpu() for key in raw_keys], dim=0)
        return {
            "mean": stacked.mean(dim=0),
            "std": stacked.std(dim=0, unbiased=False).clamp_min(1e-6),
        }

    def _normalize_activation_key(self, raw_key: torch.Tensor, stats: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        key = raw_key.detach().float().cpu()
        if stats is not None:
            if self.hparams.activation_center:
                key = key - stats["mean"]
            if self.hparams.activation_whiten:
                key = key / stats["std"]
        elif self.hparams.activation_center:
            key = key - key.mean()
            if self.hparams.activation_whiten:
                key = key / key.std(unbiased=False).clamp_min(1e-6)
        if self.hparams.key_l2_normalize:
            key = self._l2_normalize(key)
        return key

    def _entry_view_records(self, entry: dict[str, Any]) -> list[dict[str, Any]]:
        view_records = entry.get("view_keys") or []
        if view_records:
            return view_records
        return [
            {
                "view_name": "anchor",
                "text": entry.get("prompt"),
                "raw_semantic_key": entry.get("raw_semantic_key"),
                "raw_activation_key": entry.get("raw_activation_key"),
                "semantic_key": entry.get("semantic_key"),
                "activation_key": entry.get("activation_key"),
            }
        ]

    @staticmethod
    def _normalize_family_token(text: str | None) -> str:
        return " ".join(str(text or "").strip().lower().split())

    def _trace_family_ids_from_entry(self, entry: dict[str, Any]) -> list[str]:
        if entry.get("trace_family_ids"):
            return list(entry.get("trace_family_ids") or [])
        prompt = entry.get("prompt")
        subject = entry.get("subject")
        target_new = entry.get("target_new")
        template_key = self._trace_family_key_template(prompt, subject)
        family_ids = []
        if template_key:
            family_ids.append(f"template::{template_key}")
        normalized_subject = self._normalize_family_token(subject)
        if normalized_subject:
            family_ids.append(f"subject::{normalized_subject}")
        normalized_target = self._normalize_family_token(target_new)
        if normalized_target:
            family_ids.append(f"target::{normalized_target}")
        deduped = []
        seen = set()
        for family_id in family_ids:
            if family_id not in seen:
                deduped.append(family_id)
                seen.add(family_id)
        return deduped

    def _assign_trace_family_ids(self, entry: dict[str, Any]) -> list[str]:
        family_ids = self._trace_family_ids_from_entry(entry)
        entry["trace_family_ids"] = family_ids
        entry["trace_primary_family_id"] = family_ids[0] if family_ids else None
        return family_ids

    def _trace_runtime_cache_adapter_name(self, value_ref: str) -> str:
        return str(value_ref)

    def _trace_runtime_adapter_refs(self) -> list[str]:
        return list(self.trace_value_cache.keys())

    def _drop_runtime_adapter_without_disabling(self, adapter_name: str) -> None:
        if not adapter_name or not hasattr(self.model, "delete_adapter"):
            return
        if adapter_name not in getattr(self.model, "peft_config", {}):
            return
        try:
            self.model.delete_adapter(adapter_name)
        except Exception:
            return
        self._repair_active_adapter_reference()

    def _store_trace_value_reference(self, value_ref: str, adapter_name: str) -> None:
        self.trace_value_store[str(value_ref)] = self._capture_adapter_parameters(adapter_name)

    def _touch_trace_value_cache(self, value_ref: str) -> None:
        adapter_name = self.trace_value_cache.pop(value_ref, None)
        if adapter_name is None:
            return
        self.trace_value_cache[value_ref] = adapter_name

    def _evict_trace_value_cache_if_needed(self) -> None:
        limit = self._trace_value_cache_size()
        while len(self.trace_value_cache) > limit:
            stale_ref, stale_adapter = self.trace_value_cache.popitem(last=False)
            self._drop_runtime_adapter_without_disabling(stale_adapter)

    def _materialize_trace_value(self, value_ref: str | None) -> tuple[str | None, bool]:
        if not value_ref:
            return None, False
        value_ref = str(value_ref)
        if value_ref in self.trace_value_cache:
            self.trace_value_cache_hits += 1
            self._touch_trace_value_cache(value_ref)
            return self.trace_value_cache[value_ref], True
        weights = self.trace_value_store.get(value_ref)
        if weights is None:
            return None, False
        adapter_name = self._trace_runtime_cache_adapter_name(value_ref)
        self._ensure_adapter(adapter_name)
        self._load_adapter_parameters(adapter_name, weights)
        self.trace_value_cache[value_ref] = adapter_name
        self.trace_value_cache_misses += 1
        self._evict_trace_value_cache_if_needed()
        return adapter_name, False

    def _remove_trace_value_reference(self, value_ref: str | None) -> None:
        if not value_ref:
            return
        value_ref = str(value_ref)
        adapter_name = self.trace_value_cache.pop(value_ref, None)
        if adapter_name is not None:
            self._drop_runtime_adapter_without_disabling(adapter_name)
        self.trace_value_store.pop(value_ref, None)

    def _use_sparse_slots(self) -> bool:
        return bool(self.is_v2 and getattr(self.hparams, "state_memory_impl", "cell_bank") == "sparse_slots")

    def _use_side_memory_shards(self) -> bool:
        return bool(self.is_v3 and getattr(self.hparams, "state_memory_impl", "side_memory_shards") == "side_memory_shards")

    def _use_sparse_address_trace_bank(self) -> bool:
        return bool(
            self.is_v4_sparse_address_trace_bank
            or self.is_v4_overlap_aware_anchor_trace_bank
            or self.is_v4_factored_address_trace_bank
        )

    def _use_overlap_aware_anchor_trace_bank(self) -> bool:
        return bool(self.is_v4_overlap_aware_anchor_trace_bank)

    def _use_factored_address_trace_bank(self) -> bool:
        return bool(self.is_v4_factored_address_trace_bank)

    def _use_cold_trace_values(self) -> bool:
        return bool(self._use_overlap_aware_anchor_trace_bank() or self._use_factored_address_trace_bank())

    def _address_num_atoms(self) -> int:
        return max(1, int(getattr(self.hparams, "address_num_atoms", 256) or 256))

    def _address_code_topk(self) -> int:
        return max(1, int(getattr(self.hparams, "address_code_topk", 4) or 4))

    def _address_candidate_budget(self) -> int:
        return max(1, int(getattr(self.hparams, "address_candidate_budget", 64) or 64))

    def _trace_family_budget(self) -> int:
        return max(1, int(getattr(self.hparams, "trace_family_budget", 6) or 6))

    def _trace_family_boost(self) -> float:
        return float(getattr(self.hparams, "trace_family_boost", 0.35) or 0.35)

    def _address_atom_merge_threshold(self) -> float:
        return float(getattr(self.hparams, "address_atom_merge_threshold", 0.92) or 0.92)

    def _address_code_min_affinity(self) -> float:
        return float(getattr(self.hparams, "address_code_min_affinity", 1.0e-6) or 1.0e-6)

    def _address_coherence_penalty(self) -> float:
        return float(getattr(self.hparams, "address_coherence_penalty", 0.05) or 0.05)

    def _trace_energy_beta(self) -> float:
        return float(getattr(self.hparams, "trace_energy_beta", 12.0) or 12.0)

    def _trace_energy_temperature(self) -> float:
        return float(getattr(self.hparams, "trace_energy_temperature", 1.0) or 1.0)

    def _trace_abstain_margin(self) -> float:
        return float(getattr(self.hparams, "trace_abstain_margin", 0.03) or 0.03)

    def _trace_abstain_min_energy(self) -> float:
        return float(getattr(self.hparams, "trace_abstain_min_energy", 0.10) or 0.10)

    def _trace_anchor_weight(self) -> float:
        return float(getattr(self.hparams, "trace_anchor_weight", 0.25) or 0.25)

    def _trace_anchor_energy_floor(self) -> float:
        return float(getattr(self.hparams, "trace_anchor_energy_floor", 0.05) or 0.05)

    def _trace_family_negative_topk(self) -> int:
        return max(1, int(getattr(self.hparams, "trace_family_negative_topk", 8) or 8))

    def _trace_irrelevant_negative_topk(self) -> int:
        return max(1, int(getattr(self.hparams, "trace_irrelevant_negative_topk", 4) or 4))

    def _trace_family_margin_scale(self) -> float:
        return float(getattr(self.hparams, "trace_family_margin_scale", 0.60) or 0.60)

    def _trace_anchor_margin_scale(self) -> float:
        return float(getattr(self.hparams, "trace_anchor_margin_scale", 0.60) or 0.60)

    def _trace_value_cache_size(self) -> int:
        return max(1, int(getattr(self.hparams, "trace_value_cache_size", 8) or 8))

    def _factored_subject_weight(self) -> float:
        return float(getattr(self.hparams, "factored_subject_weight", 0.5) or 0.5)

    def _factored_relation_weight(self) -> float:
        return float(getattr(self.hparams, "factored_relation_weight", 0.5) or 0.5)

    def _factored_threshold(self, name: str, default: float) -> float:
        value = getattr(self.hparams, name, None)
        return float(default) if value is None else float(value)

    def _resolve_factored_layer(self, configured: int | None, hidden_count: int) -> int:
        if configured is not None:
            resolved = hidden_count + int(configured) if int(configured) < 0 else int(configured)
            return max(0, min(resolved, hidden_count - 1))
        resolved_layers = self._resolve_probe_layers(hidden_count)
        if not resolved_layers:
            return max(0, hidden_count - 1)
        return int(resolved_layers[len(resolved_layers) // 2])

    def _factored_address_layer(self, hidden_count: int) -> int:
        configured = getattr(self.hparams, "factored_address_layer", None)
        return self._resolve_factored_layer(configured, hidden_count)

    def _factored_subject_layer(self, hidden_count: int) -> int:
        configured = getattr(self.hparams, "factored_subject_layer", None)
        if configured is None:
            configured = getattr(self.hparams, "factored_address_layer", None)
        return self._resolve_factored_layer(configured, hidden_count)

    def _factored_relation_layer(self, hidden_count: int) -> int:
        configured = getattr(self.hparams, "factored_relation_layer", None)
        if configured is None:
            configured = getattr(self.hparams, "factored_address_layer", None)
        return self._resolve_factored_layer(configured, hidden_count)

    def _factored_subject_pooling(self) -> str:
        pooling = str(getattr(self.hparams, "factored_subject_pooling", "last") or "last").strip().lower()
        if pooling not in {"last", "mean"}:
            return "last"
        return pooling

    def _trace_family_key_template(self, prompt: str | None, subject: str | None) -> str:
        text = " ".join(str(prompt or "").strip().lower().split())
        subj = " ".join(str(subject or "").strip().lower().split())
        if subj:
            text = text.replace(subj, "<subj>")
        return text

    def _slot_capacity(self) -> int:
        return max(1, int(getattr(self.hparams, "state_slot_capacity", 8) or 8))

    def _slot_rank(self) -> int:
        return max(1, int(getattr(self.hparams, "slot_rank", 2) or 2))

    def _shared_basis_rank(self) -> int:
        configured = int(getattr(self.hparams, "state_basis_rank", self._slot_rank()) or self._slot_rank())
        return max(1, configured)

    def _slot_realization_impl(self) -> str:
        return str(getattr(self.hparams, "slot_realization_impl", "concatenated_lora") or "concatenated_lora")

    def _use_shared_basis_codes(self) -> bool:
        return bool(self._use_sparse_slots() and self._slot_realization_impl() in {"shared_basis_codes", "discrete_support_masks"})

    def _use_discrete_support_masks(self) -> bool:
        return bool(self._use_sparse_slots() and self._slot_realization_impl() == "discrete_support_masks")

    def _slot_top_k(self) -> int:
        return max(1, int(getattr(self.hparams, "slot_topk", 2) or 2))

    def _atom_sparse_topk(self) -> int:
        return max(1, int(getattr(self.hparams, "atom_sparse_topk", self._slot_top_k()) or self._slot_top_k()))

    def _atom_write_topk(self) -> int:
        configured = getattr(self.hparams, "atom_write_topk", None)
        if configured is None:
            return self._atom_sparse_topk()
        return max(1, int(configured or self._atom_sparse_topk()))

    def _runtime_slot_adapter_name(self) -> str:
        if self._use_side_memory_shards():
            return "__hopedit_v3_side_memory__"
        return "__hopedit_runtime_slots__"

    def _side_memory_atoms_per_shard(self) -> int:
        return max(1, int(getattr(self.hparams, "atoms_per_shard", 32) or 32))

    def _side_memory_atom_rank(self) -> int:
        return max(1, int(getattr(self.hparams, "atom_rank", 4) or 4))

    def _side_memory_read_topk(self) -> int:
        return max(1, int(getattr(self.hparams, "read_support_topk", 2) or 2))

    def _side_memory_write_topk(self) -> int:
        return max(1, int(getattr(self.hparams, "write_support_topk", 2) or 2))

    def _side_memory_group_name(self, canonical_stem: str) -> str:
        suffix = canonical_stem.split(".")[-1]
        grouping = str(getattr(self.hparams, "module_grouping", "attention") or "attention")
        attention_suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"}
        mlp_suffixes = {"up_proj", "gate_proj", "down_proj", "fc_in", "fc_out"}
        if grouping == "attention" and suffix in attention_suffixes:
            return "attention"
        if grouping in {"attention_mlp", "full"}:
            if suffix in attention_suffixes:
                return "attention"
            if suffix in mlp_suffixes:
                return "mlp"
        return suffix

    def _slot_view_records(self, slot: dict[str, Any]) -> list[dict[str, Any]]:
        prototypes = slot.get("slot_prototypes") or slot.get("view_keys") or []
        if prototypes:
            return prototypes
        semantic_key = slot.get("semantic_key")
        activation_key = slot.get("activation_key")
        if not isinstance(semantic_key, torch.Tensor) or not isinstance(activation_key, torch.Tensor):
            return []
        return [
            {
                "view_name": "slot_anchor",
                "text": slot.get("prompt"),
                "semantic_key": semantic_key,
                "activation_key": activation_key,
                "prototype_dispersion": slot.get("slot_dispersion"),
            }
        ]

    def _state_slots(self, cell: dict[str, Any]) -> list[dict[str, Any]]:
        return list(cell.get("slots") or [])

    @staticmethod
    def _canonical_module_stem(canonical_name: str) -> str:
        for suffix in (".lora_A.__ADAPTER__.weight", ".lora_B.__ADAPTER__.weight"):
            if canonical_name.endswith(suffix):
                return canonical_name[: -len(suffix)]
        return canonical_name

    def _slot_train_scale(self, slot: dict[str, Any]) -> float:
        rank = max(1, int(slot.get("slot_rank") or self._slot_rank()))
        alpha = float(slot.get("slot_alpha") or rank)
        return float(alpha / float(rank))

    def _basis_gram_matrix(self, basis: dict[str, Any]) -> torch.Tensor | None:
        left_basis = basis.get("left_basis")
        right_basis = basis.get("right_basis")
        if not isinstance(left_basis, torch.Tensor) or not isinstance(right_basis, torch.Tensor):
            return None
        left = left_basis.detach().float().cpu()
        right = right_basis.detach().float().cpu()
        return (left.transpose(0, 1) @ left) * (right @ right.transpose(0, 1))

    def _hard_sparse_pursuit_code(
        self,
        affinity: torch.Tensor,
        basis: dict[str, Any],
        *,
        topk: int,
    ) -> torch.Tensor:
        vector = affinity.detach().float().cpu().flatten()
        if vector.numel() == 0:
            return vector
        gram = self._basis_gram_matrix(basis)
        if gram is None or gram.numel() == 0:
            return vector
        limit = min(int(topk), int(vector.numel()))
        coherence = gram.abs()
        penalty_weight = float(getattr(self.hparams, "atom_coherence_penalty", 0.25) or 0.25)
        min_abs_affinity = float(getattr(self.hparams, "atom_sparse_min_abs_affinity", 1.0e-6) or 1.0e-6)
        selected: list[int] = []
        for _ in range(limit):
            penalties = torch.zeros_like(vector)
            if selected:
                penalties = coherence[:, selected].sum(dim=1)
            scores = vector.abs() - penalty_weight * penalties
            if selected:
                scores[selected] = float("-inf")
            best_idx = int(torch.argmax(scores).item())
            best_score = float(scores[best_idx].item())
            if not math.isfinite(best_score) or best_score <= min_abs_affinity:
                break
            selected.append(best_idx)
        if not selected:
            fallback_idx = int(torch.argmax(vector.abs()).item())
            if float(vector.abs()[fallback_idx].item()) <= min_abs_affinity:
                return torch.zeros_like(vector)
            selected = [fallback_idx]
        selected_tensor = torch.tensor(selected, dtype=torch.long)
        restricted_gram = gram.index_select(0, selected_tensor).index_select(1, selected_tensor)
        ridge = float(getattr(self.hparams, "atom_sparse_ridge", 1.0e-4) or 1.0e-4)
        restricted_gram = restricted_gram + ridge * torch.eye(len(selected), dtype=restricted_gram.dtype)
        restricted_affinity = vector.index_select(0, selected_tensor)
        try:
            coeff = torch.linalg.solve(restricted_gram, restricted_affinity.unsqueeze(1)).squeeze(1)
        except RuntimeError:
            coeff = torch.linalg.pinv(restricted_gram) @ restricted_affinity
        code = torch.zeros_like(vector)
        code.index_copy_(0, selected_tensor, coeff)
        code[code.abs() <= min_abs_affinity] = 0.0
        return code

    def _hard_discrete_support_mask(
        self,
        affinity: torch.Tensor,
        basis: dict[str, Any],
        *,
        topk: int,
        usage_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vector = affinity.detach().float().cpu().flatten()
        if vector.numel() == 0:
            return vector
        gram = self._basis_gram_matrix(basis)
        if gram is None or gram.numel() == 0:
            return torch.sign(vector)
        limit = min(int(topk), int(vector.numel()))
        coherence = gram.abs()
        penalty_weight = float(getattr(self.hparams, "atom_coherence_penalty", 0.25) or 0.25)
        exclusivity_weight = float(getattr(self.hparams, "atom_support_exclusivity_penalty", 0.35) or 0.35)
        min_abs_affinity = float(getattr(self.hparams, "atom_sparse_min_abs_affinity", 1.0e-6) or 1.0e-6)
        if usage_counts is None:
            usage = torch.zeros_like(vector)
        else:
            usage = usage_counts.detach().float().cpu().flatten()
            if usage.numel() < vector.numel():
                padded = torch.zeros_like(vector)
                padded[: usage.numel()] = usage
                usage = padded
            else:
                usage = usage[: vector.numel()]
        selected: list[int] = []
        for _ in range(limit):
            penalties = torch.zeros_like(vector)
            if selected:
                penalties = coherence[:, selected].sum(dim=1)
            scores = vector.abs() - penalty_weight * penalties - exclusivity_weight * usage
            if selected:
                scores[selected] = float("-inf")
            best_idx = int(torch.argmax(scores).item())
            best_score = float(scores[best_idx].item())
            if not math.isfinite(best_score) or best_score <= min_abs_affinity:
                break
            selected.append(best_idx)
        if not selected:
            fallback_idx = int(torch.argmax(vector.abs()).item())
            if float(vector.abs()[fallback_idx].item()) <= min_abs_affinity:
                return torch.zeros_like(vector)
            selected = [fallback_idx]
        mask = torch.zeros_like(vector)
        for idx in selected:
            sign = float(vector[idx].item())
            mask[idx] = 1.0 if sign >= 0.0 else -1.0
        return mask

    def _code_support_indices(self, code: torch.Tensor | None) -> set[int]:
        if not isinstance(code, torch.Tensor):
            return set()
        vector = code.detach().float().cpu().flatten()
        return {
            int(idx)
            for idx, value in enumerate(vector.tolist())
            if abs(float(value)) > 1.0e-8
        }

    def _support_overlap(
        self,
        left_codes: dict[str, torch.Tensor],
        right_codes: dict[str, torch.Tensor],
    ) -> float | None:
        intersection = 0
        union = 0
        for stem in set(left_codes.keys()) | set(right_codes.keys()):
            left_support = self._code_support_indices(left_codes.get(stem))
            right_support = self._code_support_indices(right_codes.get(stem))
            if not left_support and not right_support:
                continue
            intersection += len(left_support & right_support)
            union += len(left_support | right_support)
        if union == 0:
            return None
        return float(intersection / union)

    def _active_atom_coherence(
        self,
        cell: dict[str, Any],
        codes_by_stem: dict[str, torch.Tensor],
    ) -> float | None:
        state_basis = cell.get("state_shared_basis") or {}
        coherence_terms = []
        for stem, code in codes_by_stem.items():
            basis = state_basis.get(stem)
            if not isinstance(basis, dict):
                continue
            gram = self._basis_gram_matrix(basis)
            if gram is None:
                continue
            support = sorted(self._code_support_indices(code))
            if not support:
                continue
            if len(support) == 1:
                coherence_terms.append(0.0)
                continue
            support_tensor = torch.tensor(support, dtype=torch.long)
            restricted = gram.abs().index_select(0, support_tensor).index_select(1, support_tensor)
            offdiag_mask = ~torch.eye(len(support), dtype=torch.bool)
            if not offdiag_mask.any():
                coherence_terms.append(0.0)
                continue
            coherence_terms.append(float(restricted[offdiag_mask].mean().item()))
        if not coherence_terms:
            return None
        return float(sum(coherence_terms) / len(coherence_terms))

    def _slot_factors_by_module(self, slot: dict[str, Any]) -> dict[str, dict[str, Any]]:
        weights = slot.get("slot_weights") or {}
        modules: dict[str, dict[str, Any]] = {}
        for canonical_name, tensor in weights.items():
            stem = self._canonical_module_stem(canonical_name)
            module = modules.setdefault(stem, {})
            if ".lora_A." in canonical_name:
                module["canonical_lora_A"] = canonical_name
                module["lora_A"] = tensor.detach().float().cpu()
            elif ".lora_B." in canonical_name:
                module["canonical_lora_B"] = canonical_name
                module["lora_B"] = tensor.detach().float().cpu()
        slot_scale = self._slot_train_scale(slot)
        factors: dict[str, dict[str, Any]] = {}
        for stem, module in modules.items():
            lora_a = module.get("lora_A")
            lora_b = module.get("lora_B")
            if not isinstance(lora_a, torch.Tensor) or not isinstance(lora_b, torch.Tensor):
                continue
            factors[stem] = {
                "canonical_lora_A": module.get("canonical_lora_A"),
                "canonical_lora_B": module.get("canonical_lora_B"),
                "lora_A": lora_a,
                "lora_B": lora_b,
                "scale": float(slot_scale),
            }
        return factors

    def _slot_delta_by_module(self, slot: dict[str, Any]) -> dict[str, dict[str, Any]]:
        modules = self._slot_factors_by_module(slot)
        deltas: dict[str, dict[str, Any]] = {}
        for stem, module in modules.items():
            lora_a = module.get("lora_A")
            lora_b = module.get("lora_B")
            if not isinstance(lora_a, torch.Tensor) or not isinstance(lora_b, torch.Tensor):
                continue
            deltas[stem] = {
                "canonical_lora_A": module.get("canonical_lora_A"),
                "canonical_lora_B": module.get("canonical_lora_B"),
                "delta": (lora_b @ lora_a) * float(module.get("scale") or 1.0),
            }
        return deltas

    def _build_shared_state_basis(self, slots: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if not slots:
            return {}
        per_module_atoms: dict[str, list[dict[str, Any]]] = {}
        for slot in slots:
            for stem, payload in self._slot_factors_by_module(slot).items():
                lora_a = payload.get("lora_A")
                lora_b = payload.get("lora_B")
                scale = float(payload.get("scale") or 1.0)
                if not isinstance(lora_a, torch.Tensor) or not isinstance(lora_b, torch.Tensor):
                    continue
                rank = min(int(lora_a.shape[0]), int(lora_b.shape[1]))
                for component_idx in range(rank):
                    left = lora_b[:, component_idx].detach().float().cpu()
                    right = lora_a[component_idx, :].detach().float().cpu()
                    left_norm = float(left.norm().item())
                    right_norm = float(right.norm().item())
                    if left_norm <= 1.0e-8 or right_norm <= 1.0e-8:
                        continue
                    per_module_atoms.setdefault(stem, []).append(
                        {
                            "canonical_lora_A": payload.get("canonical_lora_A"),
                            "canonical_lora_B": payload.get("canonical_lora_B"),
                            "left_atom": left / left_norm,
                            "right_atom": right / right_norm,
                            "atom_strength": float(scale * left_norm * right_norm),
                        }
                    )
        state_basis: dict[str, dict[str, Any]] = {}
        target_rank = self._shared_basis_rank()
        for stem, atoms in per_module_atoms.items():
            if not atoms:
                continue
            atoms.sort(key=lambda row: row["atom_strength"], reverse=True)
            chosen: list[dict[str, Any]] = []
            for atom in atoms:
                if len(chosen) >= target_rank:
                    break
                if not chosen:
                    chosen.append(atom)
                    continue
                redundant = False
                for existing in chosen:
                    left_sim = float(torch.dot(atom["left_atom"], existing["left_atom"]).item())
                    right_sim = float(torch.dot(atom["right_atom"], existing["right_atom"]).item())
                    if abs(left_sim * right_sim) >= 0.995:
                        redundant = True
                        break
                if not redundant:
                    chosen.append(atom)
            if not chosen:
                continue
            realized_rank = len(chosen)
            if realized_rank <= 0:
                continue
            state_basis[stem] = {
                "canonical_lora_A": chosen[0].get("canonical_lora_A"),
                "canonical_lora_B": chosen[0].get("canonical_lora_B"),
                "left_basis": torch.stack([atom["left_atom"] for atom in chosen], dim=1).detach().cpu(),
                "right_basis": torch.stack([atom["right_atom"] for atom in chosen], dim=0).detach().cpu(),
                "singular_values": torch.tensor([float(atom["atom_strength"]) for atom in chosen], dtype=torch.float32),
                "basis_rank": realized_rank,
            }
        return state_basis

    def _fit_slot_codes_to_state_basis(self, slot: dict[str, Any], state_basis: dict[str, dict[str, Any]]) -> dict[str, torch.Tensor]:
        slot_codes: dict[str, torch.Tensor] = {}
        if not state_basis:
            return slot_codes
        slot_factors = self._slot_factors_by_module(slot)
        for stem, basis in state_basis.items():
            payload = slot_factors.get(stem)
            if payload is None:
                continue
            affinity = self._slot_affinity_to_basis(payload, basis)
            if affinity is None:
                continue
            slot_codes[stem] = self._hard_sparse_pursuit_code(affinity.detach().cpu(), basis, topk=self._atom_write_topk())
        return slot_codes

    def _slot_affinity_to_basis(
        self,
        payload: dict[str, Any],
        basis: dict[str, Any],
    ) -> torch.Tensor | None:
        lora_a = payload.get("lora_A")
        lora_b = payload.get("lora_B")
        scale = float(payload.get("scale") or 1.0)
        left_basis = basis.get("left_basis")
        right_basis = basis.get("right_basis")
        if (
            not isinstance(lora_a, torch.Tensor)
            or not isinstance(lora_b, torch.Tensor)
            or not isinstance(left_basis, torch.Tensor)
            or not isinstance(right_basis, torch.Tensor)
        ):
            return None
        realized_rank = min(int(left_basis.shape[1]), int(right_basis.shape[0]))
        if realized_rank <= 0:
            return None
        code = torch.zeros(realized_rank, dtype=torch.float32)
        slot_rank = min(int(lora_a.shape[0]), int(lora_b.shape[1]))
        for component_idx in range(slot_rank):
            left_component = lora_b[:, component_idx].detach().float().cpu()
            right_component = lora_a[component_idx, :].detach().float().cpu()
            left_proj = left_basis[:, :realized_rank].transpose(0, 1) @ left_component
            right_proj = right_basis[:realized_rank, :] @ right_component
            code += scale * left_proj * right_proj
        return code

    def _fit_slot_latent_supports(
        self,
        slots: list[dict[str, Any]],
        state_basis: dict[str, dict[str, Any]],
    ) -> None:
        if not state_basis:
            return
        usage_by_stem: dict[str, torch.Tensor] = {}
        for stem, basis in state_basis.items():
            basis_rank = int(basis.get("basis_rank") or 0)
            usage_by_stem[stem] = torch.zeros(basis_rank, dtype=torch.float32)
        for slot in slots:
            if self._use_discrete_support_masks():
                slot_codes: dict[str, torch.Tensor] = {}
                slot_factors = self._slot_factors_by_module(slot)
                for stem, basis in state_basis.items():
                    payload = slot_factors.get(stem)
                    if payload is None:
                        continue
                    affinity = self._slot_affinity_to_basis(payload, basis)
                    if affinity is None:
                        continue
                    slot_codes[stem] = self._hard_discrete_support_mask(
                        affinity=affinity,
                        basis=basis,
                        topk=self._atom_write_topk(),
                        usage_counts=usage_by_stem.get(stem),
                    )
            else:
                slot_codes = self._fit_slot_codes_to_state_basis(slot, state_basis)
            slot["slot_codes"] = slot_codes
            for stem, code in slot_codes.items():
                if stem not in usage_by_stem or not isinstance(code, torch.Tensor):
                    continue
                usage_by_stem[stem][: min(int(code.numel()), int(usage_by_stem[stem].numel()))] += (
                    code.detach().float().cpu()[: usage_by_stem[stem].numel()].abs()
                )

    def _collect_raw_activation_keys(self, entries: list[dict[str, Any]] | None = None) -> list[torch.Tensor]:
        source_entries = self.memory_entries if entries is None else entries
        raw_keys = []
        for entry in source_entries:
            for view in self._entry_view_records(entry):
                raw_key = view.get("raw_activation_key")
                if isinstance(raw_key, torch.Tensor):
                    raw_keys.append(raw_key)
        return raw_keys

    def _summarize_slot_prototypes(self, view_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float | None]:
        if not view_records:
            return [], None
        grouped: dict[str, list[dict[str, Any]]] = {}
        for view in view_records:
            grouped.setdefault(view.get("view_name") or "unknown", []).append(view)
        prototypes: list[dict[str, Any]] = []
        dispersions: list[float] = []
        for view_name, views in grouped.items():
            semantic_stack = torch.stack([view["semantic_key"].detach().float().cpu() for view in views], dim=0)
            activation_stack = torch.stack([view["activation_key"].detach().float().cpu() for view in views], dim=0)
            semantic_centroid = self._normalize_semantic_key(semantic_stack.mean(dim=0))
            activation_centroid = self._normalize_activation_key(activation_stack.mean(dim=0), self.cached_activation_stats)
            distances = []
            for view in views:
                semantic_score = float(
                    F.cosine_similarity(view["semantic_key"].detach().float().cpu().unsqueeze(0), semantic_centroid.unsqueeze(0), dim=-1)[0].item()
                )
                activation_score = float(
                    F.cosine_similarity(view["activation_key"].detach().float().cpu().unsqueeze(0), activation_centroid.unsqueeze(0), dim=-1)[0].item()
                )
                distances.append(1.0 - (self.hparams.semantic_weight * semantic_score + self.hparams.activation_weight * activation_score))
            prototype_dispersion = 0.0 if not distances else float(sum(distances) / len(distances))
            dispersions.append(prototype_dispersion)
            prototypes.append(
                {
                    "view_name": view_name,
                    "text": views[0].get("text"),
                    "semantic_key": semantic_centroid,
                    "activation_key": activation_centroid,
                    "prototype_dispersion": prototype_dispersion,
                }
            )
        return prototypes, (None if not dispersions else float(sum(dispersions) / len(dispersions)))

    def _score_slot(
        self,
        slot: dict[str, Any],
        semantic_key: torch.Tensor,
        activation_key: torch.Tensor,
    ) -> dict[str, Any] | None:
        best = None
        for prototype in self._slot_view_records(slot):
            semantic_view = prototype.get("semantic_key")
            activation_view = prototype.get("activation_key")
            if not isinstance(semantic_view, torch.Tensor) or not isinstance(activation_view, torch.Tensor):
                continue
            semantic_score = F.cosine_similarity(semantic_view.unsqueeze(0), semantic_key.unsqueeze(0), dim=-1)[0]
            activation_score = F.cosine_similarity(activation_view.unsqueeze(0), activation_key.unsqueeze(0), dim=-1)[0]
            combined = self.hparams.semantic_weight * semantic_score + self.hparams.activation_weight * activation_score
            candidate = {
                "slot": slot,
                "combined_score": float(combined.item()),
                "semantic_score": float(semantic_score.item()),
                "activation_score": float(activation_score.item()),
                "best_view_name": prototype.get("view_name"),
                "best_view_text": prototype.get("text"),
                "slot_dispersion": prototype.get("prototype_dispersion", slot.get("slot_dispersion")),
            }
            if best is None or candidate["combined_score"] > best["combined_score"]:
                best = candidate
        return best

    def _normalize_slot_selection(self, selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not selected_rows:
            return []
        values = torch.tensor([max(0.0, float(row["combined_score"])) for row in selected_rows], dtype=torch.float32)
        if torch.allclose(values, torch.zeros_like(values)):
            weights = torch.ones_like(values) / float(len(selected_rows))
        else:
            weights = F.softmax(values * self.hparams.beta, dim=0)
        min_weight = float(getattr(self.hparams, "slot_activation_min_weight", 0.15) or 0.15)
        normalized = []
        for row, weight in zip(selected_rows, weights.tolist()):
            row_copy = dict(row)
            row_copy["slot_weight"] = float(weight)
            normalized.append(row_copy)
        kept = [row for row in normalized if row["slot_weight"] >= min_weight]
        if not kept:
            kept = [max(normalized, key=lambda row: row["slot_weight"])]
        weight_sum = sum(row["slot_weight"] for row in kept)
        for row in kept:
            row["slot_weight"] = float(row["slot_weight"] / max(weight_sum, 1.0e-8))
        return kept

    def _aggregate_selected_slot_codes(
        self,
        cell: dict[str, Any],
        selected_rows: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        state_basis = cell.get("state_shared_basis") or {}
        aggregated: dict[str, torch.Tensor] = {}
        for stem, basis in state_basis.items():
            basis_rank = int(basis.get("basis_rank") or 0)
            if basis_rank <= 0:
                continue
            code = torch.zeros(basis_rank, dtype=torch.float32)
            for row in selected_rows[: self._slot_top_k()]:
                slot_codes = row["slot"].get("slot_codes") or {}
                slot_code = slot_codes.get(stem)
                if not isinstance(slot_code, torch.Tensor):
                    continue
                slot_weight = float(row.get("slot_weight") or 0.0)
                code[: min(basis_rank, int(slot_code.numel()))] += slot_weight * slot_code.detach().float().cpu()[:basis_rank]
            if self._use_discrete_support_masks():
                aggregated[stem] = self._hard_discrete_support_mask(code, basis, topk=self._atom_sparse_topk())
            else:
                aggregated[stem] = self._hard_sparse_pursuit_code(code, basis, topk=self._atom_sparse_topk())
        return aggregated

    def _factor_space_code_stats(self, codes_by_stem: dict[str, torch.Tensor]) -> dict[str, float | int | None]:
        l1_terms = []
        l2_terms = []
        support = 0
        total = 0
        for code in codes_by_stem.values():
            if not isinstance(code, torch.Tensor):
                continue
            vector = code.detach().float().cpu()
            l1_terms.append(float(vector.abs().sum().item()))
            l2_terms.append(float(vector.norm().item()))
            support += int((vector.abs() > 1.0e-8).sum().item())
            total += int(vector.numel())
        sparsity = None if total == 0 else float(support / max(1, total))
        return {
            "code_l1_mean": None if not l1_terms else float(sum(l1_terms) / len(l1_terms)),
            "code_l2_mean": None if not l2_terms else float(sum(l2_terms) / len(l2_terms)),
            "code_support": support,
            "code_total_dim": total,
            "code_nonzero_fraction": sparsity,
        }

    def _realization_coefficients(
        self,
        basis: dict[str, Any] | None,
        code: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not isinstance(code, torch.Tensor):
            return None
        vector = code.detach().float().cpu().flatten()
        if not self._use_discrete_support_masks() or not isinstance(basis, dict):
            return vector
        singular_values = basis.get("singular_values")
        if not isinstance(singular_values, torch.Tensor):
            return vector
        realized_rank = min(int(vector.numel()), int(singular_values.numel()))
        coeff = torch.zeros_like(vector)
        coeff[:realized_rank] = vector[:realized_rank] * singular_values[:realized_rank].detach().float().cpu()
        return coeff

    def _state_self_overlap(self, cell: dict[str, Any], codes_by_stem: dict[str, torch.Tensor]) -> float:
        state_basis = cell.get("state_shared_basis") or {}
        total = 0.0
        for stem, code in codes_by_stem.items():
            if stem not in state_basis or not isinstance(code, torch.Tensor):
                continue
            vector = self._realization_coefficients(state_basis.get(stem), code)
            if vector is None:
                continue
            total += float(torch.dot(vector, vector).item())
        return float(max(0.0, total))

    def _cross_state_realization_overlap(
        self,
        left_cell: dict[str, Any],
        left_codes: dict[str, torch.Tensor],
        right_cell: dict[str, Any],
        right_codes: dict[str, torch.Tensor],
    ) -> float | None:
        left_basis_map = left_cell.get("state_shared_basis") or {}
        right_basis_map = right_cell.get("state_shared_basis") or {}
        overlap = 0.0
        compared = 0
        for stem, left_code in left_codes.items():
            right_code = right_codes.get(stem)
            left_basis = left_basis_map.get(stem)
            right_basis = right_basis_map.get(stem)
            if (
                not isinstance(left_code, torch.Tensor)
                or not isinstance(right_code, torch.Tensor)
                or not isinstance(left_basis, dict)
                or not isinstance(right_basis, dict)
            ):
                continue
            left_u = left_basis.get("left_basis")
            left_v = left_basis.get("right_basis")
            right_u = right_basis.get("left_basis")
            right_v = right_basis.get("right_basis")
            if (
                not isinstance(left_u, torch.Tensor)
                or not isinstance(left_v, torch.Tensor)
                or not isinstance(right_u, torch.Tensor)
                or not isinstance(right_v, torch.Tensor)
            ):
                continue
            left_coeff = self._realization_coefficients(left_basis, left_code)
            right_coeff = self._realization_coefficients(right_basis, right_code)
            if left_coeff is None or right_coeff is None:
                continue
            left_vec = left_coeff[: int(left_u.shape[1])]
            right_vec = right_coeff[: int(right_u.shape[1])]
            if left_vec.numel() == 0 or right_vec.numel() == 0:
                continue
            left_overlap = left_u[:, : left_vec.numel()].transpose(0, 1) @ right_u[:, : right_vec.numel()]
            right_overlap = left_v[: left_vec.numel(), :] @ right_v[: right_vec.numel(), :].transpose(0, 1)
            cross_matrix = left_overlap * right_overlap
            overlap += float((left_vec.unsqueeze(0) @ cross_matrix @ right_vec.unsqueeze(1)).item())
            compared += 1
        if compared == 0:
            return None
        left_norm = math.sqrt(max(1.0e-8, self._state_self_overlap(left_cell, left_codes)))
        right_norm = math.sqrt(max(1.0e-8, self._state_self_overlap(right_cell, right_codes)))
        return float(abs(overlap) / max(1.0e-8, left_norm * right_norm))

    def _slot_factor_space_residual(
        self,
        slot: dict[str, Any],
        state_basis: dict[str, dict[str, Any]],
    ) -> float | None:
        slot_codes = slot.get("slot_codes") or {}
        slot_factors = self._slot_factors_by_module(slot)
        residual_terms = []
        for stem, code in slot_codes.items():
            basis = state_basis.get(stem)
            payload = slot_factors.get(stem)
            if not isinstance(code, torch.Tensor) or not isinstance(basis, dict) or not isinstance(payload, dict):
                continue
            lora_a = payload.get("lora_A")
            lora_b = payload.get("lora_B")
            scale = float(payload.get("scale") or 1.0)
            left_basis = basis.get("left_basis")
            right_basis = basis.get("right_basis")
            if (
                not isinstance(lora_a, torch.Tensor)
                or not isinstance(lora_b, torch.Tensor)
                or not isinstance(left_basis, torch.Tensor)
                or not isinstance(right_basis, torch.Tensor)
            ):
                continue
            realized_rank = min(int(code.numel()), int(left_basis.shape[1]), int(right_basis.shape[0]))
            if realized_rank <= 0:
                continue
            left_basis_cpu = left_basis[:, :realized_rank].detach().float().cpu()
            right_basis_cpu = right_basis[:realized_rank, :].detach().float().cpu()
            coeff = self._realization_coefficients(basis, code)
            if coeff is None:
                continue
            code_cpu = coeff[:realized_rank]
            slot_rank = min(int(lora_a.shape[0]), int(lora_b.shape[1]))
            lora_a_cpu = lora_a[:slot_rank, :].detach().float().cpu()
            lora_b_cpu = lora_b[:, :slot_rank].detach().float().cpu()

            delta_left_gram = lora_b_cpu.transpose(0, 1) @ lora_b_cpu
            delta_right_gram = lora_a_cpu @ lora_a_cpu.transpose(0, 1)
            delta_norm_sq = (scale ** 2) * float((delta_left_gram * delta_right_gram).sum().item())
            if delta_norm_sq <= 1.0e-8:
                continue
            basis_left_gram = left_basis_cpu.transpose(0, 1) @ left_basis_cpu
            basis_right_gram = right_basis_cpu @ right_basis_cpu.transpose(0, 1)
            code_outer = code_cpu.unsqueeze(1) * code_cpu.unsqueeze(0)
            recon_norm_sq = float((code_outer * basis_left_gram * basis_right_gram).sum().item())
            cross_left = lora_b_cpu.transpose(0, 1) @ left_basis_cpu
            cross_right = lora_a_cpu @ right_basis_cpu.transpose(0, 1)
            cross_term = scale * float(((cross_left * cross_right) * code_cpu.unsqueeze(0)).sum().item())
            residual_sq = max(0.0, delta_norm_sq + recon_norm_sq - 2.0 * cross_term)
            residual = float(math.sqrt(residual_sq / max(delta_norm_sq, 1.0e-8)))
            residual_terms.append(residual)
        return None if not residual_terms else float(sum(residual_terms) / len(residual_terms))

    def _active_shards(self) -> list[dict[str, Any]]:
        if not self._use_side_memory_shards():
            return []
        return [self.shard_registry[shard_id] for shard_id in sorted(self.shard_registry.keys())]

    def _shard_atoms(self, shard: dict[str, Any]) -> list[dict[str, Any]]:
        return list(shard.get("atoms") or [])

    def _score_prototype_records(
        self,
        prototypes: list[dict[str, Any]],
        semantic_key: torch.Tensor,
        activation_key: torch.Tensor,
    ) -> dict[str, Any] | None:
        best = None
        for prototype in prototypes:
            semantic_view = prototype.get("semantic_key")
            activation_view = prototype.get("activation_key")
            if not isinstance(semantic_view, torch.Tensor) or not isinstance(activation_view, torch.Tensor):
                continue
            semantic_score = F.cosine_similarity(semantic_view.unsqueeze(0), semantic_key.unsqueeze(0), dim=-1)[0]
            activation_score = F.cosine_similarity(activation_view.unsqueeze(0), activation_key.unsqueeze(0), dim=-1)[0]
            combined = self.hparams.semantic_weight * semantic_score + self.hparams.activation_weight * activation_score
            candidate = {
                "combined_score": float(combined.item()),
                "semantic_score": float(semantic_score.item()),
                "activation_score": float(activation_score.item()),
                "best_view_name": prototype.get("view_name"),
                "best_view_text": prototype.get("text"),
                "prototype_dispersion": prototype.get("prototype_dispersion"),
            }
            if best is None or candidate["combined_score"] > best["combined_score"]:
                best = candidate
        return best

    def _score_shard(self, shard: dict[str, Any], semantic_key: torch.Tensor, activation_key: torch.Tensor) -> dict[str, Any] | None:
        prototypes = shard.get("shard_prototypes") or []
        scored = self._score_prototype_records(prototypes, semantic_key, activation_key)
        if scored is None:
            return None
        scored["shard"] = shard
        return scored

    def _score_atom(self, atom: dict[str, Any], semantic_key: torch.Tensor, activation_key: torch.Tensor) -> dict[str, Any] | None:
        prototypes = atom.get("atom_prototypes") or []
        scored = self._score_prototype_records(prototypes, semantic_key, activation_key)
        if scored is None:
            return None
        scored["atom"] = atom
        return scored

    def _rank_shards_from_keys(self, semantic_key: torch.Tensor, activation_key: torch.Tensor) -> list[dict[str, Any]]:
        if not self._use_side_memory_shards():
            return []
        semantic_key = self._normalize_semantic_key(semantic_key)
        activation_stats = self.cached_activation_stats
        if activation_stats is None:
            activation_stats = self._activation_stats_from_raw_keys(self._collect_raw_activation_keys())
        activation_key = self._normalize_activation_key(activation_key, activation_stats)
        ranking = []
        for shard in self._active_shards():
            scored = self._score_shard(shard, semantic_key, activation_key)
            if scored is not None:
                ranking.append(scored)
        ranking.sort(key=lambda row: row["combined_score"], reverse=True)
        return ranking

    def _rank_atoms_in_shard(
        self,
        shard: dict[str, Any],
        semantic_key: torch.Tensor,
        activation_key: torch.Tensor,
        *,
        apply_usage_penalty: bool,
    ) -> list[dict[str, Any]]:
        atom_rows = []
        usage_penalty = float(getattr(self.hparams, "support_usage_penalty", 0.35) or 0.35)
        for atom in self._shard_atoms(shard):
            scored = self._score_atom(atom, semantic_key, activation_key)
            if scored is None:
                continue
            if apply_usage_penalty:
                scored["combined_score"] -= usage_penalty * float(atom.get("usage_count") or 0.0)
            atom_rows.append(scored)
        atom_rows.sort(key=lambda row: row["combined_score"], reverse=True)
        return atom_rows

    def _normalize_atom_selection(self, atom_rows: list[dict[str, Any]], *, topk: int) -> list[dict[str, Any]]:
        if not atom_rows:
            return []
        top_rows = atom_rows[: min(topk, len(atom_rows))]
        values = torch.tensor([float(row["combined_score"]) for row in top_rows], dtype=torch.float32)
        if torch.allclose(values, torch.zeros_like(values)):
            weights = torch.ones_like(values) / float(len(top_rows))
        else:
            weights = F.softmax(values * self.hparams.beta, dim=0)
        normalized = []
        for row, weight in zip(top_rows, weights.tolist()):
            row_copy = dict(row)
            row_copy["atom_weight"] = float(weight)
            normalized.append(row_copy)
        return normalized

    def _make_grouped_weights(self, weights: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        grouped: dict[str, dict[str, torch.Tensor]] = {}
        for canonical_name, tensor in weights.items():
            stem = self._canonical_module_stem(canonical_name)
            group = self._side_memory_group_name(stem)
            grouped.setdefault(group, {})[canonical_name] = tensor.detach().float().cpu().clone()
        return grouped

    def _new_atom_weights_from_runtime_adapter(self, adapter_name: str) -> dict[str, torch.Tensor]:
        weights = self._capture_adapter_parameters(adapter_name)
        grouped = self._make_grouped_weights(weights)
        if not grouped:
            return {}
        priority = ["attention", "mlp"]
        for group_name in priority:
            if group_name in grouped and grouped[group_name]:
                return grouped[group_name]
        first_group = next(iter(grouped.keys()))
        return grouped[first_group]

    def _initialize_new_side_memory_atom(self, atom_id: str, adapter_name: str, view_key_records: list[dict[str, Any]]) -> dict[str, Any]:
        atom_weights = self._new_atom_weights_from_runtime_adapter(adapter_name)
        prototypes, dispersion = self._summarize_slot_prototypes(view_key_records)
        group_name = "attention"
        if atom_weights:
            first_key = next(iter(atom_weights.keys()))
            group_name = self._side_memory_group_name(self._canonical_module_stem(first_key))
        return {
            "atom_id": atom_id,
            "group_name": group_name,
            "atom_rank": self._side_memory_atom_rank(),
            "atom_weights": atom_weights,
            "view_keys": self._clone_to_cpu(view_key_records),
            "atom_prototypes": self._clone_to_cpu(prototypes),
            "atom_dispersion": dispersion,
            "usage_count": 0,
            "member_edit_ids": [],
        }

    def _create_new_shard(self, *, view_key_records: list[dict[str, Any]]) -> dict[str, Any]:
        shard_id = self._next_shard_id()
        prototypes, dispersion = self._summarize_slot_prototypes(view_key_records)
        shard = {
            "shard_id": shard_id,
            "member_edit_ids": [],
            "atoms": [],
            "shard_view_keys": [],
            "shard_prototypes": self._clone_to_cpu(prototypes),
            "prototype_dispersion": dispersion,
            "created_at_edit_index": len(self.memory_entries),
            "support_exclusivity_failures": 0,
            "base_only_fallbacks": 0,
            "prototype_margin_history": [],
        }
        self.shard_registry[shard_id] = shard
        return shard

    def _refresh_single_shard_metadata(self, shard_id: str) -> None:
        shard = self.shard_registry.get(shard_id)
        if shard is None:
            return
        aggregated_views = []
        for atom in self._shard_atoms(shard):
            for view in atom.get("view_keys") or []:
                aggregated_views.append(self._clone_to_cpu(view))
        shard["shard_view_keys"] = aggregated_views
        prototypes, dispersion = self._summarize_slot_prototypes(aggregated_views)
        shard["shard_prototypes"] = self._clone_to_cpu(prototypes)
        shard["prototype_dispersion"] = dispersion
        shard["occupancy"] = len(self._shard_atoms(shard))

    def _support_overlap_with_entries(self, shard_id: str, atom_ids: list[str]) -> float:
        target = set(atom_ids)
        if not target:
            return 0.0
        overlaps = []
        for entry in self.memory_entries:
            if entry.get("shard_id") != shard_id:
                continue
            existing = set(entry.get("support_atom_ids") or [])
            if not existing:
                continue
            overlaps.append(float(len(target & existing) / max(1, len(target | existing))))
        return 0.0 if not overlaps else float(sum(overlaps) / len(overlaps))

    def _load_side_memory_runtime_adapter(self, selected_atoms: list[dict[str, Any]]) -> str | None:
        if not selected_atoms:
            return None
        adapter_name = self._ensure_runtime_slot_adapter()
        self._zero_adapter_weights(adapter_name)
        refs = self._adapter_parameter_refs(adapter_name)
        atom_rank = self._side_memory_atom_rank()
        with torch.no_grad():
            for atom_index, row in enumerate(selected_atoms[: self._side_memory_read_topk() + max(0, int(getattr(self.hparams, "max_new_atoms_per_edit", 1) or 1))]):
                atom = row["atom"] if "atom" in row else row
                atom_weight = float(row.get("atom_weight", 1.0))
                start = atom_index * atom_rank
                end = start + atom_rank
                coeff = math.sqrt(max(0.0, atom_weight))
                for canonical_name, runtime_param in refs.items():
                    atom_tensor = (atom.get("atom_weights") or {}).get(canonical_name)
                    if atom_tensor is None:
                        continue
                    target_tensor = atom_tensor.to(runtime_param.device, dtype=runtime_param.dtype)
                    if "lora_A" in canonical_name:
                        runtime_param[start:end, :].copy_(target_tensor * coeff)
                    elif "lora_B" in canonical_name:
                        runtime_param[:, start:end].copy_(target_tensor * coeff)
        self.runtime_composed_signature = tuple(str((row["atom"] if "atom" in row else row).get("atom_id")) for row in selected_atoms)
        return adapter_name

    def _evaluate_sequence_loss(self, prompts: list[str], target: str, *, adapter_name: str | None = None) -> float:
        losses = []
        if adapter_name is not None:
            self._set_runtime_adapter(adapter_name)
            context_manager = nullcontext()
        else:
            context_manager = self._adapter_disabled()
        with context_manager:
            for prompt_text in prompts:
                batch = self._training_batch(prompt_text, target)
                losses.append(float(self.model(**batch).loss.detach().item()))
        return float(sum(losses) / max(1, len(losses)))

    def _side_memory_support_stats(self, selected_atoms: list[dict[str, Any]]) -> dict[str, float | None]:
        if not selected_atoms:
            return {
                "support_size": 0.0,
                "support_overlap": None,
                "atom_coherence": None,
                "realization_overlap": None,
                "code_l1_mean": 0.0,
                "code_l2_mean": 0.0,
                "code_nonzero_fraction": 0.0,
            }
        atom_ids = [str((row["atom"] if "atom" in row else row).get("atom_id")) for row in selected_atoms]
        weights = [float(row.get("atom_weight", 1.0)) for row in selected_atoms]
        overlap = None
        if len(selected_atoms) > 1:
            overlap = float(sum(weights) - max(weights))
        l1 = float(sum(abs(weight) for weight in weights))
        l2 = float(math.sqrt(sum(weight * weight for weight in weights)))
        return {
            "support_size": float(len(atom_ids)),
            "support_overlap": overlap,
            "atom_coherence": overlap,
            "realization_overlap": overlap,
            "code_l1_mean": l1,
            "code_l2_mean": l2,
            "code_nonzero_fraction": 0.0 if not atom_ids else float(len(atom_ids) / max(1, self._side_memory_read_topk() * max(1, len({(row['atom'] if 'atom' in row else row).get('group_name') for row in selected_atoms})))),
        }

    def _train_new_side_memory_atom(
        self,
        request: dict[str, Any],
        prompt: str,
        subject_prompt: str,
        negative_entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        writer_adapter = f"__hopedit_v3_writer_{self.edit_index:05d}__"
        atom_rank = self._side_memory_atom_rank()
        self._ensure_adapter(writer_adapter, rank=atom_rank, alpha=float(atom_rank))
        trainable = self._configure_trainable_adapter(writer_adapter)
        optimizer = torch.optim.Adam(trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        positive_views = self._build_training_views(request, prompt, subject_prompt)
        target = request["target_new"]
        self.model.train()
        for _ in range(max(1, int(getattr(self.hparams, "side_memory_write_steps", 2) or 2))):
            optimizer.zero_grad()
            positive_losses = []
            for view in positive_views:
                batch = self._training_batch(view, target)
                positive_losses.append(self.model(**batch).loss)
            positive_loss = torch.stack(positive_losses).mean()
            negative_loss = self._contrastive_negative_loss(target, positive_loss, negative_entries)
            loss = positive_loss if negative_loss is None else positive_loss + self.hparams.negative_weight * negative_loss
            loss.backward()
            optimizer.step()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        return {
            "adapter_name": writer_adapter,
            "write_loss": self._evaluate_sequence_loss(positive_views, target, adapter_name=writer_adapter),
        }

    def _runtime_adapter_target_rank(self) -> int:
        if self._use_side_memory_shards():
            max_atoms = max(
                self._side_memory_read_topk(),
                self._side_memory_write_topk() + max(0, int(getattr(self.hparams, "max_new_atoms_per_edit", 1) or 1)),
            )
            return self._side_memory_atom_rank() * max_atoms
        if self._use_shared_basis_codes():
            return self._shared_basis_rank()
        return self._slot_rank() * self._slot_top_k()

    def _ensure_runtime_slot_adapter(self) -> str:
        adapter_name = self._runtime_slot_adapter_name()
        self._ensure_adapter(adapter_name, rank=self._runtime_adapter_target_rank(), alpha=float(self._runtime_adapter_target_rank()))
        return adapter_name

    def _zero_adapter_weights(self, adapter_name: str) -> None:
        refs = self._adapter_parameter_refs(adapter_name)
        with torch.no_grad():
            for parameter in refs.values():
                parameter.zero_()

    def _load_composed_slot_adapter(self, selected_rows: list[dict[str, Any]]) -> str | None:
        if not selected_rows:
            return None
        if self._use_shared_basis_codes():
            return self._load_basis_coded_runtime_adapter(selected_rows)
        adapter_name = self._ensure_runtime_slot_adapter()
        self._zero_adapter_weights(adapter_name)
        refs = self._adapter_parameter_refs(adapter_name)
        slot_rank = self._slot_rank()
        slot_scale = float(self.hparams.lora_alpha) / float(slot_rank)
        with torch.no_grad():
            for slot_index, row in enumerate(selected_rows[: self._slot_top_k()]):
                slot = row["slot"]
                slot_weight = float(row.get("slot_weight", 0.0))
                coeff = math.sqrt(max(0.0, slot_weight * slot_scale))
                slot_weights = slot.get("slot_weights") or {}
                start = slot_index * slot_rank
                end = start + slot_rank
                for canonical_name, runtime_param in refs.items():
                    slot_tensor = slot_weights.get(canonical_name)
                    if slot_tensor is None:
                        continue
                    target_tensor = slot_tensor.to(runtime_param.device, dtype=runtime_param.dtype)
                    if "lora_A" in canonical_name:
                        runtime_param[start:end, :].copy_(target_tensor * coeff)
                    elif "lora_B" in canonical_name:
                        runtime_param[:, start:end].copy_(target_tensor * coeff)
                    else:
                        runtime_param.copy_(target_tensor)
        self.runtime_composed_signature = tuple(str(row["slot"]["slot_id"]) for row in selected_rows[: self._slot_top_k()])
        return adapter_name

    def _load_basis_coded_runtime_adapter(self, selected_rows: list[dict[str, Any]]) -> str | None:
        if not selected_rows:
            return None
        chosen_cell = selected_rows[0].get("cell")
        if not isinstance(chosen_cell, dict):
            return None
        state_basis = chosen_cell.get("state_shared_basis") or {}
        if not state_basis:
            return None
        adapter_name = self._ensure_runtime_slot_adapter()
        self._zero_adapter_weights(adapter_name)
        refs = self._adapter_parameter_refs(adapter_name)
        basis_rank = self._runtime_adapter_target_rank()
        aggregate_codes = self._aggregate_selected_slot_codes(chosen_cell, selected_rows)
        with torch.no_grad():
            for stem, basis in state_basis.items():
                left_basis = basis.get("left_basis")
                right_basis = basis.get("right_basis")
                singular_values = basis.get("singular_values")
                canonical_lora_a = basis.get("canonical_lora_A")
                canonical_lora_b = basis.get("canonical_lora_B")
                if (
                    not isinstance(left_basis, torch.Tensor)
                    or not isinstance(right_basis, torch.Tensor)
                    or not isinstance(singular_values, torch.Tensor)
                    or canonical_lora_a not in refs
                    or canonical_lora_b not in refs
                ):
                    continue
                lora_a = refs[canonical_lora_a]
                lora_b = refs[canonical_lora_b]
                realized_rank = min(
                    basis_rank,
                    int(left_basis.shape[1]),
                    int(right_basis.shape[0]),
                    int(singular_values.numel()),
                    int(lora_a.shape[0]),
                    int(lora_b.shape[1]),
                )
                if realized_rank <= 0:
                    continue
                code = aggregate_codes.get(stem)
                if not isinstance(code, torch.Tensor):
                    continue
                code = code.detach().float().cpu()[:realized_rank]
                if torch.allclose(code, torch.zeros_like(code)):
                    continue
                if self._use_discrete_support_masks():
                    strengths = singular_values[:realized_rank].detach().float().cpu()
                    amplitudes = (strengths * code.abs()).clamp_min(0.0).sqrt()
                    signed_amplitudes = amplitudes * code.sign()
                else:
                    amplitudes = code.abs().sqrt()
                    signed_amplitudes = amplitudes * code.sign()
                left = left_basis[:, :realized_rank].to(lora_b.device, dtype=lora_b.dtype)
                right = right_basis[:realized_rank, :].to(lora_a.device, dtype=lora_a.dtype)
                lora_b[:, :realized_rank].copy_(left * signed_amplitudes.to(lora_b.device, dtype=lora_b.dtype).unsqueeze(0))
                lora_a[:realized_rank, :].copy_(amplitudes.to(lora_a.device, dtype=lora_a.dtype).unsqueeze(1) * right)
        self.runtime_composed_signature = tuple(str(row["slot"]["slot_id"]) for row in selected_rows[: self._slot_top_k()])
        return adapter_name

    def _state_false_activation_rate(self, cell_id: str) -> float:
        if not self.route_logs:
            return 0.0
        matches = [
            1.0
            for row in self.route_logs
            if row.get("route_event") == "inference"
            and row.get("chosen_memory_id") == cell_id
            and row.get("route_match") is False
        ]
        total = [
            1.0
            for row in self.route_logs
            if row.get("route_event") == "inference" and row.get("chosen_memory_id") == cell_id
        ]
        if not total:
            return 0.0
        return float(sum(matches) / len(total))

    def _state_locality_risk(
        self,
        chosen_row: dict[str, Any],
        runner_up_row: dict[str, Any] | None = None,
    ) -> float:
        ambiguity = 0.0
        competitor_overlap = 0.0
        if runner_up_row is not None:
            ambiguity = max(0.0, 1.0 - max(0.0, float(chosen_row["combined_score"] - runner_up_row["combined_score"])))
            chosen_slot = chosen_row.get("best_slot_row", {}).get("slot")
            runner_slot = runner_up_row.get("best_slot_row", {}).get("slot")
            if chosen_slot is not None and runner_slot is not None:
                chosen_proto = next(iter(self._slot_view_records(chosen_slot)), None)
                runner_proto = next(iter(self._slot_view_records(runner_slot)), None)
                if chosen_proto is not None and runner_proto is not None:
                    competitor_overlap = max(
                        0.0,
                        float(
                            (
                                self.hparams.semantic_weight
                                * F.cosine_similarity(
                                    chosen_proto["semantic_key"].unsqueeze(0),
                                    runner_proto["semantic_key"].unsqueeze(0),
                                    dim=-1,
                                )[0].item()
                                + self.hparams.activation_weight
                                * F.cosine_similarity(
                                    chosen_proto["activation_key"].unsqueeze(0),
                                    runner_proto["activation_key"].unsqueeze(0),
                                    dim=-1,
                                )[0].item()
                            )
                        ),
                    )
        slot_dispersion = float(chosen_row.get("slot_dispersion") or 0.0)
        within_state_conflict = float(chosen_row["cell"].get("within_cell_conflict_mean") or 0.0)
        false_activation_history = self._state_false_activation_rate(chosen_row["cell"]["cell_id"])
        risk = (
            0.30 * competitor_overlap
            + 0.20 * within_state_conflict
            + 0.20 * slot_dispersion
            + 0.15 * false_activation_history
            + 0.15 * ambiguity
        )
        return float(max(0.0, min(1.0, risk)))

    def _refresh_processed_memory_keys(self) -> None:
        activation_stats = self._activation_stats_from_raw_keys(self._collect_raw_activation_keys())
        if self._use_overlap_aware_anchor_trace_bank() or self._use_factored_address_trace_bank():
            activation_stats = None
        self.cached_activation_stats = activation_stats
        for entry in self.memory_entries:
            if isinstance(entry.get("raw_semantic_key"), torch.Tensor):
                entry["semantic_key"] = self._normalize_semantic_key(entry["raw_semantic_key"])
            if isinstance(entry.get("raw_activation_key"), torch.Tensor):
                entry["activation_key"] = self._normalize_activation_key(entry["raw_activation_key"], activation_stats)
            for view in self._entry_view_records(entry):
                if isinstance(view.get("raw_semantic_key"), torch.Tensor):
                    view["semantic_key"] = self._normalize_semantic_key(view["raw_semantic_key"])
                if isinstance(view.get("raw_activation_key"), torch.Tensor):
                    view["activation_key"] = self._normalize_activation_key(view["raw_activation_key"], activation_stats)
        if self._use_sparse_slots():
            for cell in self.cell_registry.values():
                for slot in self._state_slots(cell):
                    for view in slot.get("view_keys") or []:
                        if isinstance(view.get("raw_semantic_key"), torch.Tensor):
                            view["semantic_key"] = self._normalize_semantic_key(view["raw_semantic_key"])
                        if isinstance(view.get("raw_activation_key"), torch.Tensor):
                            view["activation_key"] = self._normalize_activation_key(view["raw_activation_key"], activation_stats)
                    if slot.get("view_keys"):
                        slot_prototypes, slot_dispersion = self._summarize_slot_prototypes(slot["view_keys"])
                        slot["slot_prototypes"] = self._clone_to_cpu(slot_prototypes)
                        slot["slot_dispersion"] = slot_dispersion
        if self._use_side_memory_shards():
            for shard_id, shard in self.shard_registry.items():
                for atom in self._shard_atoms(shard):
                    for view in atom.get("view_keys") or []:
                        if isinstance(view.get("raw_semantic_key"), torch.Tensor):
                            view["semantic_key"] = self._normalize_semantic_key(view["raw_semantic_key"])
                        if isinstance(view.get("raw_activation_key"), torch.Tensor):
                            view["activation_key"] = self._normalize_activation_key(view["raw_activation_key"], activation_stats)
                    if atom.get("view_keys"):
                        prototypes, dispersion = self._summarize_slot_prototypes(atom["view_keys"])
                        atom["atom_prototypes"] = self._clone_to_cpu(prototypes)
                        atom["atom_dispersion"] = dispersion
                self._refresh_single_shard_metadata(shard_id)
        if self._use_sparse_address_trace_bank():
            self._rebuild_sparse_trace_address_state()

    def _refresh_single_trace_entry_keys(self, entry: dict[str, Any]) -> None:
        activation_stats = None if (self._use_overlap_aware_anchor_trace_bank() or self._use_factored_address_trace_bank()) else self.cached_activation_stats
        if isinstance(entry.get("raw_semantic_key"), torch.Tensor):
            entry["semantic_key"] = self._normalize_semantic_key(entry["raw_semantic_key"])
        if isinstance(entry.get("raw_activation_key"), torch.Tensor):
            entry["activation_key"] = self._normalize_activation_key(entry["raw_activation_key"], activation_stats)
        for view in self._entry_view_records(entry):
            if isinstance(view.get("raw_semantic_key"), torch.Tensor):
                view["semantic_key"] = self._normalize_semantic_key(view["raw_semantic_key"])
            if isinstance(view.get("raw_activation_key"), torch.Tensor):
                view["activation_key"] = self._normalize_activation_key(view["raw_activation_key"], activation_stats)

    def _state_gate_enabled(self) -> bool:
        return bool(self.is_v2 and getattr(self.hparams, "state_gate_enable", False) and getattr(self.hparams, "cell_prototype_strategy", "single") != "single")

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _init_state_gate(self) -> None:
        feature_dim = len(STATE_GATE_FEATURE_NAMES)
        self.state_gate_module = nn.Linear(feature_dim, 1, bias=True)
        self.state_gate_module.to(torch.device("cpu"))
        with torch.no_grad():
            self.state_gate_module.weight.zero_()
            self.state_gate_module.bias.zero_()
        self.state_gate_optimizer = torch.optim.Adam(
            self.state_gate_module.parameters(),
            lr=float(getattr(self.hparams, "state_gate_lr", 1.0e-2) or 1.0e-2),
        )
        self.state_gate_feature_stats: dict[str, torch.Tensor] = {
            "mean": torch.zeros(feature_dim, dtype=torch.float32),
            "std": torch.ones(feature_dim, dtype=torch.float32),
        }
        self.state_gate_replay_buffer: list[dict[str, Any]] = []
        self.state_gate_seen_examples = 0
        self.state_gate_online_updates = 0
        self.state_gate_warm_start_source: list[str] = []
        self.state_gate_runtime_counts = {
            "direct_accepted_count": 0,
            "rerank_triggered_count": 0,
            "direct_win_count": 0,
            "rerank_win_count": 0,
        }
        self.state_gate_ready = False
        if self._state_gate_enabled() and bool(getattr(self.hparams, "state_gate_warm_start_from_replay", True)):
            self._maybe_warm_start_state_gate()

    def _state_gate_feature_tensor(self, feature_dict: dict[str, float | int | None]) -> torch.Tensor:
        values = []
        for name in STATE_GATE_FEATURE_NAMES:
            raw_value = feature_dict.get(name, 0.0)
            values.append(float(0.0 if raw_value is None else raw_value))
        return torch.tensor(values, dtype=torch.float32)

    def _normalize_state_gate_features(self, features: torch.Tensor) -> torch.Tensor:
        mean = self.state_gate_feature_stats.get("mean")
        std = self.state_gate_feature_stats.get("std")
        if mean is None or std is None:
            return features
        return (features - mean) / std.clamp_min(1.0e-6)

    def _predict_state_gate_score(self, feature_dict: dict[str, float | int | None]) -> float:
        if not self._state_gate_enabled():
            return 0.0
        feature_tensor = self._state_gate_feature_tensor(feature_dict).unsqueeze(0)
        normalized = self._normalize_state_gate_features(feature_tensor)
        with torch.no_grad():
            logits = self.state_gate_module(normalized)
            return float(torch.sigmoid(logits)[0, 0].item())

    def _summarize_ranking(self, ranking: list[dict[str, Any]]) -> dict[str, Any]:
        top_ranking = ranking[: min(self._route_top_k(), len(ranking))]
        if not top_ranking:
            return {
                "top_ranking": [],
                "top_scores": [],
                "top1_prob": 0.0,
                "route_margin": 0.0,
                "top1_score": 0.0,
                "top2_score": 0.0,
                "top4_entropy": 0.0,
            }
        values = torch.tensor([row["combined_score"] for row in top_ranking], dtype=torch.float32)
        probs = F.softmax(values * self.hparams.beta, dim=0)
        top_scores = [float(score) for score in probs.tolist()]
        route_margin = top_scores[0] if len(top_scores) == 1 else top_scores[0] - top_scores[1]
        top1_score = float(top_ranking[0]["combined_score"])
        top2_score = float(top_ranking[1]["combined_score"]) if len(top_ranking) > 1 else top1_score
        if len(top_scores) <= 1:
            entropy = 0.0
        else:
            prob_tensor = torch.tensor(top_scores, dtype=torch.float32).clamp_min(1.0e-8)
            entropy = float((-(prob_tensor * prob_tensor.log()).sum() / math.log(len(top_scores))).item())
        return {
            "top_ranking": top_ranking,
            "top_scores": top_scores,
            "top1_prob": float(top_scores[0]),
            "route_margin": float(route_margin),
            "top1_score": top1_score,
            "top2_score": top2_score,
            "top4_entropy": entropy,
        }

    def _state_gate_feature_dict_from_ranking(self, ranking: list[dict[str, Any]]) -> dict[str, float] | None:
        ranking_summary = self._summarize_ranking(ranking)
        top_ranking = ranking_summary["top_ranking"]
        if not top_ranking:
            return None
        top_row = top_ranking[0]
        cell = top_row["cell"]
        bucket = self.bucket_registry.get(cell.get("bucket_id")) if cell.get("bucket_id") else None
        return {
            "direct_top1_prob": ranking_summary["top1_prob"],
            "direct_route_margin": ranking_summary["route_margin"],
            "direct_top1_score": ranking_summary["top1_score"],
            "direct_top2_score": ranking_summary["top2_score"],
            "top4_entropy": ranking_summary["top4_entropy"],
            "prototype_dispersion": float(top_row.get("prototype_dispersion") or 0.0),
            "cross_view_route_gap": float(cell.get("cross_view_route_gap") or 0.0),
            "state_stability_score": float(cell.get("state_stability_score") or 0.0),
            "state_member_count": float(cell.get("member_count") or 0.0),
            "state_tier_consolidated": 1.0 if cell.get("tier") == "consolidated" else 0.0,
            "bucket_dispersion": 0.0 if bucket is None else float(bucket.get("bucket_dispersion") or 0.0),
        }

    def _append_state_gate_score(self, cell_id: str | None, score: float | None) -> None:
        if cell_id is None or score is None or cell_id not in self.cell_registry:
            return
        history = list(self.cell_registry[cell_id].get("state_gate_recent_scores") or [])
        history.append(float(score))
        history = history[-max(8, int(getattr(self.hparams, "stability_min_observations", 8) or 8)) :]
        self.cell_registry[cell_id]["state_gate_recent_scores"] = history

    def _record_state_gate_example(
        self,
        feature_dict: dict[str, float | int | None] | None,
        label: float,
        *,
        source: str,
        cell_id: str | None = None,
        predicted_score: float | None = None,
    ) -> None:
        if not self._state_gate_enabled() or feature_dict is None:
            return
        feature_tensor = self._state_gate_feature_tensor(feature_dict)
        score = predicted_score if predicted_score is not None else self._predict_state_gate_score(feature_dict)
        self.state_gate_replay_buffer.append(
            {
                "features": feature_tensor,
                "label": float(label),
                "source": source,
            }
        )
        max_size = max(1, int(getattr(self.hparams, "state_gate_replay_buffer_size", 4096) or 4096))
        if len(self.state_gate_replay_buffer) > max_size:
            self.state_gate_replay_buffer = self.state_gate_replay_buffer[-max_size:]
        self.state_gate_seen_examples += 1
        if label >= 0.5:
            self.state_gate_runtime_counts["direct_win_count"] += 1
        else:
            self.state_gate_runtime_counts["rerank_win_count"] += 1
        self._append_state_gate_score(cell_id, score)

    def _fit_state_gate_from_buffer(self, epochs: int = 1) -> None:
        if not self._state_gate_enabled() or len(self.state_gate_replay_buffer) < 2:
            return
        features = torch.stack([row["features"] for row in self.state_gate_replay_buffer], dim=0).float()
        labels = torch.tensor([float(row["label"]) for row in self.state_gate_replay_buffer], dtype=torch.float32).unsqueeze(-1)
        mean = features.mean(dim=0)
        std = features.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        self.state_gate_feature_stats = {
            "mean": mean.detach().cpu(),
            "std": std.detach().cpu(),
        }
        features = self._normalize_state_gate_features(features)
        batch_size = max(1, int(getattr(self.hparams, "state_gate_batch_size", 128) or 128))
        self.state_gate_module.train()
        for _ in range(max(1, epochs)):
            permutation = torch.randperm(features.size(0))
            for start in range(0, features.size(0), batch_size):
                batch_idx = permutation[start : start + batch_size]
                batch_features = features[batch_idx]
                batch_labels = labels[batch_idx]
                logits = self.state_gate_module(batch_features)
                loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
                self.state_gate_optimizer.zero_grad()
                loss.backward()
                self.state_gate_optimizer.step()
        self.state_gate_module.eval()
        self.state_gate_ready = True

    def _infer_replay_label(self, replay_dir: Path, entry: dict[str, Any]) -> float | None:
        replay_key = str(replay_dir).lower()
        route_stage = str(entry.get("route_stage", "")).lower()
        if entry.get("correct_route") is not True:
            return None
        if "zsre" in replay_key and "multiview" in replay_key and "rerank" not in replay_key:
            return 1.0
        if "counterfact" in replay_key and "multiview_rerank" in replay_key:
            return 0.0
        if "hopedit_v23" in replay_key:
            if route_stage == "direct_multiview":
                return 1.0
            if route_stage in {"ambiguity_rerank", "cell_rerank"}:
                return 0.0
        if "multiview_rerank" in replay_key:
            return 0.0
        if "multiview" in replay_key and "rerank" not in replay_key:
            return 1.0
        return None

    def _default_state_gate_replay_dirs(self) -> list[Path]:
        repo_root = self._repo_root()
        return [
            repo_root / "outputs" / "hopedit_v22_probe_b200" / "zsre" / "qwen2.5-7b-instruct" / "hopedit_v22_multiview" / "standard_128",
            repo_root / "outputs" / "hopedit_v22_probe_b200" / "counterfact" / "qwen2.5-7b-instruct" / "hopedit_v22_multiview_rerank" / "standard_128",
            repo_root / "outputs" / "hopedit_v23_probe_b200" / "zsre" / "qwen2.5-7b-instruct" / "hopedit_v23" / "standard_128",
            repo_root / "outputs" / "hopedit_v23_probe_b200" / "counterfact" / "qwen2.5-7b-instruct" / "hopedit_v23" / "standard_128",
        ]

    def _replay_log_file(self, replay_dir: Path) -> Path | None:
        for candidate_name in ("annotated_route_logs.jsonl", "hopedit_route_logs_annotated.jsonl"):
            candidate = replay_dir / candidate_name
            if candidate.exists():
                return candidate
        return None

    def _state_gate_feature_dict_from_replay_entry(self, entry: dict[str, Any]) -> dict[str, float]:
        top1_prob = float(entry.get("top1_prob", 0.0) or 0.0)
        route_margin = float(entry.get("route_margin", 0.0) or 0.0)
        top2_score = max(0.0, top1_prob - route_margin)
        if top1_prob <= 0.0 or top1_prob >= 1.0:
            entropy = 0.0
        else:
            entropy = float(-(top1_prob * math.log(top1_prob) + (1.0 - top1_prob) * math.log(max(1.0e-8, 1.0 - top1_prob))))
        return {
            "direct_top1_prob": top1_prob,
            "direct_route_margin": route_margin,
            "direct_top1_score": top1_prob,
            "direct_top2_score": top2_score,
            "top4_entropy": entropy,
            "prototype_dispersion": 0.0,
            "cross_view_route_gap": 0.0,
            "state_stability_score": 0.0,
            "state_member_count": 1.0,
            "state_tier_consolidated": 0.0,
            "bucket_dispersion": 0.0,
        }

    def _maybe_warm_start_state_gate(self) -> None:
        replay_paths = getattr(self.hparams, "state_gate_replay_paths", None) or []
        replay_dirs = [Path(path) for path in replay_paths] if replay_paths else self._default_state_gate_replay_dirs()
        loaded = 0
        sources = []
        for replay_dir in replay_dirs:
            log_file = self._replay_log_file(replay_dir)
            if log_file is None or not log_file.exists():
                continue
            local_loaded = 0
            with open(log_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if entry.get("event_type") not in {"rewrite", "rephrase"}:
                        continue
                    label = self._infer_replay_label(replay_dir, entry)
                    if label is None:
                        continue
                    feature_dict = self._state_gate_feature_dict_from_replay_entry(entry)
                    self._record_state_gate_example(feature_dict, label, source=f"warm_start:{replay_dir.name}")
                    local_loaded += 1
                    loaded += 1
            if local_loaded > 0:
                sources.append(str(replay_dir))
        if loaded > 0:
            self.state_gate_warm_start_source = sources
            self._fit_state_gate_from_buffer(epochs=int(getattr(self.hparams, "state_gate_warm_start_epochs", 8) or 8))

    def _maybe_online_update_state_gate(self) -> None:
        if not self._state_gate_enabled():
            return
        total_edits = len(self.memory_entries)
        interval = max(1, int(getattr(self.hparams, "state_gate_online_update_interval", 16) or 16))
        if total_edits == 0 or total_edits % interval != 0:
            return
        self._fit_state_gate_from_buffer(epochs=1)
        self.state_gate_online_updates += 1

    def _extract_batched_keys(self, semantic_texts: list[str], activation_texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_tokens = self._tokenize(semantic_texts)
        with torch.no_grad():
            input_embeddings = self._model_for_direct_call().get_input_embeddings()(semantic_tokens["input_ids"])
        semantic_keys = self._mean_pool(input_embeddings, semantic_tokens.get("attention_mask"))

        activation_tokens = self._tokenize(activation_texts)
        with torch.no_grad():
            with self._adapter_disabled():
                outputs = self._model_for_direct_call()(**activation_tokens, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        probe_layers = self._resolve_probe_layers(len(hidden_states))
        pooled_layers = [
            self._mean_pool(hidden_states[layer_idx], activation_tokens.get("attention_mask"))
            for layer_idx in probe_layers
        ]
        activation_keys = torch.stack(pooled_layers, dim=0).mean(dim=0)
        return semantic_keys.detach().float().cpu(), activation_keys.detach().float().cpu()

    def _extract_keys(self, semantic_texts: list[str], activation_texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_keys, activation_keys = self._extract_batched_keys(semantic_texts, activation_texts)
        semantic_key = semantic_keys.mean(dim=0)
        activation_key = activation_keys.mean(dim=0)
        return semantic_key.detach().float().cpu(), activation_key.detach().float().cpu()

    def _append_route_log(self, metadata: dict[str, Any], decision: dict[str, Any]) -> None:
        entry = {
            "case_id": metadata.get("case_id"),
            "prompt": metadata.get("prompt"),
            "subject": metadata.get("subject"),
            "memory_unit": decision.get("memory_unit", self.memory_unit),
            "chosen_memory_id": decision.get("chosen_memory_id"),
            "chosen_edit_id": decision.get("chosen_edit_id"),
            "chosen_cell_id": decision.get("chosen_cell_id"),
            "top_memory_ids": decision.get("top_memory_ids", []),
            "top_edit_ids": decision.get("top_edit_ids", []),
            "top_cell_ids": decision.get("top_cell_ids", []),
            "top_scores": decision["top_scores"],
            "top_view_names": decision.get("top_view_names", []),
            "route_margin": decision["route_margin"],
            "route_stage": decision.get("route_stage", self.memory_unit),
            "prototype_match": decision.get("prototype_match"),
            "prototype_ambiguity": decision.get("prototype_ambiguity"),
            "cross_view_target_match": decision.get("cross_view_target_match"),
            "state_gate_score": decision.get("state_gate_score"),
            "gate_decision": decision.get("gate_decision"),
            "selected_slot_ids": decision.get("selected_slot_ids", []),
            "selected_slot_weights": decision.get("selected_slot_weights", []),
            "locality_risk": decision.get("locality_risk"),
            "state_confidence": decision.get("state_confidence"),
            "state_margin": decision.get("state_margin"),
            "realization_overlap": decision.get("realization_overlap"),
            "selected_code_l1_mean": decision.get("selected_code_l1_mean"),
            "selected_code_l2_mean": decision.get("selected_code_l2_mean"),
            "selected_code_nonzero_fraction": decision.get("selected_code_nonzero_fraction"),
            "selected_code_support": decision.get("selected_code_support"),
            "selected_support_overlap": decision.get("selected_support_overlap"),
            "selected_atom_coherence": decision.get("selected_atom_coherence"),
            "trace_energy": decision.get("trace_energy"),
            "trace_energy_margin": decision.get("trace_energy_margin"),
            "trace_negative_energy": decision.get("trace_negative_energy"),
            "trace_locality_margin": decision.get("trace_locality_margin"),
            "trace_exclusion_score": decision.get("trace_exclusion_score"),
            "trace_anchor_energy": decision.get("trace_anchor_energy"),
            "trace_anchor_energy_threshold": decision.get("trace_anchor_energy_threshold"),
            "trace_family_margin": decision.get("trace_family_margin"),
            "trace_family_margin_threshold": decision.get("trace_family_margin_threshold"),
            "trace_family_overlap_score": decision.get("trace_family_overlap_score"),
            "trace_activation_energy_threshold": decision.get("trace_activation_energy_threshold"),
            "trace_activation_margin_threshold": decision.get("trace_activation_margin_threshold"),
            "trace_locality_margin_threshold": decision.get("trace_locality_margin_threshold"),
            "trace_exclusion_threshold": decision.get("trace_exclusion_threshold"),
            "candidate_set_size": decision.get("candidate_set_size"),
            "family_shortlist_size": decision.get("family_shortlist_size"),
            "address_support_size": decision.get("address_support_size"),
            "address_support_overlap": decision.get("address_support_overlap"),
            "address_exclusion_overlap": decision.get("address_exclusion_overlap"),
            "address_family_overlap": decision.get("address_family_overlap"),
            "cross_view_code_agreement": decision.get("cross_view_code_agreement"),
            "address_atom_coherence": decision.get("address_atom_coherence"),
            "address_abstained": decision.get("address_abstained"),
            "address_locality_vetoed": decision.get("address_locality_vetoed"),
            "factor_subject_top_edit_id": decision.get("factor_subject_top_edit_id"),
            "factor_relation_top_edit_id": decision.get("factor_relation_top_edit_id"),
            "factor_subject_energy": decision.get("factor_subject_energy"),
            "factor_relation_energy": decision.get("factor_relation_energy"),
            "factor_subject_margin": decision.get("factor_subject_margin"),
            "factor_relation_margin": decision.get("factor_relation_margin"),
            "factor_subject_pass": decision.get("factor_subject_pass"),
            "factor_relation_pass": decision.get("factor_relation_pass"),
            "factor_same_trace": decision.get("factor_same_trace"),
            "factor_failure_partition": decision.get("factor_failure_partition"),
            "resolved_subject": decision.get("resolved_subject"),
            "query_subject_found": decision.get("query_subject_found"),
            "query_relation_token_count": decision.get("query_relation_token_count"),
            "factor_score_trace_ids": decision.get("factor_score_trace_ids"),
            "factor_subject_scores": decision.get("factor_subject_scores"),
            "factor_relation_scores": decision.get("factor_relation_scores"),
            "factor_relation_storage_transform": decision.get("factor_relation_storage_transform"),
            "factor_relation_score_transform": decision.get("factor_relation_score_transform"),
            "factor_relation_score_transform_active": decision.get("factor_relation_score_transform_active"),
            "factor_relation_whiten_rank": decision.get("factor_relation_whiten_rank"),
            "factor_relation_whiten_num_vectors": decision.get("factor_relation_whiten_num_vectors"),
            "factor_subject_margin_threshold": decision.get("factor_subject_margin_threshold"),
            "factor_relation_margin_threshold": decision.get("factor_relation_margin_threshold"),
            "factor_subject_energy_threshold": decision.get("factor_subject_energy_threshold"),
            "factor_relation_energy_threshold": decision.get("factor_relation_energy_threshold"),
            "base_only_fallback": decision.get("base_only_fallback"),
            "target_edit_id": metadata.get("target_edit_id"),
            "target_cell_id": metadata.get("target_cell_id"),
            "target_memory_id": metadata.get("target_memory_id"),
            "route_match": None,
            "route_event": metadata.get("route_event", "inference"),
        }
        expected_memory_id = metadata.get("target_memory_id")
        if expected_memory_id is None:
            expected_memory_id = metadata.get("target_cell_id") if self.is_v2 else metadata.get("target_edit_id")
        if expected_memory_id is not None and decision.get("chosen_memory_id") is not None:
            entry["route_match"] = decision["chosen_memory_id"] == expected_memory_id
        self.route_logs.append(entry)

    @staticmethod
    def _anchor_view_names() -> set[str]:
        return {"anchor", "prompt"}

    def _score_entry_views(
        self,
        entry: dict[str, Any],
        semantic_key: torch.Tensor,
        activation_key: torch.Tensor,
        allowed_view_names: set[str] | None = None,
    ) -> dict[str, Any] | None:
        best_view = None
        best_combined = None
        best_semantic = None
        best_activation = None
        for view in self._entry_view_records(entry):
            view_name = view.get("view_name")
            if allowed_view_names is not None and view_name not in allowed_view_names:
                continue
            semantic_view = view.get("semantic_key")
            activation_view = view.get("activation_key")
            if not isinstance(semantic_view, torch.Tensor) or not isinstance(activation_view, torch.Tensor):
                continue
            semantic_score = F.cosine_similarity(semantic_view.unsqueeze(0), semantic_key.unsqueeze(0), dim=-1)[0]
            activation_score = F.cosine_similarity(activation_view.unsqueeze(0), activation_key.unsqueeze(0), dim=-1)[0]
            combined = self.hparams.semantic_weight * semantic_score + self.hparams.activation_weight * activation_score
            if best_combined is None or combined.item() > best_combined:
                best_combined = float(combined.item())
                best_semantic = float(semantic_score.item())
                best_activation = float(activation_score.item())
                best_view = view
        if best_combined is None:
            return None
        return {
            "entry": entry,
            "combined_conflict": best_combined,
            "semantic_conflict": best_semantic,
            "activation_conflict": best_activation,
            "best_view_name": best_view.get("view_name") if best_view else None,
            "best_view_text": best_view.get("text") if best_view else None,
        }

    def _cell_view_records(self, cell: dict[str, Any]) -> list[dict[str, Any]]:
        if self._use_sparse_slots():
            prototypes = cell.get("state_summary_prototypes") or cell.get("cell_prototypes") or []
            if prototypes:
                return prototypes
        prototype_strategy = getattr(self.hparams, "cell_prototype_strategy", "single")
        prototypes = cell.get("cell_prototypes") or []
        if prototype_strategy != "single" and prototypes:
            return prototypes
        semantic_key = cell.get("semantic_key")
        activation_key = cell.get("activation_key")
        if not isinstance(semantic_key, torch.Tensor) or not isinstance(activation_key, torch.Tensor):
            return []
        return [
            {
                "view_name": "cell_centroid",
                "text": cell.get("prototype_anchor_text"),
                "semantic_key": semantic_key,
                "activation_key": activation_key,
                "prototype_dispersion": cell.get("prototype_dispersion"),
                "coverage_count": cell.get("member_count"),
            }
        ]

    def _summarize_cell_view_prototypes(self, entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            for view in self._entry_view_records(entry):
                view_name = view.get("view_name") or "unknown"
                grouped.setdefault(view_name, []).append(view)

        prototypes: list[dict[str, Any]] = []
        dispersions: list[float] = []
        per_view_dispersion: dict[str, float | None] = {}
        coverage_counts: dict[str, int] = {}
        max_intra_conflict: float | None = None

        for view_name, views in grouped.items():
            semantic_stack = torch.stack([view["semantic_key"].detach().float().cpu() for view in views], dim=0)
            activation_stack = torch.stack([view["activation_key"].detach().float().cpu() for view in views], dim=0)
            semantic_centroid = self._normalize_semantic_key(semantic_stack.mean(dim=0))
            activation_centroid = self._normalize_activation_key(activation_stack.mean(dim=0), self.cached_activation_stats)

            best_idx = 0
            best_score = None
            dispersion_terms: list[float] = []
            pair_conflicts: list[float] = []
            for idx, view in enumerate(views):
                semantic_score = float(
                    F.cosine_similarity(view["semantic_key"].detach().float().cpu().unsqueeze(0), semantic_centroid.unsqueeze(0), dim=-1)[0].item()
                )
                activation_score = float(
                    F.cosine_similarity(view["activation_key"].detach().float().cpu().unsqueeze(0), activation_centroid.unsqueeze(0), dim=-1)[0].item()
                )
                combined_score = self.hparams.semantic_weight * semantic_score + self.hparams.activation_weight * activation_score
                dispersion_terms.append(1.0 - combined_score)
                if best_score is None or combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            for i in range(len(views)):
                for j in range(i + 1, len(views)):
                    semantic_score = float(
                        F.cosine_similarity(
                            views[i]["semantic_key"].detach().float().cpu().unsqueeze(0),
                            views[j]["semantic_key"].detach().float().cpu().unsqueeze(0),
                            dim=-1,
                        )[0].item()
                    )
                    activation_score = float(
                        F.cosine_similarity(
                            views[i]["activation_key"].detach().float().cpu().unsqueeze(0),
                            views[j]["activation_key"].detach().float().cpu().unsqueeze(0),
                            dim=-1,
                        )[0].item()
                    )
                    pair_conflicts.append(self.hparams.semantic_weight * semantic_score + self.hparams.activation_weight * activation_score)
            prototype_dispersion = None if not dispersion_terms else float(sum(dispersion_terms) / len(dispersion_terms))
            if prototype_dispersion is not None:
                dispersions.append(prototype_dispersion)
            if pair_conflicts:
                max_pair = float(max(pair_conflicts))
                if max_intra_conflict is None or max_pair > max_intra_conflict:
                    max_intra_conflict = max_pair

            representative = views[best_idx]
            prototypes.append(
                {
                    "view_name": view_name,
                    "text": representative.get("text"),
                    "semantic_key": semantic_centroid,
                    "activation_key": activation_centroid,
                    "coverage_count": len(views),
                    "prototype_dispersion": prototype_dispersion,
                }
            )
            per_view_dispersion[view_name] = prototype_dispersion
            coverage_counts[view_name] = len(views)

        stats = {
            "prototype_count_by_view": coverage_counts,
            "prototype_dispersion": None if not dispersions else float(sum(dispersions) / len(dispersions)),
            "prototype_dispersion_by_view": per_view_dispersion,
            "max_intra_cell_prototype_conflict": max_intra_conflict,
        }
        return prototypes, stats

    def _rank_conflicts_from_keys(
        self,
        semantic_key: torch.Tensor,
        activation_key: torch.Tensor,
        allowed_view_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        active_entries = [entry for entry in self.memory_entries if entry["edit_id"] not in self.disabled_adapters]
        if not active_entries:
            return []
        semantic_key = self._normalize_semantic_key(semantic_key)
        if self._use_overlap_aware_anchor_trace_bank():
            activation_key = self._normalize_activation_key(activation_key, None)
        else:
            activation_stats = self.cached_activation_stats
            if activation_stats is None:
                activation_stats = self._activation_stats_from_raw_keys(self._collect_raw_activation_keys(active_entries))
            activation_key = self._normalize_activation_key(activation_key, activation_stats)

        ranking = []
        for entry in active_entries:
            scored = self._score_entry_views(entry, semantic_key, activation_key, allowed_view_names=allowed_view_names)
            if scored is not None:
                ranking.append(scored)
        ranking.sort(key=lambda row: row["combined_conflict"], reverse=True)
        return ranking

    def _decision_from_ranking(
        self,
        ranking: list[dict[str, Any]],
        min_route_prob: float,
        min_route_margin: float,
        route_stage: str,
    ) -> dict[str, Any]:
        if not ranking:
            return {
                "memory_unit": "edit",
                "chosen_memory_id": None,
                "chosen_edit_id": None,
                "chosen_cell_id": None,
                "top_memory_ids": [],
                "top_edit_ids": [],
                "top_cell_ids": [],
                "top_scores": [],
                "route_margin": 0.0,
                "top_view_names": [],
                "route_stage": route_stage,
                "adapter_name": None,
            }

        top_ranking = ranking[: min(self._route_top_k(), len(ranking))]
        values = torch.tensor([row["combined_conflict"] for row in top_ranking], dtype=torch.float32)
        probs = F.softmax(values * self.hparams.beta, dim=0)
        top_scores = [float(score) for score in probs.tolist()]
        route_margin = top_scores[0] if len(top_scores) == 1 else top_scores[0] - top_scores[1]
        chosen_edit_id = top_ranking[0]["entry"]["edit_id"]
        if top_scores[0] < min_route_prob or route_margin < min_route_margin:
            chosen_edit_id = None
        return {
            "memory_unit": "edit",
            "chosen_memory_id": chosen_edit_id,
            "chosen_edit_id": chosen_edit_id,
            "chosen_cell_id": None,
            "top_memory_ids": [row["entry"]["edit_id"] for row in top_ranking],
            "top_edit_ids": [row["entry"]["edit_id"] for row in top_ranking],
            "top_cell_ids": [],
            "top_scores": top_scores,
            "route_margin": float(route_margin),
            "top_view_names": [row.get("best_view_name") for row in top_ranking],
            "route_stage": route_stage,
            "adapter_name": chosen_edit_id,
        }

    def _trace_decision_from_ranking(
        self,
        ranking: list[dict[str, Any]],
        min_route_prob: float,
        min_route_margin: float,
        route_stage: str,
    ) -> dict[str, Any]:
        decision = self._decision_from_ranking(ranking, min_route_prob, min_route_margin, route_stage)
        trace_id = decision.get("chosen_edit_id")
        top_trace_ids = list(decision.get("top_edit_ids") or decision.get("top_memory_ids") or [])
        decision["memory_unit"] = "trace"
        decision["chosen_memory_id"] = trace_id
        decision["top_memory_ids"] = top_trace_ids
        decision["chosen_trace_id"] = trace_id
        decision["top_trace_ids"] = top_trace_ids
        if decision.get("route_stage") == "multiview":
            decision["route_stage"] = "trace_bank_multiview"
        elif decision.get("route_stage") == "anchor":
            decision["route_stage"] = "trace_bank_anchor"
        elif decision.get("route_stage") == "fallback_multiview":
            decision["route_stage"] = "trace_bank_fallback_multiview"
        return decision

    def _rank_sparse_trace_addresses(
        self,
        semantic_key: torch.Tensor,
        activation_key: torch.Tensor,
        allowed_view_names: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], torch.Tensor, list[int], int]:
        if self._use_overlap_aware_anchor_trace_bank():
            ranking, query_code, query_support, candidate_set_size, _ = self._rank_overlap_anchor_trace_addresses(
                semantic_key,
                activation_key,
                allowed_view_names=allowed_view_names,
            )
            return ranking, query_code, query_support, candidate_set_size
        active_entries = [
            entry
            for entry in self.memory_entries
            if isinstance(entry, dict) and entry.get("edit_id") not in self.disabled_adapters
        ]
        atoms = [atom for atom in self.address_dictionary.get("atoms", []) if isinstance(atom, torch.Tensor)]
        if not active_entries or not atoms:
            return [], torch.zeros(len(atoms), dtype=torch.float32), [], 0

        semantic_key = self._normalize_semantic_key(semantic_key)
        activation_stats = self.cached_activation_stats
        if activation_stats is None:
            activation_stats = self._activation_stats_from_raw_keys(self._collect_raw_activation_keys(active_entries))
        activation_key = self._normalize_activation_key(activation_key, activation_stats)
        query_vector = self._address_vector_from_keys(semantic_key, activation_key)
        query_code = self._encode_sparse_address(query_vector, atoms)
        query_support = self._address_support_from_code(query_code)
        candidate_trace_ids = self._candidate_trace_ids_from_query_code(query_code)
        candidate_set_size = len(candidate_trace_ids)
        candidate_entries = [
            self.edit_registry.get(trace_id)
            for trace_id in candidate_trace_ids
            if self.edit_registry.get(trace_id) is not None
        ]
        if not candidate_entries:
            candidate_entries = active_entries
            candidate_set_size = len(candidate_entries)

        ranking = []
        for entry in candidate_entries:
            view_scores = []
            sparse_scores = []
            best_view_name = None
            best_view_score = None
            best_sparse_score = None
            best_support_overlap = 0.0
            for view in self._entry_view_records(entry):
                view_name = view.get("view_name")
                if allowed_view_names is not None and view_name not in allowed_view_names:
                    continue
                dense_score, sparse_score = self._score_sparse_address_view(query_vector, query_code, view)
                if dense_score is None:
                    continue
                view_scores.append(dense_score)
                if sparse_score is not None:
                    sparse_scores.append(sparse_score)
                support_overlap = self._support_overlap(query_support, view.get("address_support") or [])
                if best_view_score is None or dense_score > best_view_score:
                    best_view_score = dense_score
                    best_view_name = view_name
                    best_support_overlap = support_overlap
                    best_sparse_score = sparse_score
            if not view_scores:
                continue
            view_tensor = torch.tensor(view_scores, dtype=torch.float32)
            beta = self._trace_energy_beta()
            temperature = self._trace_energy_temperature()
            trace_energy = float((temperature * torch.logsumexp(view_tensor * beta, dim=0) / beta).item())
            exclusion_code = entry.get("trace_exclusion_code")
            exclusion_score = None
            exclusion_overlap = None
            if isinstance(exclusion_code, torch.Tensor) and exclusion_code.numel() == query_code.numel():
                exclusion_score = float(torch.dot(query_code.detach().float().cpu(), exclusion_code.detach().float().cpu()).item())
                exclusion_overlap = self._support_overlap(query_support, entry.get("trace_exclusion_support") or [])
            ranking.append(
                {
                    "entry": entry,
                    "combined_conflict": trace_energy,
                    "trace_energy": trace_energy,
                    "best_view_name": best_view_name,
                    "best_sparse_score": best_sparse_score,
                    "mean_sparse_score": None if not sparse_scores else float(sum(sparse_scores) / len(sparse_scores)),
                    "address_support_overlap": best_support_overlap,
                    "address_exclusion_overlap": exclusion_overlap,
                    "cross_view_code_agreement": entry.get("trace_address_agreement"),
                    "trace_exclusion_score": exclusion_score,
                    "candidate_set_size": candidate_set_size,
                    "query_support_size": len(query_support),
                }
            )
        ranking.sort(key=lambda row: row["trace_energy"], reverse=True)
        return ranking, query_code, query_support, candidate_set_size

    def _sparse_trace_decision_from_ranking(
        self,
        ranking: list[dict[str, Any]],
        query_code: torch.Tensor,
        query_support: list[int],
        candidate_set_size: int,
        route_stage: str,
        count_fallback: bool = True,
    ) -> dict[str, Any]:
        if not ranking:
            if count_fallback:
                self.base_only_fallback_count += 1
            return {
                "memory_unit": "trace",
                "chosen_memory_id": None,
                "chosen_edit_id": None,
                "chosen_cell_id": None,
                "top_memory_ids": [],
                "top_edit_ids": [],
                "top_cell_ids": [],
                "top_scores": [],
                "route_margin": 0.0,
                "top_view_names": [],
                "route_stage": route_stage,
                "adapter_name": None,
                "trace_energy": None,
                "trace_energy_margin": None,
                "trace_negative_energy": None,
                "trace_locality_margin": None,
                "trace_exclusion_score": None,
                "candidate_set_size": candidate_set_size,
                "address_support_size": len(query_support),
                "address_support_overlap": None,
                "address_exclusion_overlap": None,
                "cross_view_code_agreement": None,
                "address_atom_coherence": self.address_dictionary.get("coherence_mean"),
                "address_abstained": True,
                "address_locality_vetoed": False,
                "base_only_fallback": True,
            }

        top_ranking = ranking[: min(self._route_top_k(), len(ranking))]
        score_key = "combined_conflict" if self._use_overlap_aware_anchor_trace_bank() else "trace_energy"
        values = torch.tensor([row[score_key] for row in top_ranking], dtype=torch.float32)
        probs = F.softmax(values * self.hparams.beta, dim=0)
        top_scores = [float(score) for score in probs.tolist()]
        route_margin = top_scores[0] if len(top_scores) == 1 else top_scores[0] - top_scores[1]
        trace_energy = float(top_ranking[0]["trace_energy"])
        trace_energy_margin = (
            float(top_ranking[0][score_key])
            if len(top_ranking) == 1
            else float(top_ranking[0][score_key] - top_ranking[1][score_key])
        )
        trace_exclusion_score = top_ranking[0].get("trace_exclusion_score")
        trace_anchor_energy = top_ranking[0].get("trace_anchor_energy")
        trace_family_negative_energy = top_ranking[0].get("trace_family_negative_energy")
        trace_family_margin = top_ranking[0].get("trace_family_margin")
        family_overlap_score = top_ranking[0].get("family_overlap_score")
        chosen_entry = top_ranking[0]["entry"]
        chosen_edit_id = chosen_entry["edit_id"]
        calibrated_energy_threshold = chosen_entry.get("trace_min_activation_energy")
        calibrated_margin_threshold = chosen_entry.get("trace_min_activation_margin")
        calibrated_locality_threshold = chosen_entry.get("trace_max_exclusion_score")
        calibrated_anchor_threshold = chosen_entry.get("trace_min_anchor_energy")
        calibrated_family_margin = chosen_entry.get("trace_min_family_margin")
        if calibrated_energy_threshold is not None:
            activation_energy_threshold = float(max(0.0, calibrated_energy_threshold))
        else:
            activation_energy_threshold = self._trace_abstain_min_energy()
        if calibrated_margin_threshold is not None:
            activation_margin_threshold = float(max(0.0, calibrated_margin_threshold))
        else:
            activation_margin_threshold = self._trace_abstain_margin()
        locality_margin_threshold = None
        if bool(getattr(self.hparams, "trace_use_locality_veto", True)):
            if calibrated_locality_threshold is not None:
                locality_margin_threshold = float(max(0.0, calibrated_locality_threshold))
            else:
                locality_margin_threshold = 1.0
        anchor_energy_threshold = None if calibrated_anchor_threshold is None else float(max(0.0, calibrated_anchor_threshold))
        family_margin_threshold = None if calibrated_family_margin is None else float(max(0.0, calibrated_family_margin))
        address_abstained = False
        locality_vetoed = False
        trace_locality_margin = None
        if locality_margin_threshold is not None and trace_exclusion_score is not None:
            trace_locality_margin = float(locality_margin_threshold - float(trace_exclusion_score))
        if trace_energy < activation_energy_threshold or trace_energy_margin < activation_margin_threshold:
            chosen_edit_id = None
            address_abstained = True
            if count_fallback:
                self.base_only_fallback_count += 1
        elif (
            self._use_overlap_aware_anchor_trace_bank()
            and anchor_energy_threshold is not None
            and trace_anchor_energy is not None
            and float(trace_anchor_energy) < anchor_energy_threshold
        ):
            chosen_edit_id = None
            address_abstained = True
            if count_fallback:
                self.base_only_fallback_count += 1
        elif (
            self._use_overlap_aware_anchor_trace_bank()
            and family_margin_threshold is not None
            and trace_family_margin is not None
            and float(trace_family_margin) < family_margin_threshold
        ):
            chosen_edit_id = None
            address_abstained = True
            locality_vetoed = True
            if count_fallback:
                self.base_only_fallback_count += 1
        elif (
            locality_margin_threshold is not None
            and trace_exclusion_score is not None
            and float(trace_exclusion_score) > locality_margin_threshold
        ):
            chosen_edit_id = None
            address_abstained = True
            locality_vetoed = True
            if count_fallback:
                self.base_only_fallback_count += 1
        return {
            "memory_unit": "trace",
            "chosen_memory_id": chosen_edit_id,
            "chosen_edit_id": chosen_edit_id,
            "chosen_cell_id": None,
            "top_memory_ids": [row["entry"]["edit_id"] for row in top_ranking],
            "top_edit_ids": [row["entry"]["edit_id"] for row in top_ranking],
            "top_cell_ids": [],
            "top_scores": top_scores,
            "route_margin": float(route_margin),
            "top_view_names": [row.get("best_view_name") for row in top_ranking],
            "route_stage": route_stage,
            "adapter_name": chosen_edit_id,
            "prototype_match": top_ranking[0].get("best_view_name"),
            "prototype_ambiguity": None if len(top_ranking) == 1 else float(max(0.0, 1.0 - trace_energy_margin)),
            "trace_energy": trace_energy,
            "trace_energy_margin": trace_energy_margin,
            "trace_negative_energy": None
            if trace_exclusion_score is None and trace_family_negative_energy is None
            else float(trace_exclusion_score if trace_exclusion_score is not None else trace_family_negative_energy),
            "trace_locality_margin": None if trace_locality_margin is None else float(trace_locality_margin),
            "trace_exclusion_score": None if trace_exclusion_score is None else float(trace_exclusion_score),
            "trace_anchor_energy": None if trace_anchor_energy is None else float(trace_anchor_energy),
            "trace_anchor_energy_threshold": anchor_energy_threshold,
            "trace_family_margin": None if trace_family_margin is None else float(trace_family_margin),
            "trace_family_margin_threshold": family_margin_threshold,
            "trace_family_overlap_score": None if family_overlap_score is None else float(family_overlap_score),
            "trace_activation_energy_threshold": float(activation_energy_threshold),
            "trace_activation_margin_threshold": float(activation_margin_threshold),
            "trace_locality_margin_threshold": locality_margin_threshold,
            "trace_exclusion_threshold": locality_margin_threshold,
            "candidate_set_size": candidate_set_size,
            "address_support_size": len(query_support),
            "address_support_overlap": top_ranking[0].get("address_support_overlap"),
            "address_exclusion_overlap": top_ranking[0].get("address_exclusion_overlap"),
            "address_family_overlap": top_ranking[0].get("family_support_overlap"),
            "cross_view_code_agreement": top_ranking[0].get("cross_view_code_agreement"),
            "address_atom_coherence": self.address_dictionary.get("coherence_mean"),
            "family_shortlist_size": top_ranking[0].get("family_shortlist_size"),
            "address_abstained": address_abstained,
            "address_locality_vetoed": locality_vetoed,
            "base_only_fallback": address_abstained,
        }

    def _cell_entries(self, cell_id: str) -> list[dict[str, Any]]:
        return [entry for entry in self.memory_entries if entry.get("cell_id") == cell_id]

    def _active_cells(self) -> list[dict[str, Any]]:
        active = []
        for cell_id, cell in self.cell_registry.items():
            adapter_name = cell.get("adapter_name")
            if adapter_name in self.disabled_adapters:
                continue
            active.append(cell)
        return active

    def _active_tier_cells(self) -> list[dict[str, Any]]:
        return [cell for cell in self._active_cells() if cell.get("tier", "active") != "consolidated"]

    def _consolidated_cells(self) -> list[dict[str, Any]]:
        return [cell for cell in self._active_cells() if cell.get("tier") == "consolidated"]

    def _bucket_cells(self, bucket_id: str) -> list[dict[str, Any]]:
        bucket = self.bucket_registry.get(bucket_id) or {}
        state_ids = bucket.get("state_ids") or []
        return [self.cell_registry[cell_id] for cell_id in state_ids if cell_id in self.cell_registry]

    def _hierarchy_enabled(self) -> bool:
        if not self.is_v2:
            return False
        if not bool(getattr(self.hparams, "hierarchy_enable", True)):
            return False
        if len(self.memory_entries) < int(getattr(self.hparams, "hierarchy_start_edit", 1024) or 1024):
            return False
        return bool(self.bucket_registry)

    def _state_route_observation_count(self, cell_id: str) -> int:
        count = 0
        for route_entry in self.route_logs:
            if route_entry.get("chosen_memory_id") != cell_id:
                continue
            if route_entry.get("route_match") is True:
                count += 1
        return count

    def _compute_state_locality_proxy(self, cell: dict[str, Any]) -> float:
        member_count = int(cell.get("member_count") or 0)
        if member_count <= 1:
            return 1.0
        within_conflict = float(cell.get("within_cell_conflict_mean") or 0.0)
        return max(0.0, min(1.0, 1.0 - within_conflict))

    @staticmethod
    def _merge_numeric_feature_dicts(feature_dicts: list[dict[str, float | int | None]]) -> dict[str, float]:
        merged: dict[str, float] = {}
        for name in STATE_GATE_FEATURE_NAMES:
            values = [
                float(value)
                for feature_dict in feature_dicts
                for value in [feature_dict.get(name)]
                if value is not None
            ]
            merged[name] = 0.0 if not values else float(sum(values) / len(values))
        return merged

    def _mean_recent_state_gate_score(self, cell: dict[str, Any]) -> float | None:
        history = [float(score) for score in (cell.get("state_gate_recent_scores") or []) if score is not None]
        if not history:
            return None
        return float(sum(history) / len(history))

    def _collect_state_gate_training_signal(self, entry: dict[str, Any]) -> None:
        if not self._state_gate_enabled():
            return
        target_cell_id = entry.get("cell_id")
        if target_cell_id is None or target_cell_id not in self.cell_registry:
            return

        pair_views = [
            view
            for view in self._entry_view_records(entry)
            if view.get("view_name") in {"prompt", "rephrase"}
        ]
        if not pair_views:
            return

        feature_dicts: list[dict[str, float]] = []
        direct_hits: list[bool] = []
        rerank_hits: list[bool] = []
        direct_ids: list[str | None] = []

        for view in pair_views:
            semantic_key = view.get("semantic_key")
            activation_key = view.get("activation_key")
            if not isinstance(semantic_key, torch.Tensor) or not isinstance(activation_key, torch.Tensor):
                continue
            direct_ranking = self._rank_cells_from_keys(semantic_key, activation_key)
            feature_dict = self._state_gate_feature_dict_from_ranking(direct_ranking)
            if feature_dict is not None:
                feature_dicts.append(feature_dict)
            direct_decision = self._decision_from_cell_ranking(
                direct_ranking,
                self.hparams.min_route_prob,
                self.hparams.min_route_margin,
                "direct_multiview",
            )
            reranked = self._rerank_cells_from_keys(
                direct_ranking,
                semantic_key,
                activation_key,
                rerank_topk=max(1, int(getattr(self.hparams, "cell_prototype_rerank_topk", 4) or 4)),
                penalty=float(getattr(self.hparams, "cell_rerank_dispersion_penalty", 0.05) or 0.0),
            )
            rerank_decision = self._decision_from_cell_ranking(
                reranked,
                self.hparams.min_route_prob,
                self.hparams.min_route_margin,
                "ambiguity_rerank",
            )
            direct_ids.append(direct_decision.get("chosen_cell_id"))
            direct_hits.append(direct_decision.get("chosen_cell_id") == target_cell_id)
            rerank_hits.append(rerank_decision.get("chosen_cell_id") == target_cell_id)

        if not feature_dicts:
            return

        merged_features = self._merge_numeric_feature_dicts(feature_dicts)
        non_null_direct_ids = [cell_id for cell_id in direct_ids if cell_id is not None]
        direct_consistent = len(set(non_null_direct_ids)) <= 1
        direct_pair_ok = bool(direct_hits) and all(direct_hits) and direct_consistent
        rerank_pair_ok = bool(rerank_hits) and all(rerank_hits)
        label = 1.0 if direct_pair_ok else 0.0
        if not direct_pair_ok and not rerank_pair_ok and direct_consistent:
            return
        predicted_score = self._predict_state_gate_score(merged_features) if self.state_gate_ready else None
        self._record_state_gate_example(
            merged_features,
            label,
            source=f"online:{entry.get('edit_id')}",
            cell_id=target_cell_id,
            predicted_score=predicted_score,
        )

    def _rank_specific_cells_from_keys(
        self,
        cells: list[dict[str, Any]],
        semantic_key: torch.Tensor,
        activation_key: torch.Tensor,
        *,
        rerank_topk: int | None = None,
        rerank_penalty: float | None = None,
    ) -> list[dict[str, Any]]:
        if not cells:
            return []
        semantic_key = self._normalize_semantic_key(semantic_key)
        activation_stats = self.cached_activation_stats
        if activation_stats is None:
            activation_stats = self._activation_stats_from_raw_keys(self._collect_raw_activation_keys())
        activation_key = self._normalize_activation_key(activation_key, activation_stats)
        ranking = []
        for cell in cells:
            scored = self._score_cell(cell, semantic_key, activation_key)
            if scored is not None:
                ranking.append(scored)
        ranking.sort(key=lambda row: row["combined_score"], reverse=True)
        if self._use_sparse_slots():
            return ranking
        if rerank_topk is not None and rerank_topk > 1 and ranking:
            ranking = self._rerank_cells_from_keys(
                ranking,
                semantic_key,
                activation_key,
                rerank_topk=rerank_topk,
                penalty=rerank_penalty,
            )
        return ranking

    def _evaluate_state_cross_view(self, cell: dict[str, Any]) -> dict[str, Any]:
        entries = self._cell_entries(cell["cell_id"])
        if not entries:
            return {
                "prompt_route_accuracy": None,
                "rephrase_route_accuracy": None,
                "cross_view_route_gap": None,
                "num_observations": 0,
            }
        candidate_cells = self._active_cells()
        prompt_hits = 0
        prompt_total = 0
        rephrase_hits = 0
        rephrase_total = 0
        for entry in entries:
            for view in self._entry_view_records(entry):
                view_name = view.get("view_name")
                if view_name not in {"prompt", "rephrase"}:
                    continue
                semantic_key = view.get("semantic_key")
                activation_key = view.get("activation_key")
                if not isinstance(semantic_key, torch.Tensor) or not isinstance(activation_key, torch.Tensor):
                    continue
                ranking = self._rank_specific_cells_from_keys(
                    candidate_cells,
                    semantic_key,
                    activation_key,
                    rerank_topk=max(1, int(getattr(self.hparams, "cell_prototype_rerank_topk", 4))),
                    rerank_penalty=float(getattr(self.hparams, "cell_rerank_dispersion_penalty", 0.05) or 0.0),
                )
                best_cell_id = ranking[0]["cell"]["cell_id"] if ranking else None
                if view_name == "prompt":
                    prompt_total += 1
                    if best_cell_id == cell["cell_id"]:
                        prompt_hits += 1
                elif view_name == "rephrase":
                    rephrase_total += 1
                    if best_cell_id == cell["cell_id"]:
                        rephrase_hits += 1
        prompt_acc = None if prompt_total == 0 else float(prompt_hits / prompt_total)
        rephrase_acc = None if rephrase_total == 0 else float(rephrase_hits / rephrase_total)
        gap = None
        if prompt_acc is not None and rephrase_acc is not None:
            gap = abs(prompt_acc - rephrase_acc)
        return {
            "prompt_route_accuracy": prompt_acc,
            "rephrase_route_accuracy": rephrase_acc,
            "cross_view_route_gap": gap,
            "num_observations": prompt_total + rephrase_total,
        }

    def _state_is_stable(self, cell: dict[str, Any]) -> bool:
        support = int(cell.get("state_support_observations") or 0)
        min_observations = int(getattr(self.hparams, "stability_min_observations", 8) or 8)
        cross_view_gap = cell.get("cross_view_route_gap")
        prototype_dispersion = cell.get("prototype_dispersion")
        within_state_conflict = cell.get("within_cell_conflict_mean")
        locality_proxy = cell.get("locality_proxy")
        gate_score_mean = cell.get("state_gate_score_mean")
        if support < min_observations:
            return False
        if self._state_gate_enabled():
            if gate_score_mean is None or gate_score_mean < float(getattr(self.hparams, "state_gate_promotion_threshold", 0.80) or 0.80):
                return False
        if cross_view_gap is None or cross_view_gap > float(getattr(self.hparams, "stability_max_cross_view_gap", 0.10)):
            return False
        if prototype_dispersion is None or prototype_dispersion > float(getattr(self.hparams, "stability_max_prototype_dispersion", 0.30)):
            return False
        if within_state_conflict is not None and within_state_conflict > float(getattr(self.hparams, "stability_max_within_state_conflict", 0.30)):
            return False
        if locality_proxy is None or locality_proxy < float(getattr(self.hparams, "stability_min_locality", 0.95)):
            return False
        return True

    def _refresh_single_state_metadata(self, cell_id: str, *, total_edits: int, warmup: int) -> None:
        cell = self.cell_registry.get(cell_id)
        if cell is None:
            return
        created_at = int(cell.get("created_at_edit_index", total_edits))
        age = max(1, total_edits - created_at + 1)
        cross_view = self._evaluate_state_cross_view(cell)
        locality_proxy = self._compute_state_locality_proxy(cell)
        support = max(self._state_route_observation_count(cell_id), cross_view.get("num_observations", 0))
        gate_score_mean = self._mean_recent_state_gate_score(cell)
        cell["state_age_edits"] = age
        cell["state_support_observations"] = support
        cell["prompt_route_accuracy"] = cross_view.get("prompt_route_accuracy")
        cell["rephrase_route_accuracy"] = cross_view.get("rephrase_route_accuracy")
        cell["cross_view_route_gap"] = cross_view.get("cross_view_route_gap")
        cell["locality_proxy"] = locality_proxy
        cell["locality_fragility"] = 1.0 - locality_proxy if locality_proxy is not None else None
        cell["state_gate_score_mean"] = gate_score_mean
        score_terms = [
            cell.get("prompt_route_accuracy"),
            cell.get("rephrase_route_accuracy"),
            None if cell.get("prototype_dispersion") is None else 1.0 - min(1.0, float(cell.get("prototype_dispersion"))),
            None if cell.get("within_cell_conflict_mean") is None else 1.0 - min(1.0, float(cell.get("within_cell_conflict_mean"))),
            locality_proxy,
            gate_score_mean,
        ]
        filtered_terms = [float(term) for term in score_terms if term is not None]
        cell["state_stability_score"] = None if not filtered_terms else float(sum(filtered_terms) / len(filtered_terms))
        stable = total_edits >= warmup and self._state_is_stable(cell)
        cell["is_stable"] = bool(stable)
        cell["tier"] = "consolidated" if stable else "active"
        if not stable:
            cell["bucket_id"] = None

    def _refresh_state_metadata(self) -> None:
        total_edits = len(self.memory_entries)
        warmup = int(getattr(self.hparams, "memory_tier_warmup_edits", 256) or 256)
        for cell_id, cell in sorted(self.cell_registry.items()):
            self._refresh_single_state_metadata(cell_id, total_edits=total_edits, warmup=warmup)

    def _score_bucket(self, bucket: dict[str, Any], semantic_key: torch.Tensor, activation_key: torch.Tensor) -> dict[str, Any] | None:
        best = None
        for prototype in bucket.get("bucket_prototypes", []):
            semantic_view = prototype.get("semantic_key")
            activation_view = prototype.get("activation_key")
            if not isinstance(semantic_view, torch.Tensor) or not isinstance(activation_view, torch.Tensor):
                continue
            semantic_score = F.cosine_similarity(semantic_view.unsqueeze(0), semantic_key.unsqueeze(0), dim=-1)[0]
            activation_score = F.cosine_similarity(activation_view.unsqueeze(0), activation_key.unsqueeze(0), dim=-1)[0]
            combined = self.hparams.semantic_weight * semantic_score + self.hparams.activation_weight * activation_score
            candidate = {
                "bucket": bucket,
                "combined_score": float(combined.item()),
                "best_view_name": prototype.get("view_name"),
                "prototype_dispersion": bucket.get("bucket_dispersion"),
            }
            if best is None or candidate["combined_score"] > best["combined_score"]:
                best = candidate
        return best

    def _summarize_bucket_prototypes(self, cells: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float | None]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for cell in cells:
            for prototype in self._cell_view_records(cell):
                view_name = prototype.get("view_name") or "unknown"
                grouped.setdefault(view_name, []).append(prototype)
        prototypes = []
        dispersions = []
        for view_name, views in grouped.items():
            semantic_stack = torch.stack([view["semantic_key"].detach().float().cpu() for view in views], dim=0)
            activation_stack = torch.stack([view["activation_key"].detach().float().cpu() for view in views], dim=0)
            semantic_centroid = self._normalize_semantic_key(semantic_stack.mean(dim=0))
            activation_centroid = self._normalize_activation_key(activation_stack.mean(dim=0), self.cached_activation_stats)
            dispersion_terms = []
            for view in views:
                semantic_score = float(
                    F.cosine_similarity(view["semantic_key"].detach().float().cpu().unsqueeze(0), semantic_centroid.unsqueeze(0), dim=-1)[0].item()
                )
                activation_score = float(
                    F.cosine_similarity(view["activation_key"].detach().float().cpu().unsqueeze(0), activation_centroid.unsqueeze(0), dim=-1)[0].item()
                )
                dispersion_terms.append(1.0 - (self.hparams.semantic_weight * semantic_score + self.hparams.activation_weight * activation_score))
            prototype_dispersion = None if not dispersion_terms else float(sum(dispersion_terms) / len(dispersion_terms))
            if prototype_dispersion is not None:
                dispersions.append(prototype_dispersion)
            prototypes.append(
                {
                    "view_name": view_name,
                    "semantic_key": semantic_centroid,
                    "activation_key": activation_centroid,
                    "prototype_dispersion": prototype_dispersion,
                }
            )
        return prototypes, (None if not dispersions else float(sum(dispersions) / len(dispersions)))

    def _rebuild_bucket_registry(self) -> None:
        self.bucket_registry = {}
        if not bool(getattr(self.hparams, "hierarchy_enable", True)):
            return
        if len(self.memory_entries) < int(getattr(self.hparams, "hierarchy_start_edit", 1024) or 1024):
            return
        consolidated = sorted(self._consolidated_cells(), key=lambda cell: cell["cell_id"])
        if not consolidated:
            return
        max_size = int(getattr(self.hparams, "bucket_max_size", 256) or 256)
        split_dispersion = float(getattr(self.hparams, "bucket_split_dispersion", 0.35) or 0.35)
        bucket_index = 0
        for cell in consolidated:
            assigned_bucket = None
            best_score = None
            for bucket_id, bucket in self.bucket_registry.items():
                if len(bucket.get("state_ids", [])) >= max_size:
                    continue
                prototypes = bucket.get("bucket_prototypes") or []
                if not prototypes:
                    continue
                score = self._score_bucket(bucket, cell["semantic_key"], cell["activation_key"])
                if score is None:
                    continue
                if best_score is None or score["combined_score"] > best_score["combined_score"]:
                    best_score = score
                    assigned_bucket = bucket_id
            if assigned_bucket is None:
                assigned_bucket = f"hopedit_bucket_{bucket_index:05d}"
                bucket_index += 1
                self.bucket_registry[assigned_bucket] = {
                    "bucket_id": assigned_bucket,
                    "state_ids": [],
                    "bucket_prototypes": [],
                    "bucket_dispersion": None,
                }
            tentative_cells = self._bucket_cells(assigned_bucket) + [cell]
            bucket_prototypes, bucket_dispersion = self._summarize_bucket_prototypes(tentative_cells)
            if bucket_dispersion is not None and bucket_dispersion > split_dispersion and self.bucket_registry[assigned_bucket]["state_ids"]:
                assigned_bucket = f"hopedit_bucket_{bucket_index:05d}"
                bucket_index += 1
                tentative_cells = [cell]
                bucket_prototypes, bucket_dispersion = self._summarize_bucket_prototypes(tentative_cells)
                self.bucket_registry[assigned_bucket] = {
                    "bucket_id": assigned_bucket,
                    "state_ids": [],
                    "bucket_prototypes": [],
                    "bucket_dispersion": None,
                }
            bucket = self.bucket_registry[assigned_bucket]
            bucket["state_ids"].append(cell["cell_id"])
            bucket["bucket_prototypes"] = bucket_prototypes
            bucket["bucket_dispersion"] = bucket_dispersion
            cell["bucket_id"] = assigned_bucket

    def _rank_buckets_from_keys(self, semantic_key: torch.Tensor, activation_key: torch.Tensor) -> list[dict[str, Any]]:
        if not self.bucket_registry:
            return []
        semantic_key = self._normalize_semantic_key(semantic_key)
        activation_stats = self.cached_activation_stats
        if activation_stats is None:
            activation_stats = self._activation_stats_from_raw_keys(self._collect_raw_activation_keys())
        activation_key = self._normalize_activation_key(activation_key, activation_stats)
        ranking = []
        for bucket in self.bucket_registry.values():
            scored = self._score_bucket(bucket, semantic_key, activation_key)
            if scored is not None:
                ranking.append(scored)
        ranking.sort(key=lambda row: row["combined_score"], reverse=True)
        return ranking

    def _score_cell(self, cell: dict[str, Any], semantic_key: torch.Tensor, activation_key: torch.Tensor) -> dict[str, Any] | None:
        if self._use_sparse_slots():
            best_slot_row = None
            for slot in self._state_slots(cell):
                scored = self._score_slot(slot, semantic_key, activation_key)
                if scored is None:
                    continue
                if best_slot_row is None or scored["combined_score"] > best_slot_row["combined_score"]:
                    best_slot_row = scored
            if best_slot_row is None:
                return None
            return {
                "cell": cell,
                "combined_score": float(best_slot_row["combined_score"]),
                "semantic_score": float(best_slot_row["semantic_score"]),
                "activation_score": float(best_slot_row["activation_score"]),
                "best_view_name": best_slot_row.get("best_view_name"),
                "best_view_text": best_slot_row.get("best_view_text"),
                "best_prototype_score": float(best_slot_row["combined_score"]),
                "prototype_dispersion": cell.get("prototype_dispersion"),
                "slot_dispersion": best_slot_row.get("slot_dispersion"),
                "best_slot_row": best_slot_row,
                "best_slot_id": best_slot_row["slot"].get("slot_id"),
            }
        best = None
        for prototype in self._cell_view_records(cell):
            semantic_view = prototype.get("semantic_key")
            activation_view = prototype.get("activation_key")
            if not isinstance(semantic_view, torch.Tensor) or not isinstance(activation_view, torch.Tensor):
                continue
            semantic_score = F.cosine_similarity(semantic_view.unsqueeze(0), semantic_key.unsqueeze(0), dim=-1)[0]
            activation_score = F.cosine_similarity(activation_view.unsqueeze(0), activation_key.unsqueeze(0), dim=-1)[0]
            combined = self.hparams.semantic_weight * semantic_score + self.hparams.activation_weight * activation_score
            candidate = {
                "cell": cell,
                "combined_score": float(combined.item()),
                "semantic_score": float(semantic_score.item()),
                "activation_score": float(activation_score.item()),
                "best_view_name": prototype.get("view_name"),
                "best_view_text": prototype.get("text"),
                "best_prototype_score": float(combined.item()),
                "prototype_dispersion": prototype.get("prototype_dispersion"),
            }
            if best is None or candidate["combined_score"] > best["combined_score"]:
                best = candidate
        return best

    def _rerank_cells_from_keys(
        self,
        ranking: list[dict[str, Any]],
        semantic_key: torch.Tensor,
        activation_key: torch.Tensor,
        rerank_topk: int | None = None,
        penalty: float | None = None,
    ) -> list[dict[str, Any]]:
        if rerank_topk is None:
            rerank_topk = max(1, int(getattr(self.hparams, "cell_prototype_rerank_topk", 4)))
        if penalty is None:
            penalty = float(getattr(self.hparams, "cell_rerank_dispersion_penalty", 0.05) or 0.0)
        reranked = []
        for row in ranking[: min(rerank_topk, len(ranking))]:
            cell = row["cell"]
            best = None
            for prototype in self._cell_view_records(cell):
                semantic_view = prototype.get("semantic_key")
                activation_view = prototype.get("activation_key")
                if not isinstance(semantic_view, torch.Tensor) or not isinstance(activation_view, torch.Tensor):
                    continue
                semantic_score = F.cosine_similarity(semantic_view.unsqueeze(0), semantic_key.unsqueeze(0), dim=-1)[0]
                activation_score = F.cosine_similarity(activation_view.unsqueeze(0), activation_key.unsqueeze(0), dim=-1)[0]
                combined = self.hparams.semantic_weight * semantic_score + self.hparams.activation_weight * activation_score
                prototype_dispersion = float(prototype.get("prototype_dispersion") or 0.0)
                rerank_score = float(combined.item()) - penalty * prototype_dispersion
                candidate = dict(row)
                candidate.update(
                    {
                        "combined_score": rerank_score,
                        "semantic_score": float(semantic_score.item()),
                        "activation_score": float(activation_score.item()),
                        "best_view_name": prototype.get("view_name"),
                        "best_view_text": prototype.get("text"),
                        "best_prototype_score": float(combined.item()),
                        "prototype_dispersion": prototype.get("prototype_dispersion"),
                    }
                )
                if best is None or candidate["combined_score"] > best["combined_score"]:
                    best = candidate
            if best is not None:
                reranked.append(best)
        reranked.sort(key=lambda row: row["combined_score"], reverse=True)
        return reranked

    def _candidate_cells_from_keys(self, semantic_key: torch.Tensor, activation_key: torch.Tensor) -> list[dict[str, Any]]:
        active_cells = self._active_tier_cells()
        if not self._hierarchy_enabled():
            return self._active_cells()
        candidate_map = {cell["cell_id"]: cell for cell in active_cells}
        bucket_topk = max(1, int(getattr(self.hparams, "bucket_topk", 2) or 2))
        for bucket_row in self._rank_buckets_from_keys(semantic_key, activation_key)[:bucket_topk]:
            for cell in self._bucket_cells(bucket_row["bucket"]["bucket_id"]):
                candidate_map[cell["cell_id"]] = cell
        return list(candidate_map.values())

    def _rank_cells_from_keys(self, semantic_key: torch.Tensor, activation_key: torch.Tensor) -> list[dict[str, Any]]:
        candidate_cells = self._candidate_cells_from_keys(semantic_key, activation_key)
        if not candidate_cells:
            return []
        return self._rank_specific_cells_from_keys(candidate_cells, semantic_key, activation_key)

    def _decision_from_cell_ranking(
        self,
        ranking: list[dict[str, Any]],
        min_route_prob: float,
        min_route_margin: float,
        route_stage: str,
        *,
        state_gate_score: float | None = None,
        gate_decision: str | None = None,
    ) -> dict[str, Any]:
        if not ranking:
            return {
                "memory_unit": "cell",
                "chosen_memory_id": None,
                "chosen_edit_id": None,
                "chosen_cell_id": None,
                "top_memory_ids": [],
                "top_edit_ids": [],
                "top_cell_ids": [],
                "top_scores": [],
                "route_margin": 0.0,
                "top_view_names": [],
                "route_stage": route_stage,
                "adapter_name": None,
                "prototype_match": None,
                "prototype_ambiguity": None,
                "cross_view_target_match": None,
                "state_gate_score": state_gate_score,
                "gate_decision": gate_decision,
            }

        top_ranking = ranking[: min(self._route_top_k(), len(ranking))]
        values = torch.tensor([row["combined_score"] for row in top_ranking], dtype=torch.float32)
        probs = F.softmax(values * self.hparams.beta, dim=0)
        top_scores = [float(score) for score in probs.tolist()]
        route_margin = top_scores[0] if len(top_scores) == 1 else top_scores[0] - top_scores[1]
        chosen_cell = top_ranking[0]["cell"]
        chosen_cell_id = chosen_cell["cell_id"]
        if top_scores[0] < min_route_prob or route_margin < min_route_margin:
            chosen_cell_id = None
        chosen_cell = chosen_cell if chosen_cell_id is not None else None
        return {
            "memory_unit": "cell",
            "chosen_memory_id": chosen_cell_id,
            "chosen_edit_id": None,
            "chosen_cell_id": chosen_cell_id,
            "top_memory_ids": [row["cell"]["cell_id"] for row in top_ranking],
            "top_edit_ids": [],
            "top_cell_ids": [row["cell"]["cell_id"] for row in top_ranking],
            "top_scores": top_scores,
            "top_view_names": [row.get("best_view_name") for row in top_ranking],
            "route_margin": float(route_margin),
            "route_stage": route_stage,
            "adapter_name": None if chosen_cell is None else chosen_cell.get("adapter_name"),
            "prototype_match": None if chosen_cell is None else top_ranking[0].get("best_view_name"),
            "prototype_ambiguity": float(route_margin),
            "cross_view_target_match": None,
            "state_gate_score": state_gate_score,
            "gate_decision": gate_decision,
        }

    def _select_state_slots(
        self,
        cell: dict[str, Any],
        semantic_key: torch.Tensor,
        activation_key: torch.Tensor,
    ) -> list[dict[str, Any]]:
        slot_rows = []
        for slot in self._state_slots(cell):
            scored = self._score_slot(slot, semantic_key, activation_key)
            if scored is not None:
                scored["cell"] = cell
                slot_rows.append(scored)
        slot_rows.sort(key=lambda row: row["combined_score"], reverse=True)
        return self._normalize_slot_selection(slot_rows[: self._slot_top_k()])

    def _build_sparse_state_decision(
        self,
        ranking: list[dict[str, Any]],
        semantic_key: torch.Tensor,
        activation_key: torch.Tensor,
    ) -> dict[str, Any]:
        empty = self._decision_from_cell_ranking(ranking, self.hparams.min_route_prob, self.hparams.min_route_margin, "sparse_multiview")
        if not ranking:
            return empty

        finalists = [ranking[0]]
        finalist_margin = float(getattr(self.hparams, "state_finalist_margin", 0.05) or 0.05)
        if len(ranking) > 1 and float(ranking[0]["combined_score"] - ranking[1]["combined_score"]) <= finalist_margin:
            finalists.append(ranking[1])

        finalist_rows = []
        for finalist in finalists:
            slot_rows = self._select_state_slots(finalist["cell"], semantic_key, activation_key)
            if not slot_rows:
                continue
            best_slot_score = float(slot_rows[0]["combined_score"])
            slot_dispersion = None if slot_rows[0].get("slot_dispersion") is None else float(slot_rows[0].get("slot_dispersion"))
            finalist_rows.append(
                {
                    **finalist,
                    "selected_slot_rows": slot_rows,
                    "best_slot_score": best_slot_score,
                    "slot_dispersion": slot_dispersion,
                }
            )
        if not finalist_rows:
            return empty

        runner_up = finalist_rows[1] if len(finalist_rows) > 1 else None
        for row in finalist_rows:
            row["aggregate_codes"] = self._aggregate_selected_slot_codes(row["cell"], row["selected_slot_rows"])
            row["code_stats"] = self._factor_space_code_stats(row["aggregate_codes"])
            row["locality_risk"] = self._state_locality_risk(row, runner_up if runner_up is not None and runner_up["cell"]["cell_id"] != row["cell"]["cell_id"] else None)
            row["support_overlap"] = None if runner_up is None else self._support_overlap(
                row["aggregate_codes"],
                self._aggregate_selected_slot_codes(runner_up["cell"], runner_up["selected_slot_rows"]),
            )
            row["atom_coherence"] = self._active_atom_coherence(row["cell"], row["aggregate_codes"])
            row["realization_overlap"] = None if runner_up is None else self._cross_state_realization_overlap(
                row["cell"],
                row["aggregate_codes"],
                runner_up["cell"],
                self._aggregate_selected_slot_codes(runner_up["cell"], runner_up["selected_slot_rows"]),
            )
            row["final_score"] = float(
                row["best_slot_score"]
                - 0.05 * float(row.get("slot_dispersion") or 0.0)
                - 0.10 * float(row.get("locality_risk") or 0.0)
            )
        finalist_rows.sort(key=lambda row: row["final_score"], reverse=True)
        chosen = finalist_rows[0]
        state_confidence = None
        if len(finalist_rows) == 1:
            state_confidence = 1.0
        else:
            state_confidence = float(chosen["final_score"] - finalist_rows[1]["final_score"])

        selected_slot_rows = [dict(row) for row in chosen["selected_slot_rows"]]
        locality_risk = float(chosen.get("locality_risk") or 0.0)
        if locality_risk > float(getattr(self.hparams, "locality_risk_threshold", 0.35) or 0.35) and len(selected_slot_rows) > 1:
            selected_slot_rows = selected_slot_rows[:1]
            chosen["aggregate_codes"] = self._aggregate_selected_slot_codes(chosen["cell"], selected_slot_rows)
            chosen["code_stats"] = self._factor_space_code_stats(chosen["aggregate_codes"])
        chosen_cell_id = chosen["cell"]["cell_id"]
        if state_confidence is not None and state_confidence < 0.48 and locality_risk > 0.50:
            chosen_cell_id = None

        top_ranking = ranking[: min(self._route_top_k(), len(ranking))]
        values = torch.tensor([row["combined_score"] for row in top_ranking], dtype=torch.float32)
        probs = F.softmax(values * self.hparams.beta, dim=0)
        top_scores = [float(score) for score in probs.tolist()]
        route_margin = top_scores[0] if len(top_scores) == 1 else top_scores[0] - top_scores[1]
        adapter_name = None if chosen_cell_id is None else self._load_composed_slot_adapter(selected_slot_rows)
        for row in selected_slot_rows:
            row["slot"]["slot_usage_count"] = int(row["slot"].get("slot_usage_count") or 0) + 1

        return {
            "memory_unit": self.memory_unit,
            "chosen_memory_id": chosen_cell_id,
            "chosen_edit_id": None,
            "chosen_cell_id": chosen_cell_id,
            "top_memory_ids": [row["cell"]["cell_id"] for row in top_ranking],
            "top_edit_ids": [],
            "top_cell_ids": [row["cell"]["cell_id"] for row in top_ranking],
            "top_scores": top_scores,
            "top_view_names": [row.get("best_view_name") for row in top_ranking],
            "route_margin": float(route_margin),
            "route_stage": "sparse_multiview",
            "adapter_name": adapter_name,
            "prototype_match": chosen.get("best_view_name"),
            "prototype_ambiguity": None if state_confidence is None else float(max(0.0, 1.0 - state_confidence)),
            "cross_view_target_match": None,
            "state_gate_score": None,
            "gate_decision": None,
            "selected_slot_ids": [row["slot"]["slot_id"] for row in selected_slot_rows],
            "selected_slot_weights": [float(row["slot_weight"]) for row in selected_slot_rows],
            "locality_risk": locality_risk,
            "state_confidence": state_confidence,
            "state_margin": state_confidence,
            "realization_overlap": chosen.get("realization_overlap"),
            "selected_code_l1_mean": chosen.get("code_stats", {}).get("code_l1_mean"),
            "selected_code_l2_mean": chosen.get("code_stats", {}).get("code_l2_mean"),
            "selected_code_nonzero_fraction": chosen.get("code_stats", {}).get("code_nonzero_fraction"),
            "selected_code_support": chosen.get("code_stats", {}).get("code_support"),
            "selected_support_overlap": chosen.get("support_overlap"),
            "selected_atom_coherence": chosen.get("atom_coherence"),
        }

    def _route_from_keys_v1(self, semantic_key: torch.Tensor, activation_key: torch.Tensor, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = metadata or {}
        route_strategy = getattr(self.hparams, "route_strategy", "multiview")
        if route_strategy == "anchor":
            ranking = self._rank_conflicts_from_keys(semantic_key, activation_key, allowed_view_names=self._anchor_view_names())
            decision = self._decision_from_ranking(ranking, self.hparams.min_route_prob, self.hparams.min_route_margin, "anchor")
            self._append_route_log(metadata, decision)
            return decision

        if route_strategy == "staged":
            anchor_ranking = self._rank_conflicts_from_keys(semantic_key, activation_key, allowed_view_names=self._anchor_view_names())
            anchor_decision = self._decision_from_ranking(
                anchor_ranking,
                self.hparams.min_route_prob,
                self.hparams.min_route_margin,
                "anchor",
            )
            if anchor_decision["chosen_edit_id"] is not None:
                self._append_route_log(metadata, anchor_decision)
                return anchor_decision

            fallback_min_prob = self.hparams.fallback_min_route_prob
            if fallback_min_prob is None:
                fallback_min_prob = self.hparams.min_route_prob
            fallback_min_margin = self.hparams.fallback_min_route_margin
            if fallback_min_margin is None:
                fallback_min_margin = self.hparams.min_route_margin

            fallback_ranking = self._rank_conflicts_from_keys(semantic_key, activation_key)
            fallback_decision = self._decision_from_ranking(
                fallback_ranking,
                fallback_min_prob,
                fallback_min_margin,
                "fallback_multiview",
            )
            self._append_route_log(metadata, fallback_decision)
            return fallback_decision

        ranking = self._rank_conflicts_from_keys(semantic_key, activation_key)
        decision = self._decision_from_ranking(
            ranking,
            self.hparams.min_route_prob,
            self.hparams.min_route_margin,
            "multiview",
        )
        self._append_route_log(metadata, decision)
        return decision

    def _route_from_keys_v4(self, semantic_key: torch.Tensor, activation_key: torch.Tensor, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = metadata or {}
        if self._use_factored_address_trace_bank():
            query = self._extract_factored_query([str(metadata.get("prompt") or "")], metadata=metadata)
            decision = self._route_from_factored_query(query)
            self._append_route_log(metadata, decision)
            return decision
        if self._use_sparse_address_trace_bank():
            route_strategy = getattr(self.hparams, "route_strategy", "multiview")
            if route_strategy == "anchor":
                ranking, query_code, query_support, candidate_set_size = self._rank_sparse_trace_addresses(
                    semantic_key,
                    activation_key,
                    allowed_view_names=self._anchor_view_names(),
                )
                decision = self._sparse_trace_decision_from_ranking(
                    ranking,
                    query_code,
                    query_support,
                    candidate_set_size,
                    "trace_sparse_address_anchor",
                )
                self._append_route_log(metadata, decision)
                return decision

            if route_strategy == "staged":
                anchor_ranking, anchor_code, anchor_support, anchor_candidates = self._rank_sparse_trace_addresses(
                    semantic_key,
                    activation_key,
                    allowed_view_names=self._anchor_view_names(),
                )
                anchor_decision = self._sparse_trace_decision_from_ranking(
                    anchor_ranking,
                    anchor_code,
                    anchor_support,
                    anchor_candidates,
                    "trace_sparse_address_anchor",
                    count_fallback=False,
                )
                if anchor_decision["chosen_memory_id"] is not None:
                    self._append_route_log(metadata, anchor_decision)
                    return anchor_decision

                fallback_ranking, fallback_code, fallback_support, fallback_candidates = self._rank_sparse_trace_addresses(
                    semantic_key,
                    activation_key,
                )
                fallback_decision = self._sparse_trace_decision_from_ranking(
                    fallback_ranking,
                    fallback_code,
                    fallback_support,
                    fallback_candidates,
                    "trace_sparse_address_multiview",
                )
                self._append_route_log(metadata, fallback_decision)
                return fallback_decision

            ranking, query_code, query_support, candidate_set_size = self._rank_sparse_trace_addresses(
                semantic_key,
                activation_key,
            )
            decision = self._sparse_trace_decision_from_ranking(
                ranking,
                query_code,
                query_support,
                candidate_set_size,
                "trace_sparse_address_multiview",
            )
            self._append_route_log(metadata, decision)
            return decision

        route_strategy = getattr(self.hparams, "route_strategy", "multiview")
        if route_strategy == "anchor":
            ranking = self._rank_conflicts_from_keys(semantic_key, activation_key, allowed_view_names=self._anchor_view_names())
            decision = self._trace_decision_from_ranking(
                ranking,
                self.hparams.min_route_prob,
                self.hparams.min_route_margin,
                "anchor",
            )
            self._append_route_log(metadata, decision)
            return decision

        if route_strategy == "staged":
            anchor_ranking = self._rank_conflicts_from_keys(semantic_key, activation_key, allowed_view_names=self._anchor_view_names())
            anchor_decision = self._trace_decision_from_ranking(
                anchor_ranking,
                self.hparams.min_route_prob,
                self.hparams.min_route_margin,
                "anchor",
            )
            if anchor_decision["chosen_memory_id"] is not None:
                self._append_route_log(metadata, anchor_decision)
                return anchor_decision

            fallback_min_prob = self.hparams.fallback_min_route_prob
            if fallback_min_prob is None:
                fallback_min_prob = self.hparams.min_route_prob
            fallback_min_margin = self.hparams.fallback_min_route_margin
            if fallback_min_margin is None:
                fallback_min_margin = self.hparams.min_route_margin

            fallback_ranking = self._rank_conflicts_from_keys(semantic_key, activation_key)
            fallback_decision = self._trace_decision_from_ranking(
                fallback_ranking,
                fallback_min_prob,
                fallback_min_margin,
                "fallback_multiview",
            )
            self._append_route_log(metadata, fallback_decision)
            return fallback_decision

        ranking = self._rank_conflicts_from_keys(semantic_key, activation_key)
        decision = self._trace_decision_from_ranking(
            ranking,
            self.hparams.min_route_prob,
            self.hparams.min_route_margin,
            "multiview",
        )
        self._append_route_log(metadata, decision)
        return decision

    def _route_from_keys_v2(self, semantic_key: torch.Tensor, activation_key: torch.Tensor, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = metadata or {}
        ranking = self._rank_cells_from_keys(semantic_key, activation_key)
        if self._use_sparse_slots():
            decision = self._build_sparse_state_decision(ranking, semantic_key, activation_key)
            self._append_route_log(metadata, decision)
            return decision
        prototype_strategy = getattr(self.hparams, "cell_prototype_strategy", "single")
        if prototype_strategy == "single":
            decision = self._decision_from_cell_ranking(
                ranking,
                self.hparams.min_route_prob,
                self.hparams.min_route_margin,
                "single_prototype",
            )
        else:
            gate_score = None
            gate_decision = None
            if self._state_gate_enabled() and self.state_gate_ready and ranking:
                feature_dict = self._state_gate_feature_dict_from_ranking(ranking)
                gate_score = None if feature_dict is None else self._predict_state_gate_score(feature_dict)
                gate_threshold = float(getattr(self.hparams, "state_gate_direct_threshold", 0.70) or 0.70)
                use_direct = gate_score is not None and gate_score >= gate_threshold
                gate_decision = "direct_accept" if use_direct else "rerank"
                if use_direct:
                    decision = self._decision_from_cell_ranking(
                        ranking,
                        self.hparams.min_route_prob,
                        self.hparams.min_route_margin,
                        "direct_multiview",
                        state_gate_score=gate_score,
                        gate_decision=gate_decision,
                    )
                    self.state_gate_runtime_counts["direct_accepted_count"] += 1
                else:
                    reranked = self._rerank_cells_from_keys(ranking, semantic_key, activation_key)
                    decision = self._decision_from_cell_ranking(
                        reranked,
                        self.hparams.min_route_prob,
                        self.hparams.min_route_margin,
                        "ambiguity_rerank",
                        state_gate_score=gate_score,
                        gate_decision=gate_decision,
                    )
                    self.state_gate_runtime_counts["rerank_triggered_count"] += 1
            else:
                direct_min_prob = max(
                    float(self.hparams.min_route_prob),
                    float(getattr(self.hparams, "cell_direct_accept_min_prob", 0.52) or 0.52),
                )
                direct_min_margin = max(
                    float(self.hparams.min_route_margin),
                    float(getattr(self.hparams, "cell_direct_accept_min_margin", 0.05) or 0.05),
                )
                direct_decision = self._decision_from_cell_ranking(
                    ranking,
                    direct_min_prob,
                    direct_min_margin,
                    "direct_multiview",
                    gate_decision="heuristic_threshold",
                )
                if direct_decision.get("chosen_cell_id") is not None or not ranking or max(1, int(getattr(self.hparams, "cell_prototype_rerank_topk", 4))) <= 1:
                    decision = direct_decision
                else:
                    reranked = self._rerank_cells_from_keys(ranking, semantic_key, activation_key)
                    decision = self._decision_from_cell_ranking(
                        reranked,
                        self.hparams.min_route_prob,
                        self.hparams.min_route_margin,
                        "ambiguity_rerank",
                        gate_decision="heuristic_rerank",
                    )
        self._append_route_log(metadata, decision)
        return decision

    def _route_from_keys_v3(self, semantic_key: torch.Tensor, activation_key: torch.Tensor, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = metadata or {}
        ranking = self._rank_shards_from_keys(semantic_key, activation_key)
        if not ranking:
            decision = {
                "memory_unit": "shard",
                "chosen_memory_id": None,
                "chosen_edit_id": None,
                "chosen_cell_id": None,
                "top_memory_ids": [],
                "top_edit_ids": [],
                "top_cell_ids": [],
                "top_scores": [],
                "route_margin": 0.0,
                "top_view_names": [],
                "route_stage": "base_only",
                "adapter_name": None,
                "selected_slot_ids": [],
                "selected_slot_weights": [],
                "locality_risk": 0.0,
                "state_confidence": 0.0,
                "state_margin": 0.0,
                "realization_overlap": None,
                "selected_code_l1_mean": 0.0,
                "selected_code_l2_mean": 0.0,
                "selected_code_nonzero_fraction": 0.0,
                "selected_code_support": 0,
                "selected_support_overlap": None,
                "selected_atom_coherence": None,
                "base_only_fallback": True,
            }
            self.base_only_fallback_count += 1
            self._append_route_log(metadata, decision)
            return decision

        finalist_rows = ranking[: min(self._route_top_k(), len(ranking))]
        finalist_decisions = []
        for row in finalist_rows:
            shard = row["shard"]
            atom_rows = self._rank_atoms_in_shard(shard, semantic_key, activation_key, apply_usage_penalty=False)
            selected_atoms = self._normalize_atom_selection(atom_rows, topk=self._side_memory_read_topk())
            stats = self._side_memory_support_stats(selected_atoms)
            finalist_decisions.append(
                {
                    "shard": shard,
                    "combined_score": float(row["combined_score"] + sum(float(atom_row.get("atom_weight", 0.0)) for atom_row in selected_atoms)),
                    "shard_score": float(row["combined_score"]),
                    "selected_atoms": selected_atoms,
                    "stats": stats,
                    "best_view_name": row.get("best_view_name"),
                }
            )
        finalist_decisions.sort(key=lambda row: row["combined_score"], reverse=True)
        finalist_values = torch.tensor([float(row["combined_score"]) for row in finalist_decisions], dtype=torch.float32)
        finalist_probs = F.softmax(finalist_values * self.hparams.beta, dim=0)
        chosen = finalist_decisions[0]
        runner_up = finalist_decisions[1] if len(finalist_decisions) > 1 else None
        margin = float(finalist_probs[0].item()) if len(finalist_decisions) == 1 else float(finalist_probs[0].item() - finalist_probs[1].item())
        fallback_threshold = float(getattr(self.hparams, "base_only_fallback_threshold", 0.0) or 0.0)
        adapter_name = None
        route_stage = "base_only"
        if margin > fallback_threshold and chosen["selected_atoms"]:
            adapter_name = self._load_side_memory_runtime_adapter(chosen["selected_atoms"])
            route_stage = "side_memory"
        else:
            self.base_only_fallback_count += 1
            chosen["shard"]["base_only_fallbacks"] = int(chosen["shard"].get("base_only_fallbacks") or 0) + 1
        chosen["shard"].setdefault("prototype_margin_history", []).append(float(margin))
        decision = {
            "memory_unit": "shard",
            "chosen_memory_id": chosen["shard"]["shard_id"] if adapter_name is not None else None,
            "chosen_edit_id": None,
            "chosen_cell_id": None,
            "top_memory_ids": [row["shard"]["shard_id"] for row in finalist_rows],
            "top_edit_ids": [],
            "top_cell_ids": [],
            "top_scores": [float(score) for score in finalist_probs.tolist()],
            "route_margin": float(margin),
            "top_view_names": [row.get("best_view_name") for row in finalist_rows],
            "route_stage": route_stage,
            "adapter_name": adapter_name,
            "selected_slot_ids": [row["atom"]["atom_id"] for row in chosen["selected_atoms"]],
            "selected_slot_weights": [float(row.get("atom_weight", 0.0)) for row in chosen["selected_atoms"]],
            "locality_risk": chosen["stats"]["support_overlap"],
            "state_confidence": float(chosen["combined_score"]),
            "state_margin": float(margin),
            "realization_overlap": chosen["stats"]["realization_overlap"],
            "selected_code_l1_mean": chosen["stats"]["code_l1_mean"],
            "selected_code_l2_mean": chosen["stats"]["code_l2_mean"],
            "selected_code_nonzero_fraction": chosen["stats"]["code_nonzero_fraction"],
            "selected_code_support": chosen["stats"]["support_size"],
            "selected_support_overlap": chosen["stats"]["support_overlap"],
            "selected_atom_coherence": chosen["stats"]["atom_coherence"],
            "base_only_fallback": adapter_name is None,
        }
        self._append_route_log(metadata, decision)
        return decision

    def _route_from_keys(self, semantic_key: torch.Tensor, activation_key: torch.Tensor, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.is_v3:
            return self._route_from_keys_v3(semantic_key, activation_key, metadata=metadata)
        if self.is_v2:
            return self._route_from_keys_v2(semantic_key, activation_key, metadata=metadata)
        if self.is_v4:
            return self._route_from_keys_v4(semantic_key, activation_key, metadata=metadata)
        return self._route_from_keys_v1(semantic_key, activation_key, metadata=metadata)

    def _extract_texts_from_inputs(self, input_ids: torch.Tensor) -> list[str]:
        return self.tok.batch_decode(input_ids, skip_special_tokens=True)

    def _select_adapter_for_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        texts = self._extract_texts_from_inputs(input_ids)
        semantic_key, activation_key = self._extract_keys(texts, texts)
        decision = self._route_from_keys(
            semantic_key,
            activation_key,
            metadata=metadata or {"prompt": texts[0] if texts else None},
        )
        adapter_name = decision.get("adapter_name")
        if adapter_name is not None:
            cache_hit = None
            if self._use_cold_trace_values():
                adapter_name, cache_hit = self._materialize_trace_value(adapter_name)
                decision["adapter_name"] = adapter_name
                decision["value_cache_hit"] = cache_hit
                decision["value_cache_size"] = len(self.trace_value_cache)
                if self.route_logs:
                    self.route_logs[-1]["value_cache_hit"] = cache_hit
                    self.route_logs[-1]["value_cache_size"] = len(self.trace_value_cache)
            self._set_runtime_adapter(adapter_name)
        return decision

    def _call_model(self, *args, **kwargs):
        model = self._model_for_direct_call()
        if len(args) == 1 and isinstance(args[0], dict):
            return model(**args[0])
        if "batch" in kwargs and isinstance(kwargs["batch"], dict):
            batch = kwargs.pop("batch")
            return model(**batch, **kwargs)
        return model(*args, **kwargs)

    def _infer_inputs(self, args, kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            batch = args[0]
            return batch.get("input_ids"), batch.get("attention_mask")
        if len(args) >= 1 and torch.is_tensor(args[0]):
            return args[0], kwargs.get("attention_mask")
        if "batch" in kwargs and isinstance(kwargs["batch"], dict):
            batch = kwargs["batch"]
            return batch.get("input_ids"), batch.get("attention_mask")
        return kwargs.get("input_ids"), kwargs.get("attention_mask")

    def forward(self, *args, **kwargs):
        metadata = kwargs.pop("metadata", None)
        input_ids, attention_mask = self._infer_inputs(args, kwargs)
        if input_ids is None or not self.memory_entries:
            with self._adapter_disabled():
                return self._call_model(*args, **kwargs)
        decision = self._select_adapter_for_inputs(input_ids, attention_mask, metadata=metadata)
        if decision.get("adapter_name") is None:
            with self._adapter_disabled():
                return self._call_model(*args, **kwargs)
        return self._call_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        metadata = kwargs.pop("metadata", None)
        input_ids, attention_mask = self._infer_inputs(args, kwargs)
        if input_ids is None or not self.memory_entries:
            with self._adapter_disabled():
                return self._model_for_direct_call().generate(*args, **kwargs)
        decision = self._select_adapter_for_inputs(input_ids, attention_mask, metadata=metadata)
        if decision.get("adapter_name") is None:
            with self._adapter_disabled():
                return self._model_for_direct_call().generate(*args, **kwargs)
        return self._model_for_direct_call().generate(*args, **kwargs)

    def _training_batch(self, prompt: str, target: str) -> dict[str, torch.Tensor]:
        full_prompt = f"{prompt} {target}".strip()
        prompt_ids = self.tok(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.hparams.max_length,
        )["input_ids"]
        tokens = self.tok(
            [full_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.hparams.max_length,
        )
        labels = tokens["input_ids"].clone()
        pad_token_id = self.tok.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
            prompt_pad = int((prompt_ids[0] == pad_token_id).sum())
            prompt_len = int((prompt_ids[0] != pad_token_id).sum())
        else:
            prompt_pad = 0
            prompt_len = prompt_ids.shape[1]
        labels[0, prompt_pad : prompt_pad + prompt_len] = -100
        tokens["labels"] = labels
        return {key: value.to(self.device) for key, value in tokens.items()}

    def _build_training_views(self, request: dict[str, Any], prompt: str, subject_prompt: str) -> list[str]:
        views = [prompt]
        address_rephrase_prompt = request.get("address_rephrase_prompt") or request.get("rephrase_prompt")
        if self.hparams.use_rephrase_prompt and address_rephrase_prompt:
            views.append(str(address_rephrase_prompt))
        if self.hparams.use_subject_prompt:
            views.append(subject_prompt)
        deduped = []
        seen = set()
        for view in views:
            view = " ".join(str(view).split())
            if view and view not in seen:
                seen.add(view)
                deduped.append(view)
        return deduped

    def _build_view_key_records(self, request: dict[str, Any], prompt: str, subject_prompt: str) -> list[dict[str, Any]]:
        deduped_views: list[tuple[str, str]] = []
        seen = set()
        candidate_views = [("prompt", prompt)]
        address_rephrase_prompt = request.get("address_rephrase_prompt") or request.get("rephrase_prompt")
        if self.hparams.use_rephrase_prompt and address_rephrase_prompt:
            candidate_views.append(("rephrase", str(address_rephrase_prompt)))
        if self.hparams.use_subject_prompt and not self._use_factored_address_trace_bank():
            candidate_views.append(("subject", subject_prompt))
        for view_name, text in candidate_views:
            normalized_text = " ".join(str(text).split())
            if not normalized_text or normalized_text in seen:
                continue
            seen.add(normalized_text)
            deduped_views.append((view_name, normalized_text))

        if not deduped_views:
            return []

        texts = [text for _, text in deduped_views]
        semantic_keys, activation_keys = self._extract_batched_keys(texts, texts)
        factor_rows: list[dict[str, Any]] = []
        if self._use_factored_address_trace_bank():
            subject_texts = [request.get("subject") for _ in texts]
            object_texts = [request.get("target_new") for _ in texts]
            factor_rows = self._extract_batched_factored_address_keys(texts, subject_texts, object_texts)
        view_records = []
        for idx, (view_name, normalized_text) in enumerate(deduped_views):
            raw_semantic_key = semantic_keys[idx]
            raw_activation_key = activation_keys[idx]
            row = {
                "view_name": view_name,
                "text": normalized_text,
                "raw_semantic_key": raw_semantic_key,
                "raw_activation_key": raw_activation_key,
                "semantic_key": raw_semantic_key.clone(),
                "activation_key": raw_activation_key.clone(),
            }
            if factor_rows:
                factor_row = factor_rows[idx]
                row["subject_factor"] = factor_row.get("subject_factor")
                row["relation_factor"] = factor_row.get("relation_factor")
                row["relation_raw_factor"] = factor_row.get("relation_raw_factor")
                row["subject_found"] = bool(factor_row.get("subject_found"))
                row["relation_token_count"] = int(factor_row.get("relation_token_count") or 0)
                row["factored_layer_index"] = factor_row.get("layer_index")
            view_records.append(row)
        return view_records

    def _address_vector_from_keys(self, semantic_key: torch.Tensor, activation_key: torch.Tensor) -> torch.Tensor | None:
        if not isinstance(semantic_key, torch.Tensor) or not isinstance(activation_key, torch.Tensor):
            return None
        semantic_weight = max(0.0, float(self.hparams.semantic_weight))
        activation_weight = max(0.0, float(self.hparams.activation_weight))
        address_vector = torch.cat(
            [
                math.sqrt(semantic_weight) * semantic_key.detach().float().cpu(),
                math.sqrt(activation_weight) * activation_key.detach().float().cpu(),
            ],
            dim=0,
        )
        return self._l2_normalize(address_vector)

    def _address_vector_from_view(self, view: dict[str, Any]) -> torch.Tensor | None:
        cached = view.get("address_vector")
        if isinstance(cached, torch.Tensor):
            return cached.detach().float().cpu()
        semantic_key = view.get("semantic_key")
        activation_key = view.get("activation_key")
        if not isinstance(semantic_key, torch.Tensor) or not isinstance(activation_key, torch.Tensor):
            return None
        return self._address_vector_from_keys(semantic_key, activation_key)

    def _factored_factor_dim(self, factor_name: str) -> int:
        for entry in self.memory_entries:
            if not isinstance(entry, dict) or entry.get("edit_id") in self.disabled_adapters:
                continue
            vector = entry.get(f"trace_{factor_name}_factor")
            if isinstance(vector, torch.Tensor):
                return int(vector.numel())
        return 0

    def _factored_combined_address_vector(
        self,
        subject_factor: torch.Tensor | None,
        relation_factor: torch.Tensor | None,
    ) -> torch.Tensor | None:
        subject_dim = self._factored_factor_dim("subject")
        relation_dim = self._factored_factor_dim("relation")
        if isinstance(subject_factor, torch.Tensor):
            subject_factor = subject_factor.detach().float().cpu()
            subject_dim = int(subject_factor.numel())
        if isinstance(relation_factor, torch.Tensor):
            relation_factor = relation_factor.detach().float().cpu()
            relation_dim = int(relation_factor.numel())
        if subject_dim <= 0 and relation_dim <= 0:
            return None
        if not isinstance(subject_factor, torch.Tensor):
            subject_factor = torch.zeros(subject_dim, dtype=torch.float32)
        if not isinstance(relation_factor, torch.Tensor):
            relation_factor = torch.zeros(relation_dim, dtype=torch.float32)
        subject_weight = max(0.0, self._factored_subject_weight())
        relation_weight = max(0.0, self._factored_relation_weight())
        combined = torch.cat(
            [
                math.sqrt(subject_weight) * subject_factor,
                math.sqrt(relation_weight) * relation_factor,
            ],
            dim=0,
        )
        return self._l2_normalize(combined)

    def _resolved_subject_from_text(self, text: str, metadata: dict[str, Any] | None = None) -> str | None:
        metadata = metadata or {}
        if bool(getattr(self.hparams, "factored_use_subject_metadata", True)):
            subject = metadata.get("subject")
            if subject:
                return str(subject)
        resolution_mode = str(getattr(self.hparams, "factored_subject_resolution", "metadata_or_substring") or "metadata_or_substring")
        if resolution_mode == "metadata_only":
            return None
        normalized_text = self._normalize_family_token(text)
        best_match = None
        best_length = -1
        for entry in self.memory_entries:
            if not isinstance(entry, dict) or entry.get("edit_id") in self.disabled_adapters:
                continue
            candidate = self._normalize_family_token(entry.get("subject"))
            if not candidate or len(candidate) < 2:
                continue
            pattern = rf"(?<!\w){re.escape(candidate)}(?!\w)"
            if re.search(pattern, normalized_text):
                if len(candidate) > best_length:
                    best_match = entry.get("subject")
                    best_length = len(candidate)
        return None if best_match is None else str(best_match)

    def _extract_factored_query(self, texts: list[str], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = metadata or {}
        prompt_text = texts[0] if texts else ""
        subject_text = self._resolved_subject_from_text(prompt_text, metadata=metadata)
        factor_rows = self._extract_batched_factored_address_keys(
            texts or [""],
            [subject_text],
            [None],
        )
        row = factor_rows[0] if factor_rows else {
            "subject_factor": None,
            "relation_factor": None,
            "subject_found": False,
            "relation_token_count": 0,
            "layer_index": None,
        }
        subject_factor = row.get("subject_factor")
        relation_factor = row.get("relation_factor")
        combined_vector = self._factored_combined_address_vector(subject_factor, relation_factor)
        atoms = [atom for atom in self.address_dictionary.get("atoms", []) if isinstance(atom, torch.Tensor)]
        query_code = self._encode_sparse_address(combined_vector, atoms)
        return {
            "prompt": prompt_text,
            "resolved_subject": subject_text,
            "subject_factor": subject_factor,
            "relation_factor": relation_factor,
            "combined_vector": combined_vector,
            "query_code": query_code,
            "query_support": self._address_support_from_code(query_code),
            "subject_found": bool(row.get("subject_found")),
            "relation_token_count": int(row.get("relation_token_count") or 0),
            "layer_index": row.get("layer_index"),
        }

    def _address_support_from_code(self, code: torch.Tensor) -> list[int]:
        if not isinstance(code, torch.Tensor) or code.numel() == 0:
            return []
        return [int(index) for index in torch.nonzero(code > 0, as_tuple=False).flatten().tolist()]

    def _trace_exclusion_code_topk(self) -> int:
        return max(1, int(getattr(self.hparams, "trace_exclusion_code_topk", max(self._address_code_topk(), 8)) or max(self._address_code_topk(), 8)))

    def _support_overlap(self, left_support: list[int] | set[int], right_support: list[int] | set[int]) -> float:
        left = set(left_support)
        right = set(right_support)
        union = left | right
        if not union:
            return 0.0
        return float(len(left & right) / len(union))

    def _address_code_agreement(self, codes: list[torch.Tensor]) -> float | None:
        valid_codes = [code.detach().float().cpu().flatten() for code in codes if isinstance(code, torch.Tensor) and code.numel() > 0]
        if len(valid_codes) < 2:
            return 1.0 if valid_codes else None
        max_dim = max(int(code.numel()) for code in valid_codes)
        aligned_codes = []
        for code in valid_codes:
            if int(code.numel()) == max_dim:
                aligned_codes.append(code)
                continue
            padded = torch.zeros(max_dim, dtype=torch.float32)
            padded[: int(code.numel())] = code
            aligned_codes.append(padded)
        agreements = []
        for left_index in range(len(aligned_codes)):
            for right_index in range(left_index + 1, len(aligned_codes)):
                agreements.append(float(torch.dot(aligned_codes[left_index], aligned_codes[right_index]).item()))
        if not agreements:
            return None
        return float(sum(agreements) / len(agreements))

    def _address_dictionary_coherence(self, atoms: list[torch.Tensor]) -> float | None:
        valid_atoms = [atom.detach().float().cpu() for atom in atoms if isinstance(atom, torch.Tensor) and atom.numel() > 0]
        if len(valid_atoms) < 2:
            return 0.0 if valid_atoms else None
        atom_matrix = torch.stack(valid_atoms, dim=0)
        gram = atom_matrix @ atom_matrix.T
        values = []
        for left_index in range(gram.shape[0]):
            for right_index in range(left_index + 1, gram.shape[0]):
                values.append(abs(float(gram[left_index, right_index].item())))
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _sparsify_code_vector(self, raw_code: torch.Tensor | None, *, topk: int | None = None) -> torch.Tensor:
        if not isinstance(raw_code, torch.Tensor):
            return torch.zeros(0, dtype=torch.float32)
        code = raw_code.detach().float().cpu().clamp_min(0.0)
        if code.numel() == 0:
            return code
        limit = max(1, int(topk if topk is not None else self._address_code_topk()))
        limit = min(limit, code.numel())
        if limit < code.numel():
            top_values, top_indices = torch.topk(code, k=limit)
            sparse = torch.zeros_like(code)
            sparse[top_indices] = top_values
            code = sparse
        total = float(code.sum().item())
        if total <= 0.0:
            return torch.zeros_like(code)
        return code / total

    def _encode_sparse_address(self, address_vector: torch.Tensor | None, atoms: list[torch.Tensor]) -> torch.Tensor:
        if address_vector is None or not atoms:
            return torch.zeros(len(atoms), dtype=torch.float32)
        atom_matrix = torch.stack([atom.detach().float().cpu() for atom in atoms], dim=0)
        affinities = torch.mv(atom_matrix, address_vector.detach().float().cpu())
        if affinities.numel() == 0:
            return torch.zeros(len(atoms), dtype=torch.float32)

        gram = atom_matrix @ atom_matrix.T if atom_matrix.shape[0] > 1 else torch.eye(atom_matrix.shape[0], dtype=torch.float32)
        penalized = affinities.clone()
        selected: list[int] = []
        k = min(self._address_code_topk(), affinities.numel())
        min_affinity = self._address_code_min_affinity()
        coherence_penalty = self._address_coherence_penalty()
        for _ in range(k):
            if selected:
                coherence_cost = gram[:, selected].abs().max(dim=1).values
                penalized = affinities - coherence_penalty * coherence_cost
                penalized[selected] = float("-inf")
            next_index = int(torch.argmax(penalized).item())
            if penalized[next_index].item() <= min_affinity:
                break
            selected.append(next_index)
        code = torch.zeros_like(affinities, dtype=torch.float32)
        if not selected:
            return code
        weights = affinities[selected].clamp_min(min_affinity)
        code[selected] = weights
        return self._sparsify_code_vector(code, topk=k)

    def _score_sparse_address_view(
        self,
        query_vector: torch.Tensor | None,
        query_code: torch.Tensor,
        view: dict[str, Any],
    ) -> tuple[float | None, float | None]:
        view_vector = self._address_vector_from_view(view)
        if query_vector is None or not isinstance(view_vector, torch.Tensor):
            return None, None
        dense_score = float(torch.dot(query_vector.detach().float().cpu(), view_vector.detach().float().cpu()).item())
        code = view.get("address_code")
        sparse_score = None
        if isinstance(code, torch.Tensor) and code.numel() == query_code.numel():
            sparse_score = float(torch.dot(query_code.detach().float().cpu(), code.detach().float().cpu()).item())
        return dense_score, sparse_score

    def _trace_energy_from_scores(self, scores: list[float]) -> float | None:
        if not scores:
            return None
        view_tensor = torch.tensor(scores, dtype=torch.float32)
        beta = self._trace_energy_beta()
        temperature = self._trace_energy_temperature()
        return float((temperature * torch.logsumexp(view_tensor * beta, dim=0) / beta).item())

    def _entry_address_view_vectors(self, entry: dict[str, Any]) -> list[torch.Tensor]:
        vectors = []
        for view in self._entry_view_records(entry):
            vector = self._address_vector_from_view(view)
            if isinstance(vector, torch.Tensor):
                vectors.append(vector.detach().float().cpu())
        return vectors

    def _factored_trace_agreement(self, left: torch.Tensor | None, right: torch.Tensor | None) -> float:
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return 1.0
        return float(torch.dot(left.detach().float().cpu(), right.detach().float().cpu()).item())

    def _refresh_factored_relation_view_encodings(self) -> None:
        for entry in self.memory_entries:
            if not isinstance(entry, dict) or entry.get("edit_id") in self.disabled_adapters:
                continue
            for view in self._entry_view_records(entry):
                raw_factor = view.get("relation_raw_factor")
                if isinstance(raw_factor, torch.Tensor):
                    view["relation_factor"] = self._encode_factored_relation_factor(raw_factor)

    def _relation_encoder_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        group_ids: list[str],
        *,
        temperature: float,
    ) -> torch.Tensor | None:
        if embeddings.ndim != 2 or embeddings.shape[0] < 2:
            return None
        device = embeddings.device
        similarity = embeddings @ embeddings.T / max(1.0e-6, float(temperature))
        self_mask = torch.eye(embeddings.shape[0], dtype=torch.bool, device=device)
        similarity = similarity.masked_fill(self_mask, -1.0e9)
        losses = []
        for idx, group_id in enumerate(group_ids):
            if group_id is None:
                continue
            positive_mask = torch.tensor(
                [other_idx != idx and other_group == group_id for other_idx, other_group in enumerate(group_ids)],
                dtype=torch.bool,
                device=device,
            )
            if not bool(positive_mask.any().item()):
                continue
            log_denominator = torch.logsumexp(similarity[idx], dim=0)
            log_numerator = torch.logsumexp(similarity[idx][positive_mask], dim=0)
            losses.append(-(log_numerator - log_denominator))
        if not losses:
            return None
        return torch.stack(losses).mean()

    def _collect_relation_encoder_training_rows(self) -> tuple[torch.Tensor | None, list[str], list[str | None]]:
        raw_factors = []
        pair_ids = []
        relation_ids: list[str | None] = []
        for entry in self.memory_entries:
            if not isinstance(entry, dict) or entry.get("edit_id") in self.disabled_adapters:
                continue
            edit_id = str(entry.get("edit_id"))
            relation_id = entry.get("relation_id")
            relation_id = None if relation_id is None else str(relation_id)
            for view in self._entry_view_records(entry):
                raw_factor = view.get("relation_raw_factor")
                if isinstance(raw_factor, torch.Tensor):
                    raw_factors.append(raw_factor.detach().float().cpu())
                    pair_ids.append(edit_id)
                    relation_ids.append(relation_id)
        if not raw_factors:
            return None, [], []
        return torch.stack(raw_factors, dim=0), pair_ids, relation_ids

    def _maybe_train_factored_relation_encoder(self) -> bool:
        if not self._factored_relation_encoder_enabled():
            return False
        if (
            self.factored_relation_encoder_checkpoint_loaded
            and bool(getattr(self.hparams, "factored_relation_encoder_freeze_checkpoint", True))
        ):
            return False
        steps = int(getattr(self.hparams, "factored_relation_encoder_steps", 0) or 0)
        if steps <= 0:
            return False
        raw_factors, pair_ids, relation_ids = self._collect_relation_encoder_training_rows()
        if raw_factors is None:
            return False
        min_examples = int(getattr(self.hparams, "factored_relation_encoder_min_examples", 4) or 4)
        if int(raw_factors.shape[0]) < min_examples:
            return False
        encoder = self._ensure_factored_relation_encoder(int(raw_factors.shape[1]))
        if encoder is None:
            return False
        encoder.train()
        optimizer = torch.optim.AdamW(
            encoder.parameters(),
            lr=float(getattr(self.hparams, "factored_relation_encoder_lr", 1.0e-3) or 1.0e-3),
            weight_decay=0.0,
        )
        temperature = float(getattr(self.hparams, "factored_relation_encoder_temperature", 0.05) or 0.05)
        relation_weight = float(getattr(self.hparams, "factored_relation_encoder_relation_weight", 0.25) or 0.25)
        last_loss = None
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            encoded = F.normalize(encoder(raw_factors), p=2, dim=-1)
            pair_loss = self._relation_encoder_contrastive_loss(encoded, pair_ids, temperature=temperature)
            relation_loss = self._relation_encoder_contrastive_loss(encoded, relation_ids, temperature=temperature)
            if pair_loss is None and relation_loss is None:
                return False
            loss = torch.zeros((), dtype=encoded.dtype)
            if pair_loss is not None:
                loss = loss + pair_loss
            if relation_loss is not None:
                loss = loss + relation_weight * relation_loss
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
        encoder.eval()
        self.factored_relation_encoder_updates += 1
        self.factored_relation_encoder_last_loss = last_loss
        self._refresh_factored_relation_view_encodings()
        return True

    def _assign_factored_trace_address_state(
        self,
        entry: dict[str, Any],
        atoms: list[torch.Tensor],
        usage_counts: list[int],
        *,
        update_atoms: bool,
        address_version: int,
    ) -> None:
        views = self._entry_view_records(entry)
        if not views:
            entry["trace_address_support"] = []
            return
        prompt_view = next((view for view in views if view.get("view_name") == "prompt"), views[0])
        alternate_view = next((view for view in views if view is not prompt_view), None)
        subject_factor = self._clone_to_cpu(prompt_view.get("subject_factor"))
        raw_relation_factor = self._clone_to_cpu(prompt_view.get("relation_factor"))
        alternate_subject = self._clone_to_cpu(alternate_view.get("subject_factor")) if alternate_view is not None else None
        raw_alternate_relation = self._clone_to_cpu(alternate_view.get("relation_factor")) if alternate_view is not None else None
        relation_storage_context = self._factored_relation_storage_context(exclude_edit_id=entry.get("edit_id"))
        relation_factor = self._apply_factored_relation_storage_transform(raw_relation_factor, relation_storage_context)
        alternate_relation = self._apply_factored_relation_storage_transform(raw_alternate_relation, relation_storage_context)
        combined_vector = self._factored_combined_address_vector(subject_factor, relation_factor)
        if update_atoms:
            self._append_overlap_address_atoms(combined_vector, atoms, usage_counts)
        code = self._encode_sparse_address(combined_vector, atoms)
        support = self._address_support_from_code(code)
        entry["trace_address_impl"] = "factored_subject_relation"
        entry["trace_address_support"] = support
        entry["trace_address_code_topk"] = self._address_code_topk()
        entry["trace_address_codebook_version"] = address_version
        entry["trace_subject_factor"] = subject_factor
        entry["trace_relation_factor_raw"] = raw_relation_factor
        entry["trace_relation_factor"] = relation_factor
        entry["trace_subject_anchor_factor"] = alternate_subject
        entry["trace_relation_anchor_factor_raw"] = raw_alternate_relation
        entry["trace_relation_anchor_factor"] = alternate_relation
        entry["trace_relation_storage_transform"] = relation_storage_context.get("name")
        entry["trace_relation_storage_transform_active"] = bool(relation_storage_context.get("active"))
        entry["trace_relation_storage_transform_rank"] = relation_storage_context.get("rank")
        entry["trace_relation_storage_transform_num_vectors"] = relation_storage_context.get("num_vectors")
        entry["trace_subject_agreement"] = self._factored_trace_agreement(subject_factor, alternate_subject)
        entry["trace_relation_agreement"] = self._factored_trace_agreement(relation_factor, alternate_relation)
        entry["trace_address_agreement"] = 0.5 * (
            float(entry["trace_subject_agreement"]) + float(entry["trace_relation_agreement"])
        )
        entry["trace_address_centroid_vector"] = combined_vector
        entry["trace_address_centroid_code"] = code
        entry["trace_positive_support"] = support
        entry["trace_factor_views"] = [
            {
                "view_name": prompt_view.get("view_name"),
                "text": prompt_view.get("text"),
                "subject_found": bool(prompt_view.get("subject_found")),
                "relation_token_count": int(prompt_view.get("relation_token_count") or 0),
            }
        ]
        if alternate_view is not None:
            entry["trace_factor_views"].append(
                {
                    "view_name": alternate_view.get("view_name"),
                    "text": alternate_view.get("text"),
                    "subject_found": bool(alternate_view.get("subject_found")),
                    "relation_token_count": int(alternate_view.get("relation_token_count") or 0),
                }
            )
        entry["trace_address"] = {
            "address_impl": "factored_subject_relation",
            "support": support,
            "code_topk": self._address_code_topk(),
            "dictionary_version": address_version,
            "subject_found": bool(prompt_view.get("subject_found")),
            "relation_token_count": int(prompt_view.get("relation_token_count") or 0),
        }

    def _rebuild_factored_trace_address_state(self) -> None:
        trace_entries = [
            entry
            for entry in self.memory_entries
            if isinstance(entry, dict) and entry.get("edit_id") not in self.disabled_adapters
        ]
        atoms: list[torch.Tensor] = []
        usage_counts: list[int] = []
        address_version = self.address_version + 1
        postings: dict[int, list[str]] = {}
        for entry in trace_entries:
            self._assign_factored_trace_address_state(
                entry,
                atoms,
                usage_counts,
                update_atoms=True,
                address_version=address_version,
            )
        for entry in trace_entries:
            for atom_index in entry.get("trace_address_support") or []:
                postings.setdefault(int(atom_index), []).append(entry["edit_id"])
        self.address_dictionary = {
            "atoms": [self._clone_to_cpu(atom) for atom in atoms],
            "usage_counts": [int(count) for count in usage_counts],
            "coherence_mean": self._address_dictionary_coherence(atoms),
            "encoder_impl": "deterministic_topk",
            "build_entry_count": len(trace_entries),
            "code_topk": self._address_code_topk(),
            "merge_threshold": self._address_atom_merge_threshold(),
            "address_impl": "factored_subject_relation",
        }
        self.address_postings = {
            int(atom_index): sorted(set(trace_ids))
            for atom_index, trace_ids in postings.items()
        }
        self.family_postings = {}
        self.address_version = address_version

    def _index_factored_trace_entry(self, entry: dict[str, Any]) -> None:
        atoms = [atom.detach().float().cpu() for atom in self.address_dictionary.get("atoms", []) if isinstance(atom, torch.Tensor)]
        usage_counts = [int(count) for count in self.address_dictionary.get("usage_counts") or []]
        address_version = self.address_version + 1
        self._assign_factored_trace_address_state(
            entry,
            atoms,
            usage_counts,
            update_atoms=True,
            address_version=address_version,
        )
        self.address_dictionary = {
            "atoms": [self._clone_to_cpu(atom) for atom in atoms],
            "usage_counts": [int(count) for count in usage_counts],
            "coherence_mean": self._address_dictionary_coherence(atoms),
            "encoder_impl": "deterministic_topk",
            "build_entry_count": len(
                [
                    item
                    for item in self.memory_entries
                    if isinstance(item, dict) and item.get("edit_id") not in self.disabled_adapters
                ]
            ),
            "code_topk": self._address_code_topk(),
            "merge_threshold": self._address_atom_merge_threshold(),
            "address_impl": "factored_subject_relation",
        }
        for atom_index in entry.get("trace_address_support") or []:
            postings = self.address_postings.setdefault(int(atom_index), [])
            if entry["edit_id"] not in postings:
                postings.append(entry["edit_id"])
                postings.sort()
        self.address_version = address_version

    def _factored_relation_score_transform(self) -> str:
        return str(getattr(self.hparams, "factored_relation_score_transform", "identity") or "identity").strip().lower()

    def _factored_relation_storage_transform(self) -> str:
        return str(getattr(self.hparams, "factored_relation_storage_transform", "identity") or "identity").strip().lower()

    def _factored_capsule_enabled(self) -> bool:
        return bool(getattr(self.hparams, "factored_capsule_enable", False))

    def _load_factored_capsule_config(self) -> dict[str, Any]:
        path = getattr(self.hparams, "factored_capsule_config_path", None)
        if path:
            path = str(path)
            if self.factored_capsule_config is None or self.factored_capsule_config_loaded_from != path:
                with open(path, "r") as handle:
                    payload = json.load(handle)
                self.factored_capsule_config = payload if isinstance(payload, dict) else {}
                self.factored_capsule_config_loaded_from = path
            config = dict(self.factored_capsule_config or {})
        else:
            config = {}
        config.setdefault("score_family", getattr(self.hparams, "factored_capsule_score_family", "min_z"))
        if getattr(self.hparams, "factored_capsule_theta_accept", None) is not None:
            config["theta_accept"] = float(getattr(self.hparams, "factored_capsule_theta_accept"))
        config.setdefault("theta_accept", 0.0)
        config.setdefault("conflict_margin", float(getattr(self.hparams, "factored_capsule_conflict_margin", 0.0) or 0.0))
        if getattr(self.hparams, "factored_capsule_feature_weights", None) is not None:
            config["feature_weights"] = list(getattr(self.hparams, "factored_capsule_feature_weights") or [])
        if "feature_bias" not in config:
            config["feature_bias"] = float(getattr(self.hparams, "factored_capsule_feature_bias", 0.0) or 0.0)
        return config

    def _factored_capsule_row_zscores(self, scores: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(float(eps))
        return (scores - mean) / std

    def _factored_capsule_margin_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if int(scores.shape[1]) <= 1:
            return scores
        top2_values, top2_indices = torch.topk(scores, k=2, dim=1)
        top1_values = top2_values[:, 0:1]
        top2_values_only = top2_values[:, 1:2]
        top1_indices = top2_indices[:, 0:1]
        candidate_indices = torch.arange(scores.shape[1], device=scores.device).view(1, -1)
        runner = torch.where(candidate_indices == top1_indices, top2_values_only, top1_values)
        return scores - runner

    def _factored_capsule_rank_feature(self, scores: torch.Tensor) -> torch.Tensor:
        n = int(scores.shape[1])
        order = torch.argsort(scores, dim=1, descending=True)
        ranks = torch.empty_like(order, dtype=torch.float32)
        rank_values = torch.arange(1, n + 1, device=scores.device, dtype=torch.float32).view(1, -1)
        ranks.scatter_(1, order, rank_values.expand_as(order).float())
        if n <= 1:
            return torch.ones_like(ranks)
        return 1.0 - torch.log(ranks) / math.log(n + 1.0)

    def _factored_capsule_feature_tensor(self, subject_scores: torch.Tensor, relation_scores: torch.Tensor) -> torch.Tensor:
        subject_z = self._factored_capsule_row_zscores(subject_scores)
        relation_z = self._factored_capsule_row_zscores(relation_scores)
        subject_margin = self._factored_capsule_margin_scores(subject_z)
        relation_margin = self._factored_capsule_margin_scores(relation_z)
        subject_rank = self._factored_capsule_rank_feature(subject_z)
        relation_rank = self._factored_capsule_rank_feature(relation_z)
        subject_norm = subject_z / subject_z.max(dim=1, keepdim=True).values.clamp_min(1.0e-6)
        relation_norm = relation_z / relation_z.max(dim=1, keepdim=True).values.clamp_min(1.0e-6)
        return torch.stack(
            [
                subject_z,
                relation_z,
                subject_margin,
                relation_margin,
                subject_rank,
                relation_rank,
                subject_norm,
                relation_norm,
            ],
            dim=-1,
        )

    def _score_factored_capsules(
        self,
        *,
        candidate_entries: list[dict[str, Any]],
        subject_rows: list[dict[str, Any]],
        relation_rows: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not self._factored_capsule_enabled():
            return None
        config = self._load_factored_capsule_config()
        subject_by_id = {str(row["entry"].get("edit_id")): float(row["score"]) for row in subject_rows}
        relation_by_id = {str(row["entry"].get("edit_id")): float(row["score"]) for row in relation_rows}
        scored_entries = [
            entry
            for entry in candidate_entries
            if str(entry.get("edit_id")) in subject_by_id and str(entry.get("edit_id")) in relation_by_id
        ]
        if not scored_entries:
            return {
                "accepted": False,
                "chosen_entry": None,
                "abstain_reason": "no_scored_candidates",
                "config": config,
            }
        trace_ids = [str(entry.get("edit_id")) for entry in scored_entries]
        subject_scores = torch.tensor([[subject_by_id[trace_id] for trace_id in trace_ids]], dtype=torch.float32)
        relation_scores = torch.tensor([[relation_by_id[trace_id] for trace_id in trace_ids]], dtype=torch.float32)
        family = str(config.get("score_family") or "min_z").strip().lower()
        subject_z = self._factored_capsule_row_zscores(subject_scores)
        relation_z = self._factored_capsule_row_zscores(relation_scores)
        if family == "min_z":
            capsule_scores = torch.minimum(subject_z, relation_z)[0]
        elif family in {"margin_min_z", "min_margin_z"}:
            capsule_scores = torch.minimum(
                self._factored_capsule_margin_scores(subject_z),
                self._factored_capsule_margin_scores(relation_z),
            )[0]
        elif family in {"subject_margin", "subject_margin_z"}:
            capsule_scores = self._factored_capsule_margin_scores(subject_z)[0]
        elif family in {"relation_margin", "relation_margin_z"}:
            capsule_scores = self._factored_capsule_margin_scores(relation_z)[0]
        elif family == "learned_linear":
            features = self._factored_capsule_feature_tensor(subject_scores, relation_scores)[0]
            weights = torch.tensor(list(config.get("feature_weights") or []), dtype=torch.float32)
            if int(weights.numel()) != int(features.shape[-1]):
                return {
                    "accepted": False,
                    "chosen_entry": None,
                    "abstain_reason": "invalid_learned_linear_weights",
                    "config": config,
                }
            capsule_scores = features @ weights + float(config.get("feature_bias", 0.0) or 0.0)
        else:
            return {
                "accepted": False,
                "chosen_entry": None,
                "abstain_reason": f"unsupported_score_family:{family}",
                "config": config,
            }
        order = torch.argsort(capsule_scores, descending=True)
        top_idx = int(order[0].item())
        runner_idx = int(order[1].item()) if int(order.numel()) > 1 else None
        top_score = float(capsule_scores[top_idx].item())
        runner_score = None if runner_idx is None else float(capsule_scores[runner_idx].item())
        theta = float(config.get("theta_accept", 0.0) or 0.0)
        margin = float("inf") if runner_score is None else float(top_score - runner_score)
        conflict_margin = float(config.get("conflict_margin", 0.0) or 0.0)
        accepted = bool(top_score >= theta and margin >= conflict_margin)
        if top_score < theta:
            reason = "below_threshold"
        elif margin < conflict_margin:
            reason = "conflict_margin"
        else:
            reason = None
        return {
            "accepted": accepted,
            "chosen_entry": scored_entries[top_idx] if accepted else None,
            "top_entry": scored_entries[top_idx],
            "runner_entry": None if runner_idx is None else scored_entries[runner_idx],
            "top_score": top_score,
            "runner_score": runner_score,
            "margin": margin,
            "theta_accept": theta,
            "conflict_margin": conflict_margin,
            "abstain_reason": reason,
            "score_family": family,
            "num_candidates": int(len(scored_entries)),
            "config": config,
        }

    def _build_factored_relation_pc_projector(self, relation_by_trace: dict[str, torch.Tensor]) -> dict[str, Any] | None:
        rank = max(0, int(getattr(self.hparams, "factored_relation_streaming_pc_rank", 8) or 8))
        if rank <= 0:
            return None
        eps = float(getattr(self.hparams, "factored_relation_streaming_pc_eps", 1.0e-6) or 1.0e-6)
        min_traces = max(2, int(getattr(self.hparams, "factored_relation_streaming_pc_min_traces", 8) or 8))
        vectors = [vector.detach().double().cpu() for vector in relation_by_trace.values() if isinstance(vector, torch.Tensor)]
        if len(vectors) < min_traces:
            return None
        matrix = torch.stack(vectors, dim=0)
        try:
            _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        except RuntimeError:
            return None
        keep_indices = torch.nonzero(singular_values > eps, as_tuple=False).flatten()
        if keep_indices.numel() == 0:
            return None
        keep_indices = keep_indices[: min(rank, int(keep_indices.numel()))]
        basis = vh[keep_indices].T.contiguous()
        return {
            "basis": basis,
            "rank": int(basis.shape[1]),
            "num_vectors": int(matrix.shape[0]),
            "eps": eps,
        }

    def _apply_factored_relation_pc_projector(
        self,
        relation: torch.Tensor | None,
        projector: dict[str, Any] | None,
    ) -> torch.Tensor | None:
        if not isinstance(relation, torch.Tensor):
            return None
        if projector is None:
            return relation.detach().float().cpu()
        basis = projector.get("basis")
        if not isinstance(basis, torch.Tensor) or basis.numel() == 0:
            return relation.detach().float().cpu()
        relation = relation.detach().double().cpu()
        basis = basis.detach().double().cpu()
        projected = relation - (relation @ basis) @ basis.T
        norm = float(projected.norm().item())
        if norm <= 1.0e-12:
            return projected.float()
        return (projected / norm).float()

    def _build_factored_relation_whitener(self, relation_by_trace: dict[str, torch.Tensor]) -> dict[str, Any] | None:
        eps = float(getattr(self.hparams, "factored_relation_whiten_eps", 1.0e-4) or 1.0e-4)
        vectors = [vector.detach().double().cpu() for vector in relation_by_trace.values() if isinstance(vector, torch.Tensor)]
        min_traces = max(2, int(getattr(self.hparams, "factored_relation_whiten_min_traces", 2) or 2))
        if len(vectors) < min_traces:
            return None
        matrix = torch.stack(vectors, dim=0)
        mean = matrix.mean(dim=0)
        centered = matrix - mean
        try:
            _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        except RuntimeError:
            return None
        keep = singular_values > eps
        if not bool(keep.any().item()):
            return None
        basis = vh[keep].T.contiguous()
        scale = (float(max(1, matrix.shape[0] - 1)) ** 0.5) / (singular_values[keep] + eps)
        return {
            "mean": mean,
            "basis": basis,
            "scale": scale,
            "rank": int(keep.sum().item()),
            "num_vectors": int(matrix.shape[0]),
            "eps": eps,
        }

    def _apply_factored_relation_whitener(
        self,
        relation: torch.Tensor | None,
        whitener: dict[str, Any] | None,
    ) -> torch.Tensor | None:
        if not isinstance(relation, torch.Tensor):
            return None
        if whitener is None:
            return relation.detach().float().cpu()
        relation = relation.detach().double().cpu()
        whitened = ((relation - whitener["mean"]) @ whitener["basis"]) * whitener["scale"]
        norm = float(whitened.norm().item())
        if norm <= 1.0e-12:
            return whitened.float()
        return (whitened / norm).float()

    def _factored_relation_storage_context(self, *, exclude_edit_id: str | None = None) -> dict[str, Any]:
        transform = self._factored_relation_storage_transform()
        if transform not in {"streaming_pc_project", "streaming_decorrelate", "streaming_pc"}:
            return {"name": "identity", "active": False, "rank": None, "num_vectors": 0}
        relation_by_trace = {
            str(entry.get("edit_id")): entry.get("trace_relation_factor")
            for entry in self.memory_entries
            if isinstance(entry, dict)
            and entry.get("edit_id") not in self.disabled_adapters
            and str(entry.get("edit_id")) != str(exclude_edit_id)
            and isinstance(entry.get("trace_relation_factor"), torch.Tensor)
        }
        projector = self._build_factored_relation_pc_projector(relation_by_trace)
        if projector is None:
            return {
                "name": "streaming_pc_project",
                "active": False,
                "rank": None,
                "num_vectors": len(relation_by_trace),
            }
        return {
            "name": "streaming_pc_project",
            "active": True,
            "projector": projector,
            "rank": projector["rank"],
            "num_vectors": projector["num_vectors"],
        }

    def _apply_factored_relation_storage_transform(
        self,
        relation: torch.Tensor | None,
        context: dict[str, Any] | None,
    ) -> torch.Tensor | None:
        if not isinstance(relation, torch.Tensor):
            return None
        context = context or {}
        if context.get("name") == "streaming_pc_project" and context.get("active"):
            return self._apply_factored_relation_pc_projector(relation, context.get("projector"))
        return relation.detach().float().cpu()

    def _apply_factored_relation_score_transform(
        self,
        relation: torch.Tensor | None,
        context: dict[str, Any] | None,
    ) -> torch.Tensor | None:
        if not isinstance(relation, torch.Tensor):
            return None
        context = context or {}
        if context.get("name") == "global_whiten" and context.get("active"):
            return self._apply_factored_relation_whitener(relation, context.get("whitener"))
        if context.get("name") == "streaming_pc_project" and context.get("active"):
            return self._apply_factored_relation_pc_projector(relation, context.get("projector"))
        return relation.detach().float().cpu()

    def _factored_relation_score_context(self, candidate_entries: list[dict[str, Any]]) -> dict[str, Any]:
        transform = self._factored_relation_score_transform()
        if transform not in {"global_whiten", "whiten", "streaming_pc_project", "streaming_decorrelate", "streaming_pc"}:
            return {"name": "identity", "active": False, "trace_relation_by_id": {}}
        relation_by_trace = {
            str(entry.get("edit_id")): entry.get("trace_relation_factor")
            for entry in candidate_entries
            if isinstance(entry.get("trace_relation_factor"), torch.Tensor)
        }
        if transform in {"streaming_pc_project", "streaming_decorrelate", "streaming_pc"}:
            signature = (
                "streaming_pc_project",
                int(getattr(self.hparams, "factored_relation_streaming_pc_rank", 8) or 8),
                int(getattr(self.hparams, "factored_relation_streaming_pc_min_traces", 8) or 8),
                float(getattr(self.hparams, "factored_relation_streaming_pc_eps", 1.0e-6) or 1.0e-6),
                int(self.address_version),
                int(self.factored_relation_encoder_updates),
                tuple(str(entry.get("edit_id")) for entry in candidate_entries),
            )
            if self.factored_relation_score_transform_signature == signature:
                return self.factored_relation_score_transform_cache
            projector = self._build_factored_relation_pc_projector(relation_by_trace)
            if projector is None:
                context = {
                    "name": "streaming_pc_project",
                    "active": False,
                    "trace_relation_by_id": {},
                    "rank": None,
                    "num_vectors": len(relation_by_trace),
                }
            else:
                context = {
                    "name": "streaming_pc_project",
                    "active": True,
                    "projector": projector,
                    "trace_relation_by_id": {},
                    "rank": projector["rank"],
                    "num_vectors": projector["num_vectors"],
                }
            self.factored_relation_score_transform_signature = signature
            self.factored_relation_score_transform_cache = context
            return context
        signature = (
            "global_whiten",
            float(getattr(self.hparams, "factored_relation_whiten_eps", 1.0e-4) or 1.0e-4),
            int(getattr(self.hparams, "factored_relation_whiten_min_traces", 2) or 2),
            int(self.address_version),
            int(self.factored_relation_encoder_updates),
            tuple(str(entry.get("edit_id")) for entry in candidate_entries),
        )
        if self.factored_relation_score_transform_signature == signature:
            return self.factored_relation_score_transform_cache
        whitener = self._build_factored_relation_whitener(relation_by_trace)
        if whitener is None:
            context = {
                "name": "global_whiten",
                "active": False,
                "trace_relation_by_id": {},
                "rank": None,
                "num_vectors": len(relation_by_trace),
            }
        else:
            transformed_by_id = {
                trace_id: transformed
                for trace_id, vector in relation_by_trace.items()
                if (transformed := self._apply_factored_relation_whitener(vector, whitener)) is not None
            }
            context = {
                "name": "global_whiten",
                "active": True,
                "whitener": whitener,
                "trace_relation_by_id": transformed_by_id,
                "rank": whitener["rank"],
                "num_vectors": whitener["num_vectors"],
            }
        self.factored_relation_score_transform_signature = signature
        self.factored_relation_score_transform_cache = context
        return context

    def _route_from_factored_query(self, query: dict[str, Any]) -> dict[str, Any]:
        active_entries = [
            entry
            for entry in self.memory_entries
            if isinstance(entry, dict) and entry.get("edit_id") not in self.disabled_adapters
        ]
        # v4.3 is a correctness-first factorization probe: the final decision
        # must be made over the full live trace bank so a flat sparse shortlist
        # cannot remove the correct trace before the hard-AND fires.
        candidate_entries = active_entries
        candidate_set_size = len(candidate_entries)
        subject_factor = query.get("subject_factor")
        relation_factor = query.get("relation_factor")
        query_support = query.get("query_support") or []
        relation_score_context = self._factored_relation_score_context(candidate_entries)
        scored_relation_factor = self._apply_factored_relation_score_transform(relation_factor, relation_score_context)
        subject_rows = []
        relation_rows = []
        for entry in candidate_entries:
            trace_subject = entry.get("trace_subject_factor")
            if isinstance(subject_factor, torch.Tensor) and isinstance(trace_subject, torch.Tensor):
                subject_rows.append(
                    {
                        "entry": entry,
                        "score": float(torch.dot(subject_factor.detach().float().cpu(), trace_subject.detach().float().cpu()).item()),
                    }
                )
            trace_relation = entry.get("trace_relation_factor")
            if relation_score_context.get("name") == "global_whiten" and relation_score_context.get("active"):
                trace_relation = relation_score_context.get("trace_relation_by_id", {}).get(str(entry.get("edit_id")))
            if isinstance(scored_relation_factor, torch.Tensor) and isinstance(trace_relation, torch.Tensor):
                relation_rows.append(
                    {
                        "entry": entry,
                        "score": float(torch.dot(scored_relation_factor.detach().float().cpu(), trace_relation.detach().float().cpu()).item()),
                    }
                )
        subject_rows.sort(key=lambda row: row["score"], reverse=True)
        relation_rows.sort(key=lambda row: row["score"], reverse=True)
        subject_top = subject_rows[0] if subject_rows else None
        relation_top = relation_rows[0] if relation_rows else None
        subject_runner = subject_rows[1] if len(subject_rows) > 1 else None
        subject_margin = None if subject_top is None else float(subject_top["score"] - (subject_runner["score"] if subject_runner else 0.0))
        subject_energy = None if subject_top is None else float(-subject_top["score"])
        relation_match_rule = str(getattr(self.hparams, "factored_relation_match_rule", "top1_same_trace") or "top1_same_trace").strip().lower()
        relation_candidate_row = None
        relation_runner = relation_rows[1] if len(relation_rows) > 1 else None
        if relation_match_rule == "subject_candidate" and subject_top is not None:
            candidate_id = subject_top["entry"]["edit_id"]
            relation_candidate_row = next((row for row in relation_rows if row["entry"]["edit_id"] == candidate_id), None)
            candidate_relation_id = subject_top["entry"].get("relation_id")
            exclude_same_relation = bool(getattr(self.hparams, "factored_relation_exclude_same_relation_id_from_margin", True))
            runner_pool = []
            for row in relation_rows:
                if row["entry"]["edit_id"] == candidate_id:
                    continue
                if exclude_same_relation and candidate_relation_id is not None and row["entry"].get("relation_id") == candidate_relation_id:
                    continue
                runner_pool.append(row)
            relation_runner = runner_pool[0] if runner_pool else None
        relation_decision_row = relation_candidate_row if relation_candidate_row is not None else relation_top
        relation_margin = None if relation_decision_row is None else float(relation_decision_row["score"] - (relation_runner["score"] if relation_runner else 0.0))
        relation_energy = None if relation_decision_row is None else float(-relation_decision_row["score"])
        relation_candidate_score = None if relation_candidate_row is None else float(relation_candidate_row["score"])
        top1_same_trace = bool(
            subject_top is not None
            and relation_top is not None
            and subject_top["entry"]["edit_id"] == relation_top["entry"]["edit_id"]
        )
        same_trace = bool(top1_same_trace if relation_match_rule != "subject_candidate" else relation_candidate_row is not None)
        subject_pass = bool(
            subject_energy is not None
            and subject_margin is not None
            and subject_energy < self._factored_threshold("factored_subject_energy_threshold", 0.0)
            and subject_margin > self._factored_threshold("factored_subject_margin_threshold", 0.03)
        )
        relation_pass = bool(
            relation_energy is not None
            and relation_margin is not None
            and relation_energy < self._factored_threshold("factored_relation_energy_threshold", 0.0)
            and relation_margin > self._factored_threshold("factored_relation_margin_threshold", 0.03)
        )
        capsule_result = self._score_factored_capsules(
            candidate_entries=candidate_entries,
            subject_rows=subject_rows,
            relation_rows=relation_rows,
        )
        capsule_active = bool(capsule_result is not None)
        chosen_entry = subject_top["entry"] if same_trace and subject_pass and relation_pass else None
        if capsule_active:
            chosen_entry = capsule_result.get("chosen_entry")
        address_abstained = chosen_entry is None
        if address_abstained:
            self.base_only_fallback_count += 1
        fail_subject = not same_trace or not subject_pass
        fail_relation = not same_trace or not relation_pass
        if fail_subject and fail_relation:
            fail_partition = "both"
        elif fail_subject:
            fail_partition = "subject"
        elif fail_relation:
            fail_partition = "relation"
        else:
            fail_partition = "none"
        chosen_support_overlap = None
        if chosen_entry is not None:
            chosen_support_overlap = self._support_overlap(query_support, chosen_entry.get("trace_address_support") or [])
        route_margin = 0.0 if subject_margin is None or relation_margin is None else float(min(subject_margin, relation_margin))
        if capsule_active:
            route_margin = float(capsule_result.get("margin") or 0.0)
        top_scores = []
        if subject_top is not None:
            top_scores.append(float(subject_top["score"]))
        if relation_top is not None:
            top_scores.append(float(relation_top["score"]))
        route_stage = "trace_factored_capsule" if capsule_active else "trace_factored_address_hard_and"
        decision = {
            "memory_unit": "trace",
            "chosen_memory_id": None if chosen_entry is None else chosen_entry["edit_id"],
            "chosen_edit_id": None if chosen_entry is None else chosen_entry["edit_id"],
            "chosen_cell_id": None,
            "top_memory_ids": [
                row["entry"]["edit_id"]
                for row in [subject_top, relation_top]
                if row is not None
            ],
            "top_edit_ids": [
                row["entry"]["edit_id"]
                for row in [subject_top, relation_top]
                if row is not None
            ],
            "top_cell_ids": [],
            "top_scores": top_scores,
            "route_margin": route_margin,
            "top_view_names": ["subject_factor", "relation_factor"],
            "route_stage": route_stage,
            "adapter_name": None if chosen_entry is None else chosen_entry["edit_id"],
            "trace_energy": None if subject_top is None or relation_top is None else float(0.5 * (subject_top["score"] + relation_top["score"])),
            "trace_energy_margin": route_margin,
            "candidate_set_size": candidate_set_size,
            "family_shortlist_size": 0,
            "address_support_size": len(query_support),
            "address_support_overlap": chosen_support_overlap,
            "cross_view_code_agreement": None if chosen_entry is None else chosen_entry.get("trace_address_agreement"),
            "address_atom_coherence": self.address_dictionary.get("coherence_mean"),
            "address_abstained": address_abstained,
            "address_locality_vetoed": False,
            "base_only_fallback": address_abstained,
            "factor_subject_top_edit_id": None if subject_top is None else subject_top["entry"]["edit_id"],
            "factor_relation_top_edit_id": None if relation_top is None else relation_top["entry"]["edit_id"],
            "factor_subject_energy": subject_energy,
            "factor_relation_energy": relation_energy,
            "factor_subject_margin": subject_margin,
            "factor_relation_margin": relation_margin,
            "factor_subject_pass": subject_pass,
            "factor_relation_pass": relation_pass,
            "factor_same_trace": same_trace,
            "factor_top1_same_trace": top1_same_trace,
            "factor_relation_match_rule": relation_match_rule,
            "factor_relation_storage_transform": self._factored_relation_storage_transform(),
            "factor_relation_score_transform": relation_score_context.get("name"),
            "factor_relation_score_transform_active": bool(relation_score_context.get("active")),
            "factor_relation_whiten_rank": relation_score_context.get("rank"),
            "factor_relation_whiten_num_vectors": relation_score_context.get("num_vectors"),
            "factor_relation_candidate_score": relation_candidate_score,
            "factor_relation_candidate_margin": relation_margin if relation_candidate_row is not None else None,
            "factor_failure_partition": fail_partition,
            "resolved_subject": query.get("resolved_subject"),
            "query_subject_found": query.get("subject_found"),
            "query_relation_token_count": query.get("relation_token_count"),
        }
        if capsule_active:
            capsule_top = capsule_result.get("top_entry")
            capsule_runner = capsule_result.get("runner_entry")
            decision.update(
                {
                    "capsule_enabled": True,
                    "capsule_accepted": bool(capsule_result.get("accepted")),
                    "capsule_score_family": capsule_result.get("score_family"),
                    "capsule_theta_accept": capsule_result.get("theta_accept"),
                    "capsule_conflict_margin": capsule_result.get("conflict_margin"),
                    "capsule_top_edit_id": None if capsule_top is None else capsule_top.get("edit_id"),
                    "capsule_runner_edit_id": None if capsule_runner is None else capsule_runner.get("edit_id"),
                    "capsule_top_score": capsule_result.get("top_score"),
                    "capsule_runner_score": capsule_result.get("runner_score"),
                    "capsule_margin": capsule_result.get("margin"),
                    "capsule_abstain_reason": capsule_result.get("abstain_reason"),
                    "capsule_num_candidates": capsule_result.get("num_candidates"),
                    "capsule_config_path": getattr(self.hparams, "factored_capsule_config_path", None),
                }
            )
        if bool(getattr(self.hparams, "log_full_factor_scores", False)):
            trace_ids = [entry.get("edit_id") for entry in candidate_entries]
            subject_score_by_id = {
                row["entry"].get("edit_id"): float(row["score"])
                for row in subject_rows
            }
            relation_score_by_id = {
                row["entry"].get("edit_id"): float(row["score"])
                for row in relation_rows
            }
            decision.update(
                {
                    "factor_score_trace_ids": trace_ids,
                    "factor_score_relation_ids": [entry.get("relation_id") for entry in candidate_entries],
                    "factor_subject_scores": [subject_score_by_id.get(trace_id) for trace_id in trace_ids],
                    "factor_relation_scores": [relation_score_by_id.get(trace_id) for trace_id in trace_ids],
                    "factor_subject_margin_threshold": self._factored_threshold("factored_subject_margin_threshold", 0.03),
                    "factor_relation_margin_threshold": self._factored_threshold("factored_relation_margin_threshold", 0.03),
                    "factor_subject_energy_threshold": self._factored_threshold("factored_subject_energy_threshold", 0.0),
                    "factor_relation_energy_threshold": self._factored_threshold("factored_relation_energy_threshold", 0.0),
                }
            )
        return decision

    def _trace_anchor_records(self, entry: dict[str, Any]) -> list[dict[str, Any]]:
        anchors = entry.get("trace_anchor_views") or []
        if anchors:
            return anchors
        views = self._entry_view_records(entry)
        if not views:
            return []
        prompt_view = next((view for view in views if view.get("view_name") == "prompt"), views[0])
        alternate = next((view for view in views if view is not prompt_view), prompt_view)
        return [prompt_view, alternate]

    def _select_trace_anchor_records(self, entry: dict[str, Any]) -> list[dict[str, Any]]:
        views = self._entry_view_records(entry)
        if not views:
            entry["trace_anchor_views"] = []
            return []
        prompt_view = next((view for view in views if view.get("view_name") == "prompt"), views[0])
        best_alternate = None
        best_score = None
        prompt_vector = self._address_vector_from_view(prompt_view)
        for view in views:
            if view is prompt_view:
                continue
            view_vector = self._address_vector_from_view(view)
            if not isinstance(prompt_vector, torch.Tensor) or not isinstance(view_vector, torch.Tensor):
                continue
            score = float(torch.dot(prompt_vector, view_vector).item())
            if best_score is None or score > best_score:
                best_score = score
                best_alternate = view
        anchors = [prompt_view]
        if best_alternate is not None:
            anchors.append(best_alternate)
        elif len(views) > 1:
            anchors.append(views[1])
        entry["trace_anchor_views"] = [
            {
                "view_name": anchor.get("view_name"),
                "text": anchor.get("text"),
                "address_support": list(anchor.get("address_support") or []),
                "address_code": self._clone_to_cpu(anchor.get("address_code")),
                "address_vector": self._clone_to_cpu(self._address_vector_from_view(anchor)),
            }
            for anchor in anchors
        ]
        entry["trace_anchor_names"] = [anchor.get("view_name") for anchor in anchors]
        return entry["trace_anchor_views"]

    def _trace_anchor_energy(
        self,
        query_vector: torch.Tensor,
        entry: dict[str, Any],
        allowed_view_names: set[str] | None = None,
    ) -> float | None:
        anchor_scores = []
        for anchor in self._trace_anchor_records(entry):
            if allowed_view_names is not None and anchor.get("view_name") not in allowed_view_names:
                continue
            anchor_vector = anchor.get("address_vector")
            if not isinstance(anchor_vector, torch.Tensor):
                anchor_vector = self._address_vector_from_view(anchor)
            if not isinstance(anchor_vector, torch.Tensor):
                continue
            anchor_scores.append(float(torch.dot(query_vector.detach().float().cpu(), anchor_vector.detach().float().cpu()).item()))
        return self._trace_energy_from_scores(anchor_scores)

    def _family_entries_for_entry(self, entry: dict[str, Any]) -> list[dict[str, Any]]:
        family_trace_ids = set()
        for family_id in entry.get("trace_family_ids") or []:
            for trace_id in self.family_postings.get(family_id, []):
                if trace_id != entry.get("edit_id"):
                    family_trace_ids.add(trace_id)
        family_entries = []
        for trace_id in family_trace_ids:
            other = self.edit_registry.get(trace_id)
            if other is not None and other.get("edit_id") not in self.disabled_adapters:
                family_entries.append(other)
        return family_entries

    def _current_overlap_neighbor_rows(
        self,
        entry: dict[str, Any],
        trace_entries: list[dict[str, Any]],
        *,
        topk: int,
    ) -> list[dict[str, Any]]:
        own_semantic = entry.get("semantic_key")
        own_activation = entry.get("activation_key")
        own_centroid = entry.get("trace_address_centroid_vector")
        if (
            not isinstance(own_semantic, torch.Tensor)
            or not isinstance(own_activation, torch.Tensor)
            or not isinstance(own_centroid, torch.Tensor)
        ):
            return []
        ranked: list[dict[str, Any]] = []
        for other in trace_entries:
            if other.get("edit_id") == entry.get("edit_id") or other.get("edit_id") in self.disabled_adapters:
                continue
            other_semantic = other.get("semantic_key")
            other_activation = other.get("activation_key")
            other_centroid = other.get("trace_address_centroid_vector")
            if (
                not isinstance(other_semantic, torch.Tensor)
                or not isinstance(other_activation, torch.Tensor)
                or not isinstance(other_centroid, torch.Tensor)
            ):
                continue
            semantic_conflict = float(torch.dot(own_semantic.detach().float().cpu(), other_semantic.detach().float().cpu()).item())
            activation_conflict = float(torch.dot(own_activation.detach().float().cpu(), other_activation.detach().float().cpu()).item())
            combined_conflict = float(torch.dot(own_centroid.detach().float().cpu(), other_centroid.detach().float().cpu()).item())
            best_view_name = None
            view_records = self._entry_view_records(other)
            if view_records:
                best_view_name = view_records[0].get("view_name")
            ranked.append(
                {
                    "edit_id": other["edit_id"],
                    "combined_conflict": combined_conflict,
                    "semantic_conflict": semantic_conflict,
                    "activation_conflict": activation_conflict,
                    "best_view_name": best_view_name,
                }
            )
        ranked.sort(key=lambda row: row["combined_conflict"], reverse=True)
        return ranked[:topk]

    def _current_overlap_neighbor_entries(
        self,
        entry: dict[str, Any],
        trace_entries: list[dict[str, Any]],
        *,
        topk: int,
    ) -> list[dict[str, Any]]:
        neighbors = []
        for row in self._current_overlap_neighbor_rows(entry, trace_entries, topk=topk):
            other = self.edit_registry.get(row.get("edit_id"))
            if other is not None and other.get("edit_id") not in self.disabled_adapters:
                neighbors.append(other)
        return neighbors

    def _trace_negative_sets(
        self,
        entry: dict[str, Any],
        trace_entries: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        family_entries = []
        seen_ids = set()
        for other in self._family_entries_for_entry(entry):
            if other["edit_id"] in seen_ids:
                continue
            family_entries.append(other)
            seen_ids.add(other["edit_id"])
        for other in self._current_overlap_neighbor_entries(
            entry,
            trace_entries,
            topk=max(self._trace_family_negative_topk(), int(getattr(self.hparams, "top_k", 4) or 4)),
        ):
            if other["edit_id"] in seen_ids:
                continue
            family_entries.append(other)
            seen_ids.add(other["edit_id"])
            if len(family_entries) >= self._trace_family_negative_topk():
                break

        own_family_ids = set(entry.get("trace_family_ids") or [])
        irrelevant: list[tuple[float, dict[str, Any]]] = []
        own_centroid = entry.get("trace_address_centroid_vector")
        for other in trace_entries:
            if other["edit_id"] == entry["edit_id"] or other["edit_id"] in seen_ids:
                continue
            if own_family_ids & set(other.get("trace_family_ids") or []):
                continue
            other_centroid = other.get("trace_address_centroid_vector")
            if not isinstance(own_centroid, torch.Tensor) or not isinstance(other_centroid, torch.Tensor):
                continue
            similarity = float(torch.dot(own_centroid.detach().float().cpu(), other_centroid.detach().float().cpu()).item())
            irrelevant.append((similarity, other))
        irrelevant.sort(key=lambda item: item[0], reverse=True)
        irrelevant_entries = [other for _, other in irrelevant[: self._trace_irrelevant_negative_topk()]]
        return family_entries[: self._trace_family_negative_topk()], irrelevant_entries

    def _trace_negative_energy(
        self,
        query_vector: torch.Tensor,
        negative_entries: list[dict[str, Any]],
    ) -> float | None:
        if not negative_entries:
            return None
        energies = []
        for other in negative_entries:
            other_vectors = self._entry_address_view_vectors(other)
            if not other_vectors:
                continue
            scores = [float(torch.dot(query_vector.detach().float().cpu(), other_vector.detach().float().cpu()).item()) for other_vector in other_vectors]
            energy = self._trace_energy_from_scores(scores)
            if energy is not None:
                energies.append(energy)
        if not energies:
            return None
        return float(max(energies))

    def _family_support_from_negatives(self, negative_entries: list[dict[str, Any]]) -> tuple[torch.Tensor | None, list[int]]:
        code_dim = len(self.address_dictionary.get("atoms") or [])
        if code_dim <= 0 or not negative_entries:
            return None, []
        aggregate = torch.zeros(code_dim, dtype=torch.float32)
        for other in negative_entries[: self._trace_family_negative_topk()]:
            centroid_code = other.get("trace_address_centroid_code")
            if isinstance(centroid_code, torch.Tensor) and centroid_code.numel() == code_dim:
                aggregate += centroid_code.detach().float().cpu()
        if float(aggregate.sum().item()) <= 0.0:
            return None, []
        family_code = self._sparsify_code_vector(aggregate, topk=max(self._address_code_topk(), self._trace_family_negative_topk()))
        family_support = self._address_support_from_code(family_code)
        return family_code, family_support

    def _calibrate_overlap_anchor_trace_entry(
        self,
        entry: dict[str, Any],
        trace_entries: list[dict[str, Any]],
    ) -> None:
        own_vectors = self._entry_address_view_vectors(entry)
        anchors = self._trace_anchor_records(entry)
        if not own_vectors or not anchors:
            entry["trace_family_negative_trace_ids"] = []
            entry["trace_irrelevant_negative_trace_ids"] = []
            entry["trace_family_support"] = []
            entry["trace_family_code"] = None
            entry["trace_min_activation_energy"] = None
            entry["trace_min_activation_margin"] = None
            entry["trace_min_anchor_energy"] = None
            entry["trace_min_family_margin"] = None
            entry["trace_family_negative_energy_ceiling"] = None
            entry["trace_irrelevant_negative_energy_ceiling"] = None
            return

        family_entries, irrelevant_entries = self._trace_negative_sets(entry, trace_entries)
        family_code, family_support = self._family_support_from_negatives(family_entries)
        positive_energies: list[float] = []
        positive_anchor_energies: list[float] = []
        family_margins: list[float] = []
        irrelevant_margins: list[float] = []
        family_negative_energies: list[float] = []
        irrelevant_negative_energies: list[float] = []

        for query_index, query_vector in enumerate(own_vectors):
            own_scores = [
                float(torch.dot(query_vector, candidate_vector).item())
                for candidate_index, candidate_vector in enumerate(own_vectors)
                if len(own_vectors) == 1 or candidate_index != query_index
            ]
            own_energy = self._trace_energy_from_scores(own_scores)
            if own_energy is None:
                continue
            positive_energies.append(own_energy)
            anchor_energy = self._trace_anchor_energy(query_vector, entry)
            if anchor_energy is not None:
                positive_anchor_energies.append(anchor_energy)
            family_energy = self._trace_negative_energy(query_vector, family_entries)
            irrelevant_energy = self._trace_negative_energy(query_vector, irrelevant_entries)
            if family_energy is not None:
                family_negative_energies.append(family_energy)
                family_margins.append(own_energy - family_energy)
            if irrelevant_energy is not None:
                irrelevant_negative_energies.append(irrelevant_energy)
                irrelevant_margins.append(own_energy - irrelevant_energy)

        positive_floor = None if not positive_energies else float(min(positive_energies))
        margin_values = family_margins + irrelevant_margins
        margin_floor = None if not margin_values else float(min(margin_values))
        anchor_floor = None if not positive_anchor_energies else float(min(positive_anchor_energies))
        family_margin_floor = None if not family_margins else float(min(family_margins))
        agreement = float(entry.get("trace_address_agreement") or 1.0)
        width = float(max(0.0, 1.0 - agreement))
        family_negative_ceiling = None if not family_negative_energies else float(max(family_negative_energies))
        irrelevant_negative_ceiling = None if not irrelevant_negative_energies else float(max(irrelevant_negative_energies))
        entry["trace_min_activation_energy"] = (
            None
            if positive_floor is None
            else float(max(0.0, positive_floor - getattr(self.hparams, "trace_shape_energy_relax_scale", 0.35) * width))
        )
        entry["trace_min_activation_margin"] = (
            None
            if margin_floor is None
            else float(max(self._trace_abstain_margin(), self._trace_family_margin_scale() * max(0.0, margin_floor)))
        )
        entry["trace_min_anchor_energy"] = (
            None
            if anchor_floor is None
            else float(max(self._trace_anchor_energy_floor(), self._trace_anchor_margin_scale() * max(0.0, anchor_floor)))
        )
        entry["trace_min_family_margin"] = (
            None
            if family_margin_floor is None
            else float(max(0.0, self._trace_family_margin_scale() * max(0.0, family_margin_floor)))
        )
        entry["trace_family_negative_energy_ceiling"] = family_negative_ceiling
        entry["trace_irrelevant_negative_energy_ceiling"] = irrelevant_negative_ceiling
        entry["trace_family_support"] = family_support
        entry["trace_family_code"] = family_code
        entry["trace_family_negative_trace_ids"] = [other["edit_id"] for other in family_entries]
        entry["trace_irrelevant_negative_trace_ids"] = [other["edit_id"] for other in irrelevant_entries]
        entry["trace_family_size"] = len(set().union(*(set(other.get("trace_family_ids") or []) for other in family_entries))) if family_entries else 0

    def _overlap_trace_rank_row(
        self,
        entry: dict[str, Any],
        query_vector: torch.Tensor,
        query_code: torch.Tensor,
        query_support: list[int],
        candidate_set_size: int,
        allowed_view_names: set[str] | None = None,
    ) -> dict[str, Any] | None:
        view_scores = []
        best_view_name = None
        best_view_score = None
        best_support_overlap = 0.0
        sparse_scores = []
        source_views = self._trace_anchor_records(entry) if allowed_view_names is not None else self._entry_view_records(entry)
        for view in source_views:
            if allowed_view_names is not None and not self._trace_anchor_records(entry) and view.get("view_name") not in allowed_view_names:
                continue
            dense_score, sparse_score = self._score_sparse_address_view(query_vector, query_code, view)
            if dense_score is None:
                continue
            view_scores.append(dense_score)
            if sparse_score is not None:
                sparse_scores.append(sparse_score)
            support_overlap = self._support_overlap(query_support, view.get("address_support") or [])
            if best_view_score is None or dense_score > best_view_score:
                best_view_score = dense_score
                best_view_name = view.get("view_name")
                best_support_overlap = support_overlap
        if not view_scores:
            return None
        trace_energy = self._trace_energy_from_scores(view_scores)
        if trace_energy is None:
            return None
        anchor_filter = None if allowed_view_names is not None else allowed_view_names
        anchor_energy = self._trace_anchor_energy(query_vector, entry, allowed_view_names=anchor_filter)
        family_negative_entries = [
            self.edit_registry.get(trace_id)
            for trace_id in entry.get("trace_family_negative_trace_ids") or []
            if self.edit_registry.get(trace_id) is not None
        ]
        family_negative_energy = self._trace_negative_energy(query_vector, family_negative_entries)
        family_margin = None if family_negative_energy is None else float(trace_energy - family_negative_energy)
        family_code = entry.get("trace_family_code")
        family_overlap_score = None
        family_overlap = None
        if isinstance(family_code, torch.Tensor) and family_code.numel() == query_code.numel():
            family_overlap_score = float(torch.dot(query_code.detach().float().cpu(), family_code.detach().float().cpu()).item())
            family_overlap = self._support_overlap(query_support, entry.get("trace_family_support") or [])
        combined_score = float(trace_energy + self._trace_anchor_weight() * float(anchor_energy or 0.0))
        return {
            "entry": entry,
            "combined_conflict": combined_score,
            "trace_energy": float(trace_energy),
            "trace_anchor_energy": None if anchor_energy is None else float(anchor_energy),
            "trace_family_negative_energy": None if family_negative_energy is None else float(family_negative_energy),
            "trace_family_margin": None if family_margin is None else float(family_margin),
            "family_overlap_score": family_overlap_score,
            "family_support_overlap": family_overlap,
            "best_view_name": best_view_name,
            "mean_sparse_score": None if not sparse_scores else float(sum(sparse_scores) / len(sparse_scores)),
            "address_support_overlap": best_support_overlap,
            "cross_view_code_agreement": entry.get("trace_address_agreement"),
            "candidate_set_size": candidate_set_size,
            "query_support_size": len(query_support),
            "family_count": len(entry.get("trace_family_ids") or []),
        }

    def _candidate_trace_ids_from_query_code_and_families(self, query_code: torch.Tensor) -> tuple[list[str], int]:
        support = self._address_support_from_code(query_code)
        if not support:
            return [
                entry["edit_id"]
                for entry in self.memory_entries
                if isinstance(entry, dict) and entry.get("edit_id") not in self.disabled_adapters
            ], 0
        candidate_scores: dict[str, float] = {}
        for atom_index in support:
            weight = float(query_code[atom_index].item())
            for trace_id in self.address_postings.get(int(atom_index), []):
                candidate_scores[trace_id] = candidate_scores.get(trace_id, 0.0) + weight
        if not candidate_scores:
            return [
                entry["edit_id"]
                for entry in self.memory_entries
                if isinstance(entry, dict) and entry.get("edit_id") not in self.disabled_adapters
            ], 0
        ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        family_scores: dict[str, float] = defaultdict(float)
        for trace_id, score in ranked[: self._address_candidate_budget()]:
            entry = self.edit_registry.get(trace_id)
            if entry is None:
                continue
            for family_id in entry.get("trace_family_ids") or []:
                family_scores[family_id] += float(score)
        top_families = sorted(family_scores.items(), key=lambda item: item[1], reverse=True)[: self._trace_family_budget()]
        expanded_scores = dict(candidate_scores)
        for family_id, family_score in top_families:
            for trace_id in self.family_postings.get(family_id, []):
                expanded_scores[trace_id] = expanded_scores.get(trace_id, 0.0) + self._trace_family_boost() * float(family_score)
        budget = self._address_candidate_budget()
        shortlisted = [trace_id for trace_id, _ in sorted(expanded_scores.items(), key=lambda item: item[1], reverse=True)[:budget]]
        return shortlisted, len(top_families)

    def _rank_overlap_anchor_trace_addresses(
        self,
        semantic_key: torch.Tensor,
        activation_key: torch.Tensor,
        allowed_view_names: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], torch.Tensor, list[int], int, int]:
        active_entries = [
            entry
            for entry in self.memory_entries
            if isinstance(entry, dict) and entry.get("edit_id") not in self.disabled_adapters
        ]
        atoms = [atom for atom in self.address_dictionary.get("atoms", []) if isinstance(atom, torch.Tensor)]
        if not active_entries or not atoms:
            return [], torch.zeros(len(atoms), dtype=torch.float32), [], 0, 0
        semantic_key = self._normalize_semantic_key(semantic_key)
        if self._use_overlap_aware_anchor_trace_bank():
            activation_key = self._normalize_activation_key(activation_key, None)
        else:
            activation_stats = self.cached_activation_stats
            if activation_stats is None:
                activation_stats = self._activation_stats_from_raw_keys(self._collect_raw_activation_keys(active_entries))
            activation_key = self._normalize_activation_key(activation_key, activation_stats)
        query_vector = self._address_vector_from_keys(semantic_key, activation_key)
        query_code = self._encode_sparse_address(query_vector, atoms)
        query_support = self._address_support_from_code(query_code)
        candidate_trace_ids, family_shortlist_size = self._candidate_trace_ids_from_query_code_and_families(query_code)
        candidate_set_size = len(candidate_trace_ids)
        candidate_entries = [
            self.edit_registry.get(trace_id)
            for trace_id in candidate_trace_ids
            if self.edit_registry.get(trace_id) is not None
        ]
        ranking = []
        for entry in candidate_entries:
            row = self._overlap_trace_rank_row(
                entry,
                query_vector,
                query_code,
                query_support,
                candidate_set_size,
                allowed_view_names=allowed_view_names,
            )
            if row is not None:
                row["family_shortlist_size"] = family_shortlist_size
                ranking.append(row)
        ranking.sort(key=lambda row: row["combined_conflict"], reverse=True)
        return ranking, query_code, query_support, candidate_set_size, family_shortlist_size

    def _calibrate_sparse_trace_thresholds(
        self,
        trace_entries: list[dict[str, Any]],
        atom_bank: Optional[list[torch.Tensor]] = None,
    ) -> None:
        if self._use_overlap_aware_anchor_trace_bank():
            for entry in trace_entries:
                self._calibrate_overlap_anchor_trace_entry(entry, trace_entries)
            return
        if not bool(getattr(self.hparams, "trace_use_calibrated_thresholds", True)):
            for entry in trace_entries:
                entry["trace_min_activation_energy"] = None
                entry["trace_min_activation_margin"] = None
                entry["trace_min_locality_margin"] = None
                entry["trace_max_exclusion_score"] = None
                entry["trace_negative_energy_ceiling"] = None
                entry["trace_positive_energy_floor"] = None
                entry["trace_positive_locality_margin_floor"] = None
                entry["trace_positive_locality_margin_mean"] = None
                entry["trace_positive_exclusion_score_ceiling"] = None
                entry["trace_negative_exclusion_score_floor"] = None
                entry["trace_exclusion_code"] = None
                entry["trace_exclusion_support"] = []
                entry["trace_exclusion_trace_ids"] = []
                entry["trace_negative_trace_ids"] = []
            return

        negative_topk = max(1, int(getattr(self.hparams, "trace_calibration_negative_topk", 8) or 8))
        energy_blend = float(getattr(self.hparams, "trace_calibration_energy_blend", 0.60) or 0.60)
        margin_scale = float(getattr(self.hparams, "trace_calibration_margin_scale", 0.50) or 0.50)
        min_margin = float(getattr(self.hparams, "trace_calibration_min_margin", 0.03) or 0.03)
        energy_relax_scale = float(getattr(self.hparams, "trace_shape_energy_relax_scale", 0.35) or 0.35)
        margin_relax_scale = float(getattr(self.hparams, "trace_shape_margin_relax_scale", 0.20) or 0.20)
        agreement_weight = float(getattr(self.hparams, "trace_shape_agreement_weight", 0.50) or 0.50)
        locality_margin_scale = float(getattr(self.hparams, "trace_locality_margin_scale", 0.50) or 0.50)
        locality_min_margin = float(getattr(self.hparams, "trace_locality_min_margin", 0.02) or 0.02)
        locality_relax_scale = float(getattr(self.hparams, "trace_locality_relax_scale", 0.10) or 0.10)
        exclusion_code_topk = max(1, int(getattr(self.hparams, "trace_exclusion_code_topk", 8) or 8))
        exclusion_threshold_blend = float(getattr(self.hparams, "trace_exclusion_threshold_blend", 0.50) or 0.50)
        exclusion_relax_scale = float(getattr(self.hparams, "trace_exclusion_relax_scale", 0.05) or 0.05)

        active_atoms = atom_bank
        if active_atoms is None:
            active_atoms = self.address_dictionary.get("atoms") or []

        centroid_vectors = {
            entry["edit_id"]: entry.get("trace_address_centroid_vector")
            for entry in trace_entries
            if isinstance(entry.get("trace_address_centroid_vector"), torch.Tensor)
        }
        centroid_codes = {
            entry["edit_id"]: entry.get("trace_address_centroid_code")
            for entry in trace_entries
            if isinstance(entry.get("trace_address_centroid_code"), torch.Tensor)
        }

        for entry in trace_entries:
            own_vectors = self._entry_address_view_vectors(entry)
            if not own_vectors:
                entry["trace_min_activation_energy"] = None
                entry["trace_min_activation_margin"] = None
                entry["trace_min_locality_margin"] = None
                entry["trace_max_exclusion_score"] = None
                entry["trace_negative_energy_ceiling"] = None
                entry["trace_positive_energy_floor"] = None
                entry["trace_positive_locality_margin_floor"] = None
                entry["trace_positive_locality_margin_mean"] = None
                entry["trace_positive_exclusion_score_ceiling"] = None
                entry["trace_negative_exclusion_score_floor"] = None
                entry["trace_exclusion_code"] = None
                entry["trace_exclusion_support"] = []
                entry["trace_exclusion_trace_ids"] = []
                entry["trace_negative_trace_ids"] = []
                continue

            own_centroid = entry.get("trace_address_centroid_vector")
            candidate_negatives: list[tuple[float, dict[str, Any]]] = []
            if isinstance(own_centroid, torch.Tensor):
                for other in trace_entries:
                    if other["edit_id"] == entry["edit_id"]:
                        continue
                    other_centroid = centroid_vectors.get(other["edit_id"])
                    if not isinstance(other_centroid, torch.Tensor):
                        continue
                    similarity = float(torch.dot(own_centroid.detach().float().cpu(), other_centroid.detach().float().cpu()).item())
                    candidate_negatives.append((similarity, other))
                candidate_negatives.sort(key=lambda item: item[0], reverse=True)
            negative_entries = [other for _, other in candidate_negatives[:negative_topk]]
            exclusion_code = None
            exclusion_support: list[int] = []
            if negative_entries:
                code_dim = len(active_atoms or [])
                if code_dim > 0:
                    exclusion_raw = torch.zeros(code_dim, dtype=torch.float32)
                    for similarity, other in candidate_negatives[:negative_topk]:
                        other_code = centroid_codes.get(other["edit_id"])
                        if not isinstance(other_code, torch.Tensor) or other_code.numel() != code_dim:
                            continue
                        weight = max(0.0, float(similarity))
                        if weight <= 0.0:
                            weight = 1.0
                        exclusion_raw += weight * other_code.detach().float().cpu()
                    exclusion_code = self._sparsify_code_vector(exclusion_raw, topk=exclusion_code_topk)
                    exclusion_support = self._address_support_from_code(exclusion_code)

            positive_energies: list[float] = []
            positive_margins: list[float] = []
            negative_activation_energies: list[float] = []
            positive_locality_margins: list[float] = []
            positive_exclusion_scores: list[float] = []
            negative_exclusion_scores: list[float] = []

            own_view_codes = [
                view.get("address_code")
                for view in self._entry_view_records(entry)
                if isinstance(view.get("address_code"), torch.Tensor)
            ]

            for query_index, query_vector in enumerate(own_vectors):
                own_scores = [
                    float(torch.dot(query_vector, candidate_vector).item())
                    for candidate_index, candidate_vector in enumerate(own_vectors)
                    if len(own_vectors) == 1 or candidate_index != query_index
                ]
                own_energy = self._trace_energy_from_scores(own_scores)
                if own_energy is None:
                    continue
                positive_energies.append(own_energy)

                competing_energies: list[float] = []
                for other in negative_entries:
                    other_vectors = self._entry_address_view_vectors(other)
                    if not other_vectors:
                        continue
                    other_scores = [float(torch.dot(query_vector, other_vector).item()) for other_vector in other_vectors]
                    other_energy = self._trace_energy_from_scores(other_scores)
                    if other_energy is not None:
                        competing_energies.append(other_energy)
                if isinstance(exclusion_code, torch.Tensor) and exclusion_code.numel() > 0:
                    query_code = None
                    if query_index < len(own_view_codes):
                        candidate_code = own_view_codes[query_index]
                        if isinstance(candidate_code, torch.Tensor) and candidate_code.numel() == exclusion_code.numel():
                            query_code = candidate_code.detach().float().cpu()
                    if query_code is None:
                        query_code = self._encode_sparse_address(query_vector, active_atoms or [])
                    exclusion_score = float(torch.dot(query_code, exclusion_code).item())
                    positive_exclusion_scores.append(exclusion_score)
                    negative_activation_energies.append(exclusion_score)

                runner_energy = max(competing_energies) if competing_energies else 0.0
                positive_margins.append(own_energy - runner_energy)

            if isinstance(exclusion_code, torch.Tensor) and exclusion_code.numel() > 0:
                for other in negative_entries:
                    other_code = centroid_codes.get(other["edit_id"])
                    if isinstance(other_code, torch.Tensor) and other_code.numel() == exclusion_code.numel():
                        negative_exclusion_scores.append(float(torch.dot(other_code.detach().float().cpu(), exclusion_code).item()))

            positive_floor = None if not positive_energies else float(min(positive_energies))
            positive_mean = None if not positive_energies else float(sum(positive_energies) / len(positive_energies))
            negative_ceiling = None if not negative_activation_energies else float(max(negative_activation_energies))
            calibrated_energy = None
            agreement = entry.get("trace_address_agreement")
            if agreement is None:
                agreement = 1.0
            width = 0.0
            if positive_floor is not None and positive_mean is not None:
                width += max(0.0, positive_mean - positive_floor)
            width += agreement_weight * max(0.0, 1.0 - float(agreement))
            if positive_floor is not None:
                calibrated_energy = positive_floor
                if negative_ceiling is not None:
                    if positive_floor > negative_ceiling:
                        calibrated_energy = float(negative_ceiling + energy_blend * (positive_floor - negative_ceiling))
                    else:
                        calibrated_energy = positive_floor
                calibrated_energy = float(max(0.0, calibrated_energy - energy_relax_scale * width))
            positive_margin_floor = None if not positive_margins else float(min(positive_margins))
            positive_margin_mean = None if not positive_margins else float(sum(positive_margins) / len(positive_margins))
            calibrated_margin = None
            if positive_margin_floor is not None:
                margin_width = 0.0
                if positive_margin_mean is not None:
                    margin_width += max(0.0, positive_margin_mean - positive_margin_floor)
                margin_width += agreement_weight * max(0.0, 1.0 - float(agreement))
                calibrated_margin = float(
                    max(
                        0.0,
                        max(min_margin, margin_scale * max(0.0, positive_margin_floor)) - margin_relax_scale * margin_width,
                    )
                )
            positive_locality_margin_floor = (
                None if not positive_locality_margins else float(min(positive_locality_margins))
            )
            positive_locality_margin_mean = (
                None if not positive_locality_margins else float(sum(positive_locality_margins) / len(positive_locality_margins))
            )
            positive_exclusion_ceiling = None if not positive_exclusion_scores else float(max(positive_exclusion_scores))
            negative_exclusion_floor = None if not negative_exclusion_scores else float(min(negative_exclusion_scores))
            calibrated_locality_margin = None
            if positive_exclusion_ceiling is not None:
                locality_width = 0.0
                positive_exclusion_mean = float(sum(positive_exclusion_scores) / len(positive_exclusion_scores))
                locality_width += max(0.0, positive_exclusion_ceiling - positive_exclusion_mean)
                locality_width += agreement_weight * max(0.0, 1.0 - float(agreement))
                calibrated_locality_margin = positive_exclusion_ceiling
                if negative_exclusion_floor is not None and negative_exclusion_floor > positive_exclusion_ceiling:
                    calibrated_locality_margin = float(
                        positive_exclusion_ceiling
                        + exclusion_threshold_blend * (negative_exclusion_floor - positive_exclusion_ceiling)
                    )
                calibrated_locality_margin = float(max(0.0, calibrated_locality_margin + exclusion_relax_scale * locality_width))

            entry["trace_min_activation_energy"] = calibrated_energy
            entry["trace_min_activation_margin"] = calibrated_margin
            entry["trace_min_locality_margin"] = calibrated_locality_margin
            entry["trace_max_exclusion_score"] = calibrated_locality_margin
            entry["trace_negative_energy_ceiling"] = negative_ceiling
            entry["trace_positive_energy_floor"] = positive_floor
            entry["trace_positive_energy_mean"] = positive_mean
            entry["trace_positive_margin_floor"] = positive_margin_floor
            entry["trace_positive_margin_mean"] = positive_margin_mean
            entry["trace_positive_locality_margin_floor"] = positive_locality_margin_floor
            entry["trace_positive_locality_margin_mean"] = positive_locality_margin_mean
            entry["trace_positive_exclusion_score_ceiling"] = positive_exclusion_ceiling
            entry["trace_negative_exclusion_score_floor"] = negative_exclusion_floor
            entry["trace_shape_width"] = width
            entry["trace_exclusion_code"] = exclusion_code
            entry["trace_exclusion_support"] = exclusion_support
            entry["trace_exclusion_trace_ids"] = [other["edit_id"] for other in negative_entries]
            entry["trace_negative_trace_ids"] = [other["edit_id"] for other in negative_entries]

    def _append_overlap_address_atoms(
        self,
        address_vector: torch.Tensor | None,
        atoms: list[torch.Tensor],
        usage_counts: list[int],
    ) -> None:
        if not isinstance(address_vector, torch.Tensor):
            return
        merge_threshold = self._address_atom_merge_threshold()
        if not atoms:
            atoms.append(address_vector.detach().float().cpu().clone())
            usage_counts.append(1)
            return
        atom_matrix = torch.stack(atoms, dim=0)
        similarities = torch.mv(atom_matrix, address_vector.detach().float().cpu())
        best_score, best_index = torch.max(similarities, dim=0)
        if float(best_score.item()) < merge_threshold and len(atoms) < self._address_num_atoms():
            atoms.append(address_vector.detach().float().cpu().clone())
            usage_counts.append(1)
            return
        usage_counts[int(best_index.item())] = int(usage_counts[int(best_index.item())]) + 1

    def _assign_overlap_trace_address_state(
        self,
        entry: dict[str, Any],
        atoms: list[torch.Tensor],
        usage_counts: list[int],
        *,
        update_atoms: bool,
        address_version: int,
    ) -> None:
        views = self._entry_view_records(entry)
        view_vectors: list[torch.Tensor | None] = []
        centroid_vectors = []
        view_names = []
        for view in views:
            address_vector = self._address_vector_from_view(view)
            if isinstance(address_vector, torch.Tensor):
                address_vector = address_vector.detach().float().cpu()
            view["address_vector"] = address_vector
            view_vectors.append(address_vector)
            if update_atoms:
                self._append_overlap_address_atoms(address_vector, atoms, usage_counts)
            if isinstance(address_vector, torch.Tensor):
                centroid_vectors.append(address_vector)
            view_names.append(view.get("view_name"))

        codes = []
        supports = []
        for view, address_vector in zip(views, view_vectors):
            code = self._encode_sparse_address(address_vector, atoms)
            support = self._address_support_from_code(code)
            view["address_code"] = code
            view["address_support"] = support
            codes.append(code)
            supports.append(set(support))
        union_support = sorted({index for support in supports for index in support})
        entry["trace_address_impl"] = "overlap_sparse_dictionary"
        entry["trace_address_support"] = union_support
        entry["trace_address_code_topk"] = self._address_code_topk()
        entry["trace_address_codebook_version"] = address_version
        entry["trace_address_agreement"] = self._address_code_agreement(codes)
        entry["trace_address_centroid_code"] = None if not codes else torch.stack(codes, dim=0).mean(dim=0)
        entry["trace_address_centroid_vector"] = (
            None if not centroid_vectors else self._l2_normalize(torch.stack(centroid_vectors, dim=0).mean(dim=0))
        )
        entry["trace_positive_support"] = union_support
        entry["trace_address"] = {
            "num_views": len(view_names),
            "view_names": view_names,
            "address_impl": "overlap_sparse_dictionary",
            "support": union_support,
            "code_topk": self._address_code_topk(),
            "dictionary_version": address_version,
        }
        self._assign_trace_family_ids(entry)
        self._select_trace_anchor_records(entry)

    def _recompute_overlap_trace_postings(self) -> None:
        postings: dict[int, list[str]] = {}
        family_postings: dict[str, list[str]] = defaultdict(list)
        for entry in self.memory_entries:
            if not isinstance(entry, dict) or entry.get("edit_id") in self.disabled_adapters:
                continue
            for atom_index in entry.get("trace_address_support") or []:
                postings.setdefault(int(atom_index), []).append(entry["edit_id"])
            for family_id in entry.get("trace_family_ids") or []:
                family_postings[family_id].append(entry["edit_id"])
        self.address_postings = {int(idx): sorted(set(trace_ids)) for idx, trace_ids in postings.items()}
        self.family_postings = {family_id: sorted(set(trace_ids)) for family_id, trace_ids in family_postings.items()}

    def _refresh_overlap_conflict_neighbors(self, affected_trace_ids: list[str]) -> None:
        active_entries = [
            entry
            for entry in self.memory_entries
            if isinstance(entry, dict) and entry.get("edit_id") not in self.disabled_adapters
        ]
        affected = sorted(set(affected_trace_ids))
        for trace_id in affected:
            entry = self.edit_registry.get(trace_id)
            if entry is None:
                continue
            entry["conflict_neighbors"] = self._current_overlap_neighbor_rows(
                entry,
                active_entries,
                topk=max(self._trace_family_negative_topk(), int(getattr(self.hparams, "top_k", 4) or 4)),
            )

    def _reverse_overlap_affected_trace_ids(self, entry: dict[str, Any]) -> list[str]:
        active_entries = [
            item
            for item in self.memory_entries
            if isinstance(item, dict) and item.get("edit_id") not in self.disabled_adapters
        ]
        new_trace_id = entry.get("edit_id")
        new_centroid = entry.get("trace_address_centroid_vector")
        new_family_ids = set(entry.get("trace_family_ids") or [])
        topk = max(self._trace_family_negative_topk(), int(getattr(self.hparams, "top_k", 4) or 4))
        affected: list[str] = []
        for other in active_entries:
            other_id = other.get("edit_id")
            if other_id is None or other_id == new_trace_id:
                continue
            other_family_ids = set(other.get("trace_family_ids") or [])
            if new_family_ids & other_family_ids:
                affected.append(other_id)
                continue
            other_centroid = other.get("trace_address_centroid_vector")
            if not isinstance(new_centroid, torch.Tensor) or not isinstance(other_centroid, torch.Tensor):
                continue
            similarity = float(torch.dot(new_centroid.detach().float().cpu(), other_centroid.detach().float().cpu()).item())
            current_neighbors = list(other.get("conflict_neighbors") or [])
            if len(current_neighbors) < topk:
                affected.append(other_id)
                continue
            current_cutoff = min(float(row.get("combined_conflict", float("-inf"))) for row in current_neighbors[:topk])
            if similarity >= current_cutoff:
                affected.append(other_id)
        return affected

    def _refresh_overlap_trace_neighbors(self, affected_trace_ids: list[str]) -> None:
        active_entries = [
            entry
            for entry in self.memory_entries
            if isinstance(entry, dict) and entry.get("edit_id") not in self.disabled_adapters
        ]
        affected = sorted(set(affected_trace_ids))
        for trace_id in affected:
            entry = self.edit_registry.get(trace_id)
            if entry is None:
                continue
            self._calibrate_overlap_anchor_trace_entry(entry, active_entries)

    def _rebuild_overlap_anchor_trace_state(self) -> None:
        trace_entries = [
            entry
            for entry in self.memory_entries
            if isinstance(entry, dict) and entry.get("edit_id") not in self.disabled_adapters
        ]
        atoms: list[torch.Tensor] = []
        usage_counts: list[int] = []
        address_version = self.address_version + 1
        for entry in trace_entries:
            self._assign_overlap_trace_address_state(
                entry,
                atoms,
                usage_counts,
                update_atoms=True,
                address_version=address_version,
            )
        self.address_dictionary = {
            "atoms": [self._clone_to_cpu(atom) for atom in atoms],
            "usage_counts": [int(count) for count in usage_counts],
            "coherence_mean": self._address_dictionary_coherence(atoms),
            "encoder_impl": "deterministic_topk",
            "build_entry_count": len(trace_entries),
            "code_topk": self._address_code_topk(),
            "merge_threshold": self._address_atom_merge_threshold(),
            "family_count": len({family_id for entry in trace_entries for family_id in entry.get("trace_family_ids") or []}),
        }
        self.address_version = address_version
        self._recompute_overlap_trace_postings()
        self._refresh_overlap_conflict_neighbors([entry["edit_id"] for entry in trace_entries])
        self._calibrate_sparse_trace_thresholds(trace_entries, atom_bank=atoms)

    def _index_overlap_anchor_trace_entry(self, entry: dict[str, Any]) -> None:
        atoms = [atom.detach().float().cpu() for atom in self.address_dictionary.get("atoms") or [] if isinstance(atom, torch.Tensor)]
        usage_counts = [int(count) for count in self.address_dictionary.get("usage_counts") or []]
        address_version = self.address_version + 1
        self._assign_overlap_trace_address_state(
            entry,
            atoms,
            usage_counts,
            update_atoms=True,
            address_version=address_version,
        )
        self.address_dictionary = {
            "atoms": [self._clone_to_cpu(atom) for atom in atoms],
            "usage_counts": [int(count) for count in usage_counts],
            "coherence_mean": self._address_dictionary_coherence(atoms),
            "encoder_impl": "deterministic_topk",
            "build_entry_count": len(
                [
                    item
                    for item in self.memory_entries
                    if isinstance(item, dict) and item.get("edit_id") not in self.disabled_adapters
                ]
            ),
            "code_topk": self._address_code_topk(),
            "merge_threshold": self._address_atom_merge_threshold(),
            "family_count": 0,
        }
        self.address_version = address_version
        self._recompute_overlap_trace_postings()
        self.address_dictionary["family_count"] = len(self.family_postings)
        affected_trace_ids = [entry["edit_id"]]
        for family_id in entry.get("trace_family_ids") or []:
            affected_trace_ids.extend(self.family_postings.get(family_id, []))
        affected_trace_ids.extend(self._reverse_overlap_affected_trace_ids(entry))
        self._refresh_overlap_conflict_neighbors(affected_trace_ids)
        self._refresh_overlap_trace_neighbors(affected_trace_ids)

    def _rebuild_sparse_trace_address_state(self) -> None:
        if not self._use_sparse_address_trace_bank():
            return
        if self._use_factored_address_trace_bank():
            self._rebuild_factored_trace_address_state()
            return
        if self._use_overlap_aware_anchor_trace_bank():
            self._rebuild_overlap_anchor_trace_state()
            return
        trace_entries = [
            entry
            for entry in self.memory_entries
            if isinstance(entry, dict) and entry.get("edit_id") not in self.disabled_adapters
        ]
        atoms: list[torch.Tensor] = []
        usage_counts: list[int] = []
        merge_threshold = self._address_atom_merge_threshold()
        for entry in trace_entries:
            for view in self._entry_view_records(entry):
                address_vector = self._address_vector_from_view(view)
                if address_vector is None:
                    continue
                if not atoms:
                    atoms.append(address_vector.clone())
                    usage_counts.append(1)
                    continue
                atom_matrix = torch.stack(atoms, dim=0)
                similarities = torch.mv(atom_matrix, address_vector)
                best_score, best_index = torch.max(similarities, dim=0)
                if float(best_score.item()) < merge_threshold and len(atoms) < self._address_num_atoms():
                    atoms.append(address_vector.clone())
                    usage_counts.append(1)
                else:
                    atom_index = int(best_index.item())
                    prev_count = usage_counts[atom_index]
                    updated = (
                        atoms[atom_index].detach().float().cpu() * float(prev_count)
                        + address_vector.detach().float().cpu()
                    ) / float(prev_count + 1)
                    atoms[atom_index] = self._l2_normalize(updated)
                    usage_counts[atom_index] = prev_count + 1

        address_version = self.address_version + 1
        coherence = self._address_dictionary_coherence(atoms)
        postings: dict[int, list[str]] = {}
        for entry in trace_entries:
            codes = []
            supports = []
            view_names = []
            centroid_vectors = []
            for view in self._entry_view_records(entry):
                address_vector = self._address_vector_from_view(view)
                if isinstance(address_vector, torch.Tensor):
                    address_vector = address_vector.detach().float().cpu()
                view["address_vector"] = address_vector
                code = self._encode_sparse_address(address_vector, atoms)
                support = self._address_support_from_code(code)
                view["address_code"] = code
                view["address_support"] = support
                view_names.append(view.get("view_name"))
                codes.append(code)
                supports.append(set(support))
                if isinstance(address_vector, torch.Tensor):
                    centroid_vectors.append(address_vector)
            union_support = sorted({index for support in supports for index in support})
            for atom_index in union_support:
                postings.setdefault(int(atom_index), []).append(entry["edit_id"])
            entry["trace_address_impl"] = "sparse_dictionary"
            entry["trace_address_support"] = union_support
            entry["trace_address_code_topk"] = self._address_code_topk()
            entry["trace_address_codebook_version"] = address_version
            entry["trace_address_agreement"] = self._address_code_agreement(codes)
            entry["trace_address_centroid_code"] = (
                None
                if not codes
                else torch.stack(codes, dim=0).mean(dim=0)
            )
            entry["trace_address_centroid_vector"] = (
                None
                if not centroid_vectors
                else self._l2_normalize(torch.stack(centroid_vectors, dim=0).mean(dim=0))
            )
            entry["trace_address"] = {
                "num_views": len(view_names),
                "view_names": view_names,
                "address_impl": "sparse_dictionary",
                "support": union_support,
                "code_topk": self._address_code_topk(),
                "dictionary_version": address_version,
            }

        self._calibrate_sparse_trace_thresholds(trace_entries, atom_bank=atoms)

        self.address_dictionary = {
            "atoms": [self._clone_to_cpu(atom) for atom in atoms],
            "usage_counts": [int(count) for count in usage_counts],
            "coherence_mean": coherence,
            "encoder_impl": getattr(self.hparams, "address_encoder_impl", "deterministic_topk"),
            "build_entry_count": len(trace_entries),
            "code_topk": self._address_code_topk(),
            "merge_threshold": merge_threshold,
        }
        self.address_postings = {
            int(atom_index): sorted(set(trace_ids))
            for atom_index, trace_ids in postings.items()
        }
        self.address_version = address_version

    def _candidate_trace_ids_from_query_code(self, query_code: torch.Tensor) -> list[str]:
        support = self._address_support_from_code(query_code)
        if not support:
            return [
                entry["edit_id"]
                for entry in self.memory_entries
                if isinstance(entry, dict) and entry.get("edit_id") not in self.disabled_adapters
            ]
        candidate_scores: dict[str, float] = {}
        for atom_index in support:
            weight = float(query_code[atom_index].item())
            for trace_id in self.address_postings.get(int(atom_index), []):
                candidate_scores[trace_id] = candidate_scores.get(trace_id, 0.0) + weight
        if not candidate_scores:
            return [
                entry["edit_id"]
                for entry in self.memory_entries
                if isinstance(entry, dict) and entry.get("edit_id") not in self.disabled_adapters
            ]
        ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        budget = self._address_candidate_budget()
        return [trace_id for trace_id, _ in ranked[:budget]]

    def _build_negative_views(self, entry: dict[str, Any]) -> list[str]:
        views = [entry.get("prompt")]
        if self.hparams.use_rephrase_prompt and entry.get("rephrase_prompt"):
            views.append(entry.get("rephrase_prompt"))
        if self.hparams.use_subject_prompt and entry.get("subject") and entry.get("prompt"):
            prompt = str(entry["prompt"])
            subject = str(entry["subject"])
            if subject not in prompt:
                views.append(f"{subject}. {prompt}")
        deduped = []
        seen = set()
        for view in views:
            if view is None:
                continue
            view = " ".join(str(view).split())
            if view and view not in seen:
                seen.add(view)
                deduped.append(view)
        return deduped

    def _select_negative_entries(self, conflict_ranking: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.hparams.negative_top_k <= 0 or self.hparams.negative_weight <= 0:
            return []
        selected = []
        for row in conflict_ranking:
            if row["combined_conflict"] < self.hparams.negative_min_conflict:
                continue
            selected.append(row["entry"])
            if len(selected) >= self.hparams.negative_top_k:
                break
        return selected

    def _contrastive_negative_loss(self, target: str, positive_loss: torch.Tensor, negative_entries: list[dict[str, Any]]) -> torch.Tensor | None:
        if not negative_entries or self.hparams.negative_weight <= 0:
            return None
        margin_terms = []
        positive_anchor = positive_loss.detach()
        for entry in negative_entries:
            for negative_view in self._build_negative_views(entry):
                batch = self._training_batch(negative_view, target)
                negative_loss = self.model(**batch).loss
                margin_terms.append(F.relu(self.hparams.negative_margin + positive_anchor - negative_loss))
        if not margin_terms:
            return None
        return torch.stack(margin_terms).mean()

    def _train_adapter(self, adapter_name: str, request: dict[str, Any], prompt: str, subject_prompt: str, negative_entries: list[dict[str, Any]]) -> None:
        trainable = self._configure_trainable_adapter(adapter_name)
        optimizer = torch.optim.Adam(trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        positive_views = self._build_training_views(request, prompt, subject_prompt)
        target = request["target_new"]

        self.model.train()
        for step_idx in range(self.hparams.num_steps):
            optimizer.zero_grad()
            positive_losses = []
            for view in positive_views:
                batch = self._training_batch(view, target)
                positive_losses.append(self.model(**batch).loss)
            positive_loss = torch.stack(positive_losses).mean()
            loss = positive_loss

            negative_loss = self._contrastive_negative_loss(target, positive_loss, negative_entries)
            if negative_loss is not None:
                loss = loss + self.hparams.negative_weight * negative_loss

            if self.hparams.log_training_loss and ((step_idx + 1) % max(1, self.hparams.loss_log_every) == 0):
                print(
                    json.dumps(
                        {
                            "event": "hopedit_train_step",
                            "adapter_name": adapter_name,
                            "case_id": request.get("case_id"),
                            "step": step_idx + 1,
                            "num_steps": self.hparams.num_steps,
                            "positive_loss": float(positive_loss.detach().item()),
                            "negative_loss": None if negative_loss is None else float(negative_loss.detach().item()),
                            "total_loss": float(loss.detach().item()),
                            "num_positive_views": len(positive_views),
                            "num_negative_entries": len(negative_entries),
                            "hopedit_mode": self.hparams.hopedit_mode,
                        }
                    ),
                    flush=True,
                )

            loss.backward()
            optimizer.step()

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _adapter_parameter_refs(self, adapter_name: str) -> dict[str, torch.nn.Parameter]:
        refs = {}
        for name, parameter in self.model.named_parameters():
            if "lora_" not in name or adapter_name not in name:
                continue
            canonical_name = name.replace(adapter_name, "__ADAPTER__", 1)
            refs[canonical_name] = parameter
        return refs

    def _capture_adapter_parameters(self, adapter_name: str) -> dict[str, torch.Tensor]:
        refs = self._adapter_parameter_refs(adapter_name)
        return {name: parameter.detach().cpu().clone() for name, parameter in refs.items()}

    def _load_adapter_parameters(self, adapter_name: str, weights: dict[str, torch.Tensor]) -> None:
        refs = self._adapter_parameter_refs(adapter_name)
        with torch.no_grad():
            for name, parameter in refs.items():
                if name not in weights:
                    continue
                parameter.copy_(weights[name].to(parameter.device, dtype=parameter.dtype))

    def _copy_adapter_parameters(self, source_adapter: str, target_adapter: str) -> None:
        self._load_adapter_parameters(target_adapter, self._capture_adapter_parameters(source_adapter))

    def _compute_merge_weights(self, cell: dict[str, Any], assignment: dict[str, Any]) -> tuple[float, float]:
        merge_rule = getattr(self.hparams, "cell_merge_rule", "weighted_delta_average")
        member_count = max(1, int(len(cell.get("member_edit_ids", []))))
        best_conflict = assignment.get("best_conflict")
        assignment_margin = assignment.get("assignment_margin")
        if merge_rule not in {"weighted_delta_average", "ties_sparse_average"}:
            return float(member_count), 1.0
        if best_conflict is None:
            incoming_weight = 1.0
        else:
            incoming_weight = max(0.10, 1.0 - float(best_conflict) + max(0.0, float(assignment_margin or 0.0)))
        return float(member_count), float(incoming_weight)

    def _trim_tensor_by_quantile(self, tensor: torch.Tensor, quantile: float) -> torch.Tensor:
        if quantile <= 0.0:
            return tensor
        flat = tensor.detach().float().abs().flatten()
        if flat.numel() == 0:
            return tensor
        threshold = torch.quantile(flat, min(max(quantile, 0.0), 1.0))
        trimmed = tensor.clone()
        trimmed[trimmed.abs() < threshold] = 0
        return trimmed

    def _ties_sparse_merge_tensor(
        self,
        target_tensor: torch.Tensor,
        source_tensor: torch.Tensor,
        existing_weight: float,
        incoming_weight: float,
        trim_quantile: float,
    ) -> torch.Tensor:
        target_trim = self._trim_tensor_by_quantile(target_tensor, trim_quantile)
        source_trim = self._trim_tensor_by_quantile(source_tensor, trim_quantile)
        elected_sign = torch.sign((existing_weight * target_trim) + (incoming_weight * source_trim))
        target_sign = torch.sign(target_trim)
        source_sign = torch.sign(source_trim)

        target_keep = (target_sign != 0) & ((elected_sign == 0) | (target_sign == elected_sign))
        source_keep = (source_sign != 0) & ((elected_sign == 0) | (source_sign == elected_sign))

        merged = target_tensor.clone()
        both_keep = target_keep & source_keep
        target_only = target_keep & ~source_keep
        source_only = source_keep & ~target_keep

        if both_keep.any():
            merged[both_keep] = (
                (existing_weight * target_trim[both_keep]) + (incoming_weight * source_trim[both_keep])
            ) / (existing_weight + incoming_weight)
        if source_only.any():
            merged[source_only] = source_trim[source_only]
        if target_only.any():
            merged[target_only] = target_trim[target_only]
        return merged

    def _merge_adapter_into_cell(self, source_adapter: str, target_cell: dict[str, Any], assignment: dict[str, Any]) -> None:
        target_adapter = target_cell["adapter_name"]
        target_weights = self._capture_adapter_parameters(target_adapter)
        source_weights = self._capture_adapter_parameters(source_adapter)
        existing_weight, incoming_weight = self._compute_merge_weights(target_cell, assignment)
        merge_rule = getattr(self.hparams, "cell_merge_rule", "weighted_delta_average")
        trim_quantile = float(getattr(self.hparams, "cell_merge_trim_quantile", 0.0) or 0.0)
        merged = {}
        for name, source_tensor in source_weights.items():
            target_tensor = target_weights.get(name)
            if target_tensor is None:
                merged[name] = source_tensor.clone()
                continue
            if merge_rule == "ties_sparse_average":
                merged[name] = self._ties_sparse_merge_tensor(
                    target_tensor,
                    source_tensor,
                    existing_weight,
                    incoming_weight,
                    trim_quantile,
                )
            else:
                merged[name] = ((existing_weight * target_tensor) + (incoming_weight * source_tensor)) / (existing_weight + incoming_weight)
        self._load_adapter_parameters(target_adapter, merged)

    def _delete_adapter(self, adapter_name: str) -> None:
        if hasattr(self.model, "delete_adapter"):
            try:
                self.model.delete_adapter(adapter_name)
                self._repair_active_adapter_reference()
                return
            except Exception:
                pass
        self.disabled_adapters.add(adapter_name)
        self._repair_active_adapter_reference()

    def _cell_conflict_for_new_edit(self, cell: dict[str, Any], semantic_key: torch.Tensor, activation_key: torch.Tensor) -> float:
        entries = self._cell_entries(cell["cell_id"])
        if not entries:
            return -1.0
        scores = []
        for entry in entries:
            scored = self._score_entry_views(entry, semantic_key, activation_key)
            if scored is not None:
                scores.append(scored["combined_conflict"])
        if not scores:
            return -1.0
        stat = getattr(self.hparams, "cell_conflict_stat", "mean")
        if stat == "max":
            return float(max(scores))
        if stat == "p90":
            score_tensor = torch.tensor(scores, dtype=torch.float32)
            return float(torch.quantile(score_tensor, 0.90).item())
        if stat == "topk_mean":
            top_k = min(4, len(scores))
            return float(sum(sorted(scores, reverse=True)[:top_k]) / top_k)
        return float(sum(scores) / len(scores))

    def _select_cell_assignment(self, semantic_key: torch.Tensor, activation_key: torch.Tensor) -> dict[str, Any]:
        if self._use_sparse_slots():
            return self._select_state_assignment(semantic_key, activation_key)
        active_cells = self._active_tier_cells()
        cell_budget = int(getattr(self.hparams, "cell_budget", 0) or 0)
        can_allocate = cell_budget <= 0 or len(self._active_cells()) < cell_budget
        policy = getattr(self.hparams, "cell_assignment_policy", "conflict_aware")

        scored = []
        for cell in active_cells:
            mean_conflict = self._cell_conflict_for_new_edit(cell, semantic_key, activation_key)
            scored.append(
                {
                    "cell": cell,
                    "mean_conflict": mean_conflict,
                    "member_count": len(cell.get("member_edit_ids", [])),
                }
            )
        scored.sort(key=lambda row: row["mean_conflict"])

        best_conflict = None if not scored else scored[0]["mean_conflict"]
        assignment_margin = None
        if len(scored) >= 2:
            assignment_margin = float(scored[1]["mean_conflict"] - scored[0]["mean_conflict"])
        threshold = float(getattr(self.hparams, "cell_conflict_threshold", 0.65))
        min_margin = float(getattr(self.hparams, "cell_min_assignment_margin", 0.0))
        should_allocate = not scored
        warmup_edits = int(getattr(self.hparams, "cell_warmup_edits", 0) or 0)
        if not should_allocate and can_allocate and warmup_edits > 0 and len(self.memory_entries) < warmup_edits:
            should_allocate = True
        if not should_allocate and can_allocate:
            if best_conflict is not None and best_conflict > threshold:
                should_allocate = True
            elif assignment_margin is not None and assignment_margin < min_margin:
                should_allocate = True

        if should_allocate:
            return {
                "action": "allocate",
                "policy": policy,
                "best_conflict": best_conflict,
                "assignment_margin": assignment_margin,
                "chosen_cell": None,
            }

        if policy == "random":
            chosen = random.choice(scored)
        elif policy == "count_balanced":
            chosen = min(scored, key=lambda row: (row["member_count"], row["mean_conflict"], row["cell"]["cell_id"]))
        else:
            chosen = scored[0]
        return {
            "action": "merge",
            "policy": policy,
            "best_conflict": chosen["mean_conflict"],
            "assignment_margin": assignment_margin,
            "chosen_cell": chosen["cell"],
        }

    def _select_state_assignment(self, semantic_key: torch.Tensor, activation_key: torch.Tensor) -> dict[str, Any]:
        active_states = [
            cell
            for cell in self._active_tier_cells()
            if len(self._state_slots(cell)) < self._slot_capacity()
        ]
        cell_budget = int(getattr(self.hparams, "cell_budget", 0) or 0)
        can_allocate = cell_budget <= 0 or len(self._active_cells()) < cell_budget
        policy = getattr(self.hparams, "cell_assignment_policy", "conflict_aware")

        semantic_key = self._normalize_semantic_key(semantic_key)
        activation_stats = self.cached_activation_stats
        if activation_stats is None:
            activation_stats = self._activation_stats_from_raw_keys(self._collect_raw_activation_keys())
        activation_key = self._normalize_activation_key(activation_key, activation_stats)

        scored = []
        for cell in active_states:
            scored_row = self._score_cell(cell, semantic_key, activation_key)
            if scored_row is None:
                continue
            scored.append(scored_row)
        scored.sort(key=lambda row: row["combined_score"], reverse=True)

        best_score = None if not scored else float(scored[0]["combined_score"])
        assignment_margin = None
        if len(scored) >= 2:
            assignment_margin = float(scored[0]["combined_score"] - scored[1]["combined_score"])

        threshold = float(getattr(self.hparams, "cell_conflict_threshold", 0.35) or 0.35)
        min_margin = float(getattr(self.hparams, "cell_min_assignment_margin", 0.0) or 0.0)
        should_allocate = not scored
        if not should_allocate and can_allocate:
            if best_score is None or best_score < threshold:
                should_allocate = True
            elif assignment_margin is not None and assignment_margin < min_margin:
                should_allocate = True

        if should_allocate:
            return {
                "action": "allocate",
                "policy": policy,
                "best_conflict": best_score,
                "assignment_margin": assignment_margin,
                "chosen_cell": None,
            }

        if policy == "random":
            chosen = random.choice(scored)
        elif policy == "count_balanced":
            chosen = min(scored, key=lambda row: (len(self._state_slots(row["cell"])), -row["combined_score"], row["cell"]["cell_id"]))
        else:
            chosen = scored[0]
        return {
            "action": "append_slot",
            "policy": policy,
            "best_conflict": float(chosen["combined_score"]),
            "assignment_margin": assignment_margin,
            "chosen_cell": chosen["cell"],
        }

    def _trial_merge_state_score(self, target_cell: dict[str, Any], source_cell: dict[str, Any]) -> dict[str, Any]:
        merged_entries = self._cell_entries(target_cell["cell_id"]) + self._cell_entries(source_cell["cell_id"])
        if not merged_entries:
            return {
                "merged_gate_score": None,
                "merged_prototype_dispersion": None,
                "merged_within_state_conflict": None,
                "locality_proxy": None,
                "accepted": False,
                "reject_reasons": ["empty_merge"],
            }
        _, prototype_stats = self._summarize_cell_view_prototypes(merged_entries)
        within_scores = []
        for i in range(len(merged_entries)):
            for j in range(i + 1, len(merged_entries)):
                scored = self._score_entry_views(merged_entries[i], merged_entries[j]["semantic_key"], merged_entries[j]["activation_key"])
                if scored is not None:
                    within_scores.append(scored["combined_conflict"])
        within_conflict = None if not within_scores else float(sum(within_scores) / len(within_scores))
        locality_proxy = 1.0 if len(merged_entries) <= 1 else max(0.0, min(1.0, 1.0 - float(within_conflict or 0.0)))
        gate_scores = [
            float(score)
            for score in [
                target_cell.get("state_gate_score_mean"),
                source_cell.get("state_gate_score_mean"),
            ]
            if score is not None
        ]
        merged_gate_score = None if not gate_scores else float(min(gate_scores))
        accepted = True
        reject_reasons: list[str] = []
        if self._state_gate_enabled() and (
            merged_gate_score is None
            or merged_gate_score < float(getattr(self.hparams, "state_gate_merge_threshold", 0.75) or 0.75)
        ):
            accepted = False
            reject_reasons.append("gate_score")
        if prototype_stats.get("prototype_dispersion") is None or prototype_stats.get("prototype_dispersion") > float(getattr(self.hparams, "stability_max_prototype_dispersion", 0.30)):
            accepted = False
            reject_reasons.append("prototype_dispersion")
        if within_conflict is not None and within_conflict > float(getattr(self.hparams, "stability_max_within_state_conflict", 0.30)):
            accepted = False
            reject_reasons.append("within_state_conflict")
        if locality_proxy < float(getattr(self.hparams, "stability_min_locality", 0.95)):
            accepted = False
            reject_reasons.append("locality_proxy")
        return {
            "merged_gate_score": merged_gate_score,
            "merged_prototype_dispersion": prototype_stats.get("prototype_dispersion"),
            "merged_within_state_conflict": within_conflict,
            "locality_proxy": locality_proxy,
            "accepted": accepted,
            "reject_reasons": reject_reasons,
        }

    def _merge_states(self, target_cell: dict[str, Any], source_cell: dict[str, Any]) -> None:
        if self._use_sparse_slots():
            raise RuntimeError("Sparse-slot HopEdit-v2.3 uses slot transfers instead of whole-state merges.")
        assignment = {
            "best_conflict": source_cell.get("within_cell_conflict_mean"),
            "assignment_margin": None,
        }
        self._merge_adapter_into_cell(source_cell["adapter_name"], target_cell, assignment)
        source_cell_id = source_cell["cell_id"]
        target_cell_id = target_cell["cell_id"]
        for entry in self.memory_entries:
            if entry.get("cell_id") == source_cell_id:
                entry["cell_id"] = target_cell_id
        target_members = target_cell.setdefault("member_edit_ids", [])
        for edit_id in source_cell.get("member_edit_ids", []):
            if edit_id not in target_members:
                target_members.append(edit_id)
        self._delete_adapter(source_cell["adapter_name"])
        self.cell_registry.pop(source_cell_id, None)

    def _trial_slot_transfer_score(self, target_cell: dict[str, Any], source_cell: dict[str, Any], slot: dict[str, Any]) -> dict[str, Any]:
        slot_dispersion = None if slot.get("slot_dispersion") is None else float(slot.get("slot_dispersion"))
        within_state_conflict = float(target_cell.get("within_cell_conflict_mean") or 0.0)
        cross_view_gap = float(source_cell.get("cross_view_route_gap") or 0.0)
        locality_proxy = min(
            float(target_cell.get("locality_proxy") or 1.0),
            float(source_cell.get("locality_proxy") or 1.0),
        )
        accepted = True
        reject_reasons: list[str] = []
        if cross_view_gap > 0.05:
            accepted = False
            reject_reasons.append("cross_view_route_gap")
        if slot_dispersion is not None and slot_dispersion > 0.25:
            accepted = False
            reject_reasons.append("slot_dispersion")
        if within_state_conflict > 0.30:
            accepted = False
            reject_reasons.append("within_state_conflict")
        if locality_proxy < 0.95:
            accepted = False
            reject_reasons.append("locality_proxy")
        return {
            "accepted": accepted,
            "reject_reasons": reject_reasons,
            "slot_dispersion": slot_dispersion,
            "within_state_conflict": within_state_conflict,
            "locality_proxy": locality_proxy,
        }

    def _transfer_slot_to_state(self, target_cell: dict[str, Any], source_cell: dict[str, Any], slot: dict[str, Any]) -> None:
        source_slots = self._state_slots(source_cell)
        source_cell["slots"] = [row for row in source_slots if row.get("slot_id") != slot.get("slot_id")]
        target_cell.setdefault("slots", []).append(slot)
        source_edit_id = slot.get("source_edit_id")
        for entry in self.memory_entries:
            if entry.get("edit_id") == source_edit_id:
                entry["cell_id"] = target_cell["cell_id"]
        target_cell.setdefault("member_edit_ids", [])
        if source_edit_id is not None and source_edit_id not in target_cell["member_edit_ids"]:
            target_cell["member_edit_ids"].append(source_edit_id)
        source_cell["member_edit_ids"] = [edit_id for edit_id in source_cell.get("member_edit_ids", []) if edit_id != source_edit_id]
        if not source_cell["slots"]:
            self.cell_registry.pop(source_cell["cell_id"], None)

    def maybe_consolidate_states(self) -> None:
        if not self.is_v2:
            return
        total_edits = len(self.memory_entries)
        warmup = int(getattr(self.hparams, "memory_tier_warmup_edits", 256) or 256)
        interval = int(getattr(self.hparams, "consolidation_interval_edits", 16) or 16)
        if total_edits < warmup or total_edits % max(1, interval) != 0:
            return
        max_pairs = int(getattr(self.hparams, "consolidation_max_pairs_per_pass", 16) or 16)
        if self._use_sparse_slots():
            stable_cells = sorted([cell for cell in self._consolidated_cells() if cell.get("is_stable")], key=lambda cell: cell["cell_id"])
            transfers_done = 0
            for target_cell in stable_cells:
                if transfers_done >= max_pairs or target_cell["cell_id"] not in self.cell_registry:
                    break
                bucket_id = target_cell.get("bucket_id")
                if bucket_id is None:
                    continue
                if len(self._state_slots(target_cell)) >= self._slot_capacity():
                    continue
                for source_cell in self._bucket_cells(bucket_id):
                    if transfers_done >= max_pairs:
                        break
                    if source_cell["cell_id"] == target_cell["cell_id"]:
                        continue
                    if source_cell.get("cell_id") not in self.cell_registry or not source_cell.get("is_stable"):
                        continue
                    for slot in list(self._state_slots(source_cell)):
                        if len(self._state_slots(target_cell)) >= self._slot_capacity():
                            break
                        self.slot_transfer_attempts += 1
                        trial = self._trial_slot_transfer_score(target_cell, source_cell, slot)
                        if not trial.get("accepted"):
                            for reason in trial.get("reject_reasons", []) or ["unknown"]:
                                self.slot_transfer_rejected_by_reason[reason] = self.slot_transfer_rejected_by_reason.get(reason, 0) + 1
                            continue
                        self._transfer_slot_to_state(target_cell, source_cell, slot)
                        self.slot_transfer_accepted += 1
                        transfers_done += 1
                        break
            if transfers_done > 0:
                self._refresh_processed_memory_keys()
                self._recompute_cell_statistics()
            return
        stable_cells = sorted([cell for cell in self._consolidated_cells() if cell.get("is_stable")], key=lambda cell: cell["cell_id"])
        merges_done = 0
        visited = set()
        for target_cell in stable_cells:
            if merges_done >= max_pairs or target_cell["cell_id"] not in self.cell_registry:
                break
            bucket_id = target_cell.get("bucket_id")
            if bucket_id is None:
                continue
            for source_cell in self._bucket_cells(bucket_id):
                if merges_done >= max_pairs:
                    break
                if source_cell["cell_id"] == target_cell["cell_id"]:
                    continue
                pair_key = tuple(sorted((target_cell["cell_id"], source_cell["cell_id"])))
                if pair_key in visited:
                    continue
                visited.add(pair_key)
                if source_cell.get("cell_id") not in self.cell_registry:
                    continue
                if not source_cell.get("is_stable"):
                    continue
                self.consolidation_attempts += 1
                trial = self._trial_merge_state_score(target_cell, source_cell)
                if not trial.get("accepted"):
                    for reason in trial.get("reject_reasons", []) or ["unknown"]:
                        self.consolidation_rejected_by_reason[reason] = self.consolidation_rejected_by_reason.get(reason, 0) + 1
                    continue
                self._merge_states(target_cell, source_cell)
                self.consolidation_accepted += 1
                merges_done += 1
        if merges_done > 0:
            self._refresh_processed_memory_keys()
            self._recompute_cell_statistics()

    def _refresh_single_cell_statistics(self, cell_id: str, *, compute_factor_residuals: bool = False) -> None:
        cell = self.cell_registry.get(cell_id)
        if cell is None:
            return
        entries = self._cell_entries(cell_id)
        if not entries:
            return
        slots = self._state_slots(cell)
        cell["member_count"] = len(entries)
        cell["member_edit_ids"] = [entry["edit_id"] for entry in entries]
        cell["prototype_anchor_text"] = entries[0].get("prompt") if entries else None
        if self._use_shared_basis_codes():
            cell["state_shared_basis"] = self._build_shared_state_basis(slots)
            self._fit_slot_latent_supports(slots, cell["state_shared_basis"])
            for slot in slots:
                if compute_factor_residuals:
                    slot["factor_space_residual"] = self._slot_factor_space_residual(slot, cell["state_shared_basis"])
        summary_sources = []
        slot_usage = []
        slot_residuals = []
        for slot in slots:
            summary_sources.extend(self._slot_view_records(slot))
            slot_usage.append(int(slot.get("slot_usage_count") or 0))
            if slot.get("factor_space_residual") is not None:
                slot_residuals.append(float(slot["factor_space_residual"]))
        state_summary_prototypes, prototype_dispersion = self._summarize_slot_prototypes(summary_sources)
        cell["state_summary_prototypes"] = state_summary_prototypes
        cell["cell_prototypes"] = state_summary_prototypes
        cell["prototype_dispersion"] = prototype_dispersion
        cell["prototype_stats"] = {
            "prototype_count_by_view": {proto.get("view_name"): 1 for proto in state_summary_prototypes},
            "prototype_dispersion": prototype_dispersion,
            "max_intra_cell_prototype_conflict": None,
        }
        cell["prototype_count_by_view"] = cell["prototype_stats"]["prototype_count_by_view"]
        semantic_views = [proto.get("semantic_key") for proto in state_summary_prototypes if isinstance(proto.get("semantic_key"), torch.Tensor)]
        activation_views = [proto.get("activation_key") for proto in state_summary_prototypes if isinstance(proto.get("activation_key"), torch.Tensor)]
        if semantic_views:
            cell["semantic_key"] = self._normalize_semantic_key(torch.stack(semantic_views, dim=0).mean(dim=0))
        if activation_views:
            cell["activation_key"] = self._normalize_activation_key(torch.stack(activation_views, dim=0).mean(dim=0), self.cached_activation_stats)
        slot_pair_scores = []
        for idx in range(len(slots)):
            for jdx in range(idx + 1, len(slots)):
                left = self._score_slot(slots[idx], slots[jdx].get("semantic_key"), slots[jdx].get("activation_key"))
                if left is not None:
                    slot_pair_scores.append(left["combined_score"])
        cell["within_cell_conflict_mean"] = None if not slot_pair_scores else float(sum(slot_pair_scores) / len(slot_pair_scores))
        cell["within_cell_conflict_max"] = None if not slot_pair_scores else float(max(slot_pair_scores))
        cell["slot_usage_mean"] = None if not slot_usage else float(sum(slot_usage) / len(slot_usage))
        cell["factor_space_residual_mean"] = None if not slot_residuals else float(sum(slot_residuals) / len(slot_residuals))
        self._refresh_single_state_metadata(
            cell_id,
            total_edits=len(self.memory_entries),
            warmup=int(getattr(self.hparams, "memory_tier_warmup_edits", 256) or 256),
        )

    def _recompute_cell_statistics(self) -> None:
        if not self.is_v2:
            return
        if self._use_sparse_slots():
            for cell_id in list(self.cell_registry.keys()):
                self._refresh_single_cell_statistics(cell_id, compute_factor_residuals=False)
            self._rebuild_bucket_registry()
            return
        for cell_id, cell in list(self.cell_registry.items()):
            entries = self._cell_entries(cell_id)
            if not entries:
                continue
            semantic_stack = torch.stack([entry["semantic_key"].detach().float().cpu() for entry in entries], dim=0)
            activation_stack = torch.stack([entry["activation_key"].detach().float().cpu() for entry in entries], dim=0)
            centroid_semantic = semantic_stack.mean(dim=0)
            centroid_activation = activation_stack.mean(dim=0)
            cell["semantic_key"] = self._normalize_semantic_key(centroid_semantic)
            cell["activation_key"] = self._normalize_activation_key(centroid_activation, self.cached_activation_stats)
            cell["member_count"] = len(entries)
            cell["member_edit_ids"] = [entry["edit_id"] for entry in entries]
            cell["prototype_anchor_text"] = entries[0].get("prompt") if entries else None

            prototypes, prototype_stats = self._summarize_cell_view_prototypes(entries)
            cell["cell_prototypes"] = prototypes
            cell["prototype_stats"] = prototype_stats
            cell["prototype_dispersion"] = prototype_stats.get("prototype_dispersion")
            cell["prototype_count_by_view"] = prototype_stats.get("prototype_count_by_view")
            cell["max_intra_cell_prototype_conflict"] = prototype_stats.get("max_intra_cell_prototype_conflict")

            within_scores = []
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    scored = self._score_entry_views(entries[i], entries[j]["semantic_key"], entries[j]["activation_key"])
                    if scored is not None:
                        within_scores.append(scored["combined_conflict"])
            cell["within_cell_conflict_mean"] = None if not within_scores else float(sum(within_scores) / len(within_scores))
            cell["within_cell_conflict_max"] = None if not within_scores else float(max(within_scores))
        self._refresh_state_metadata()
        self._rebuild_bucket_registry()

    def _create_new_cell(self, edit_id: str, source_adapter: str) -> dict[str, Any]:
        cell_id = self._next_cell_id()
        adapter_name = None
        if not self._use_sparse_slots():
            adapter_name = cell_id
            self._ensure_adapter(adapter_name)
            self._copy_adapter_parameters(source_adapter, adapter_name)
        cell = {
            "cell_id": cell_id,
            "adapter_name": adapter_name,
            "member_edit_ids": [edit_id],
            "member_count": 1,
            "semantic_key": None,
            "activation_key": None,
            "within_cell_conflict_mean": None,
            "within_cell_conflict_max": None,
            "cell_prototypes": [],
            "prototype_stats": {},
            "prototype_dispersion": None,
            "prototype_count_by_view": {},
            "max_intra_cell_prototype_conflict": None,
            "prototype_anchor_text": None,
            "state_stability_score": None,
            "cross_view_route_gap": None,
            "locality_fragility": None,
            "locality_proxy": None,
            "tier": "active",
            "bucket_id": None,
            "is_stable": False,
            "state_support_observations": 0,
            "state_age_edits": 0,
            "created_at_edit_index": len(self.memory_entries) + 1,
            "state_gate_recent_scores": [],
            "state_gate_score_mean": None,
            "slots": [],
            "state_summary_prototypes": [],
            "state_shared_basis": {},
            "locality_risk_mean": None,
        }
        self.cell_registry[cell_id] = cell
        return cell

    def _build_slot_from_adapter(
        self,
        *,
        cell_id: str,
        edit_id: str,
        request: dict[str, Any],
        prompt: str,
        subject_prompt: str,
        adapter_name: str,
        view_key_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        slot_id = f"{cell_id}_slot_{len(self._state_slots(self.cell_registry[cell_id])):03d}"
        slot_prototypes, slot_dispersion = self._summarize_slot_prototypes(view_key_records)
        slot = {
            "slot_id": slot_id,
            "source_edit_id": edit_id,
            "prompt": prompt,
            "subject": request.get("subject"),
            "relation_id": request.get("relation_id"),
            "source_index": request.get("source_index"),
            "address_rephrase_prompt": request.get("address_rephrase_prompt"),
            "rephrase_prompt": request.get("rephrase_prompt"),
            "subject_prompt": subject_prompt,
            "view_keys": self._clone_to_cpu(view_key_records),
            "slot_prototypes": self._clone_to_cpu(slot_prototypes),
            "slot_dispersion": slot_dispersion,
            "slot_conflict": None,
            "slot_usage_count": 0,
            "slot_false_activation_count": 0,
            "slot_last_selected_at": None,
            "slot_rank": self._slot_rank(),
            "slot_alpha": float(self._slot_rank()),
            "slot_weights": self._capture_adapter_parameters(adapter_name),
            "slot_codes": {},
        }
        semantic_candidates = [proto.get("semantic_key") for proto in slot["slot_prototypes"] if isinstance(proto.get("semantic_key"), torch.Tensor)]
        activation_candidates = [proto.get("activation_key") for proto in slot["slot_prototypes"] if isinstance(proto.get("activation_key"), torch.Tensor)]
        slot["semantic_key"] = None if not semantic_candidates else semantic_candidates[0].clone()
        slot["activation_key"] = None if not activation_candidates else activation_candidates[0].clone()
        return slot

    def _build_edit_entry(
        self,
        *,
        edit_id: str,
        request: dict[str, Any],
        prompt: str,
        raw_semantic_key: torch.Tensor,
        raw_activation_key: torch.Tensor,
        view_key_records: list[dict[str, Any]],
        conflict_ranking: list[dict[str, Any]],
        cell_id: str | None = None,
        assignment: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = {
            "edit_id": edit_id,
            "prompt": prompt,
            "subject": request.get("subject"),
            "relation_id": request.get("relation_id"),
            "source_index": request.get("source_index"),
            "address_rephrase_prompt": request.get("address_rephrase_prompt"),
            "rephrase_prompt": request.get("rephrase_prompt"),
            "target_new": request["target_new"],
            "raw_semantic_key": raw_semantic_key,
            "raw_activation_key": raw_activation_key,
            "semantic_key": raw_semantic_key.clone(),
            "activation_key": raw_activation_key.clone(),
            "view_keys": view_key_records,
            "conflict_neighbors": [
                {
                    "edit_id": row["entry"]["edit_id"],
                    "combined_conflict": row["combined_conflict"],
                    "semantic_conflict": row["semantic_conflict"],
                    "activation_conflict": row["activation_conflict"],
                    "best_view_name": row.get("best_view_name"),
                }
                for row in conflict_ranking[: self.hparams.top_k]
            ],
            "cell_id": cell_id,
            "created_at_edit_index": len(self.memory_entries) + 1,
        }
        if assignment is not None:
            entry["assignment_policy"] = assignment.get("policy")
            entry["assignment_action"] = assignment.get("action")
            entry["assignment_conflict"] = assignment.get("best_conflict")
            entry["assignment_margin"] = assignment.get("assignment_margin")
        return entry

    def _add_edit_v1(
        self,
        request: dict[str, Any],
        prompt: str,
        subject_prompt: str,
        raw_semantic_key: torch.Tensor,
        raw_activation_key: torch.Tensor,
        conflict_ranking: list[dict[str, Any]],
        negative_entries: list[dict[str, Any]],
        view_key_records: list[dict[str, Any]],
    ) -> str:
        edit_id = self._next_edit_id()
        self._ensure_adapter(edit_id)
        self._train_adapter(edit_id, request, prompt, subject_prompt, negative_entries)
        entry = self._build_edit_entry(
            edit_id=edit_id,
            request=request,
            prompt=prompt,
            raw_semantic_key=raw_semantic_key,
            raw_activation_key=raw_activation_key,
            view_key_records=view_key_records,
            conflict_ranking=conflict_ranking,
        )
        self.edit_registry[edit_id] = entry
        self.memory_entries.append(entry)
        self._refresh_processed_memory_keys()
        self._route_from_keys(
            raw_semantic_key,
            raw_activation_key,
            metadata={
                "case_id": request.get("case_id"),
                "prompt": prompt,
                "subject": request.get("subject"),
                "target_edit_id": edit_id,
                "target_memory_id": edit_id,
                "route_event": "post_edit",
            },
        )
        return edit_id

    def _add_edit_v2(
        self,
        request: dict[str, Any],
        prompt: str,
        subject_prompt: str,
        raw_semantic_key: torch.Tensor,
        raw_activation_key: torch.Tensor,
        conflict_ranking: list[dict[str, Any]],
        negative_entries: list[dict[str, Any]],
        view_key_records: list[dict[str, Any]],
    ) -> str:
        edit_id = self._next_edit_id()
        transient_adapter = edit_id
        transient_rank = self._slot_rank() if self._use_sparse_slots() else None
        transient_alpha = float(self._slot_rank()) if self._use_sparse_slots() else None
        self._ensure_adapter(transient_adapter, rank=transient_rank, alpha=transient_alpha)
        self._train_adapter(transient_adapter, request, prompt, subject_prompt, negative_entries)

        assignment = self._select_cell_assignment(raw_semantic_key, raw_activation_key)
        if assignment["action"] == "allocate":
            cell = self._create_new_cell(edit_id, transient_adapter)
        elif self._use_sparse_slots():
            cell = assignment["chosen_cell"]
        else:
            cell = assignment["chosen_cell"]
            self._merge_adapter_into_cell(transient_adapter, cell, assignment)
        cell_id = cell["cell_id"]

        if self._use_sparse_slots():
            slot = self._build_slot_from_adapter(
                cell_id=cell_id,
                edit_id=edit_id,
                request=request,
                prompt=prompt,
                subject_prompt=subject_prompt,
                adapter_name=transient_adapter,
                view_key_records=view_key_records,
            )
            cell.setdefault("slots", []).append(slot)

        entry = self._build_edit_entry(
            edit_id=edit_id,
            request=request,
            prompt=prompt,
            raw_semantic_key=raw_semantic_key,
            raw_activation_key=raw_activation_key,
            view_key_records=view_key_records,
            conflict_ranking=conflict_ranking,
            cell_id=cell_id,
            assignment=assignment,
        )
        if self._use_sparse_slots():
            entry["slot_id"] = slot["slot_id"]
        self.edit_registry[edit_id] = entry
        self.memory_entries.append(entry)
        member_ids = cell.setdefault("member_edit_ids", [])
        if edit_id not in member_ids:
            member_ids.append(edit_id)
        self._delete_adapter(transient_adapter)
        self._refresh_processed_memory_keys()
        self._refresh_single_cell_statistics(cell_id, compute_factor_residuals=False)
        if not self._use_sparse_slots():
            self._collect_state_gate_training_signal(entry)
            self._maybe_online_update_state_gate()
        if bool(getattr(self.hparams, "hierarchy_enable", True)) and len(self.memory_entries) >= int(getattr(self.hparams, "hierarchy_start_edit", 1024) or 1024):
            self._rebuild_bucket_registry()
        self.maybe_consolidate_states()
        self._route_from_keys(
            raw_semantic_key,
            raw_activation_key,
            metadata={
                "case_id": request.get("case_id"),
                "prompt": prompt,
                "subject": request.get("subject"),
                "target_edit_id": edit_id,
                "target_cell_id": cell_id,
                "target_memory_id": cell_id,
                "route_event": "post_edit",
            },
        )
        return edit_id

    def _add_edit_v3(
        self,
        request: dict[str, Any],
        prompt: str,
        subject_prompt: str,
        raw_semantic_key: torch.Tensor,
        raw_activation_key: torch.Tensor,
        conflict_ranking: list[dict[str, Any]],
        negative_entries: list[dict[str, Any]],
        view_key_records: list[dict[str, Any]],
    ) -> str:
        edit_id = self._next_edit_id()
        positive_views = self._build_training_views(request, prompt, subject_prompt)
        base_loss = self._evaluate_sequence_loss(positive_views, request["target_new"], adapter_name=None)
        shard_ranking = self._rank_shards_from_keys(raw_semantic_key, raw_activation_key)
        chosen_shard = shard_ranking[0]["shard"] if shard_ranking else None
        selected_atoms: list[dict[str, Any]] = []
        selected_atom_ids: list[str] = []
        support_loss = None
        projected_overlap = 0.0
        assignment_action = "new_shard"
        shard_margin = 0.0

        if chosen_shard is not None:
            runner_up = shard_ranking[1]["combined_score"] if len(shard_ranking) > 1 else None
            shard_margin = float(shard_ranking[0]["combined_score"] - runner_up) if runner_up is not None else float(shard_ranking[0]["combined_score"])
            atom_rows = self._rank_atoms_in_shard(chosen_shard, raw_semantic_key, raw_activation_key, apply_usage_penalty=True)
            selected_atoms = self._normalize_atom_selection(atom_rows, topk=self._side_memory_write_topk())
            selected_atom_ids = [row["atom"]["atom_id"] for row in selected_atoms]
            projected_overlap = self._support_overlap_with_entries(chosen_shard["shard_id"], selected_atom_ids)
            adapter_name = self._load_side_memory_runtime_adapter(selected_atoms) if selected_atoms else None
            if adapter_name is not None:
                support_loss = self._evaluate_sequence_loss(positive_views, request["target_new"], adapter_name=adapter_name)
            capacity_full = len(self._shard_atoms(chosen_shard)) >= self._side_memory_atoms_per_shard()
            overlap_veto = projected_overlap > float(getattr(self.hparams, "support_overlap_veto", 0.75) or 0.75)
            residual_veto = support_loss is None or support_loss > float(getattr(self.hparams, "new_shard_residual_threshold", 1.0) or 1.0)
            reuse_threshold = float(getattr(self.hparams, "side_memory_loss_threshold", 2.5) or 2.5)
            if selected_atoms and support_loss is not None and support_loss <= min(base_loss, reuse_threshold) and not overlap_veto:
                assignment_action = "reuse_support"
            else:
                if overlap_veto:
                    self.support_exclusivity_failures += 1
                    chosen_shard["support_exclusivity_failures"] = int(chosen_shard.get("support_exclusivity_failures") or 0) + 1
                if capacity_full or (overlap_veto and residual_veto):
                    chosen_shard = None
                    assignment_action = "new_shard"
                else:
                    assignment_action = "new_atom"

        if chosen_shard is None:
            chosen_shard = self._create_new_shard(view_key_records=view_key_records)
            selected_atoms = []
            selected_atom_ids = []
            projected_overlap = 0.0

        write_loss = support_loss if support_loss is not None else base_loss
        if assignment_action != "reuse_support":
            trained = self._train_new_side_memory_atom(request, prompt, subject_prompt, negative_entries)
            atom_id = self._next_atom_id()
            new_atom = self._initialize_new_side_memory_atom(atom_id, trained["adapter_name"], view_key_records)
            self._delete_adapter(trained["adapter_name"])
            chosen_shard.setdefault("atoms", []).append(new_atom)
            selected_atoms = [{"atom": new_atom, "atom_weight": 1.0}]
            selected_atom_ids = [atom_id]
            write_loss = float(trained["write_loss"])
            assignment_action = "new_atom" if chosen_shard.get("member_edit_ids") else "new_shard"

        for row in selected_atoms:
            atom = row["atom"]
            atom.setdefault("member_edit_ids", []).append(edit_id)
            atom["usage_count"] = int(atom.get("usage_count") or 0) + 1
            atom_view_keys = atom.setdefault("view_keys", [])
            atom_view_keys.extend(self._clone_to_cpu(view_key_records))
            prototypes, dispersion = self._summarize_slot_prototypes(atom_view_keys)
            atom["atom_prototypes"] = self._clone_to_cpu(prototypes)
            atom["atom_dispersion"] = dispersion

        entry = self._build_edit_entry(
            edit_id=edit_id,
            request=request,
            prompt=prompt,
            raw_semantic_key=raw_semantic_key,
            raw_activation_key=raw_activation_key,
            view_key_records=view_key_records,
            conflict_ranking=conflict_ranking,
            cell_id=None,
            assignment={
                "policy": "support_first_side_memory",
                "action": assignment_action,
                "best_conflict": projected_overlap,
                "assignment_margin": shard_margin,
            },
        )
        entry["shard_id"] = chosen_shard["shard_id"]
        entry["support_atom_ids"] = list(selected_atom_ids)
        entry["support_amplitudes"] = [float(row.get("atom_weight", 1.0)) for row in selected_atoms]
        entry["write_residual"] = None if support_loss is None else float(max(0.0, support_loss - write_loss))
        entry["base_loss"] = float(base_loss)
        entry["side_memory_loss"] = float(write_loss)
        self.edit_registry[edit_id] = entry
        self.memory_entries.append(entry)
        chosen_shard.setdefault("member_edit_ids", []).append(edit_id)
        self._refresh_processed_memory_keys()
        self._refresh_single_shard_metadata(chosen_shard["shard_id"])
        self._route_from_keys(
            raw_semantic_key,
            raw_activation_key,
            metadata={
                "case_id": request.get("case_id"),
                "prompt": prompt,
                "subject": request.get("subject"),
                "target_edit_id": edit_id,
                "target_memory_id": chosen_shard["shard_id"],
                "route_event": "post_edit",
            },
        )
        return edit_id

    def _add_edit_v4(
        self,
        request: dict[str, Any],
        prompt: str,
        subject_prompt: str,
        raw_semantic_key: torch.Tensor,
        raw_activation_key: torch.Tensor,
        conflict_ranking: list[dict[str, Any]],
        negative_entries: list[dict[str, Any]],
        view_key_records: list[dict[str, Any]],
    ) -> str:
        edit_id = self._next_edit_id()
        self._ensure_adapter(edit_id)
        self._train_adapter(edit_id, request, prompt, subject_prompt, negative_entries)
        value_ref = edit_id
        if self._use_cold_trace_values():
            self._store_trace_value_reference(value_ref, edit_id)
            self._drop_runtime_adapter_without_disabling(edit_id)
        entry = self._build_edit_entry(
            edit_id=edit_id,
            request=request,
            prompt=prompt,
            raw_semantic_key=raw_semantic_key,
            raw_activation_key=raw_activation_key,
            view_key_records=view_key_records,
            conflict_ranking=conflict_ranking,
        )
        entry["trace_id"] = edit_id
        entry["trace_address"] = {
            "num_views": len(view_key_records),
            "view_names": [view.get("view_name") for view in view_key_records],
        }
        if self._use_sparse_address_trace_bank():
            entry["trace_address_impl"] = "factored_subject_relation" if self._use_factored_address_trace_bank() else ("overlap_sparse_dictionary" if self._use_overlap_aware_anchor_trace_bank() else "sparse_dictionary")
        entry["trace_value_impl"] = "exact_lora_cold_store" if self._use_cold_trace_values() else "exact_lora"
        entry["value_ref"] = value_ref
        entry["value_adapter_name"] = None if self._use_cold_trace_values() else edit_id
        self.edit_registry[edit_id] = entry
        self.memory_entries.append(entry)
        if self._use_factored_address_trace_bank():
            self._refresh_single_trace_entry_keys(entry)
            if self._maybe_train_factored_relation_encoder() and bool(getattr(self.hparams, "factored_relation_encoder_rebuild_on_train", True)):
                self._rebuild_factored_trace_address_state()
            else:
                self._index_factored_trace_entry(entry)
        elif self._use_overlap_aware_anchor_trace_bank():
            self._refresh_single_trace_entry_keys(entry)
            self._index_overlap_anchor_trace_entry(entry)
        else:
            self._refresh_processed_memory_keys()
        self._route_from_keys(
            raw_semantic_key,
            raw_activation_key,
            metadata={
                "case_id": request.get("case_id"),
                "prompt": prompt,
                "subject": request.get("subject"),
                "target_edit_id": edit_id,
                "target_memory_id": edit_id,
                "route_event": "post_edit",
            },
        )
        return edit_id

    def add_edit(self, request: dict[str, Any]) -> str:
        prompt = self._format_prompt(request)
        subject_prompt = self._subject_conditioned_prompt(request, prompt)
        raw_semantic_key, raw_activation_key = self._extract_keys([prompt], [subject_prompt])
        conflict_ranking = self._rank_conflicts_from_keys(raw_semantic_key, raw_activation_key)
        negative_entries = self._select_negative_entries(conflict_ranking)
        view_key_records = self._build_view_key_records(request, prompt, subject_prompt)

        if self.is_v2:
            return self._add_edit_v2(
                request,
                prompt,
                subject_prompt,
                raw_semantic_key,
                raw_activation_key,
                conflict_ranking,
                negative_entries,
                view_key_records,
            )
        if self.is_v3:
            return self._add_edit_v3(
                request,
                prompt,
                subject_prompt,
                raw_semantic_key,
                raw_activation_key,
                conflict_ranking,
                negative_entries,
                view_key_records,
            )
        if self.is_v4:
            return self._add_edit_v4(
                request,
                prompt,
                subject_prompt,
                raw_semantic_key,
                raw_activation_key,
                conflict_ranking,
                negative_entries,
                view_key_records,
            )
        return self._add_edit_v1(
            request,
            prompt,
            subject_prompt,
            raw_semantic_key,
            raw_activation_key,
            conflict_ranking,
            negative_entries,
            view_key_records,
        )

    def rollback_edit(self, edit_id: str) -> None:
        if self.is_v2 or self.is_v3:
            raise RuntimeError("Merged-memory HopEdit edits are not individually rollbackable.")
        self.memory_entries = [entry for entry in self.memory_entries if entry["edit_id"] != edit_id]
        self.edit_registry.pop(edit_id, None)
        if self._use_cold_trace_values():
            self._remove_trace_value_reference(edit_id)
        else:
            self._delete_adapter(edit_id)
        self._refresh_processed_memory_keys()

    def save_route_logs(self, path: str | None = None) -> str:
        if path is None:
            path = os.path.join(self.hparams.route_log_dir, "hopedit_route_logs.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for entry in self.route_logs:
                handle.write(json.dumps(entry) + "\n")
        return path

    def export_memory_snapshot(self, include_keys: bool = False) -> list[dict[str, Any]]:
        snapshot = []
        num_cells = len(self.cell_registry) if self.is_v2 else 0
        cell_sizes = {cell_id: len(self._cell_entries(cell_id)) for cell_id in self.cell_registry}
        shard_sizes = {shard_id: len((self.shard_registry.get(shard_id) or {}).get("member_edit_ids") or []) for shard_id in self.shard_registry}
        for entry in self.memory_entries:
            row = {
                "memory_unit": self.memory_unit,
                "edit_id": entry.get("edit_id"),
                "trace_id": entry.get("trace_id"),
                "cell_id": entry.get("cell_id"),
                "shard_id": entry.get("shard_id"),
                "prompt": entry.get("prompt"),
                "subject": entry.get("subject"),
                "rephrase_prompt": entry.get("rephrase_prompt"),
                "target_new": entry.get("target_new"),
                "probe_layers": self.hparams.probe_layers if self.hparams.probe_layers else [self.hparams.probe_layer],
                "raw_semantic_norm": float(entry["raw_semantic_key"].norm().item()) if isinstance(entry.get("raw_semantic_key"), torch.Tensor) else None,
                "raw_activation_norm": float(entry["raw_activation_key"].norm().item()) if isinstance(entry.get("raw_activation_key"), torch.Tensor) else None,
                "semantic_norm": float(entry["semantic_key"].norm().item()) if isinstance(entry.get("semantic_key"), torch.Tensor) else None,
                "activation_norm": float(entry["activation_key"].norm().item()) if isinstance(entry.get("activation_key"), torch.Tensor) else None,
                "num_views": len(self._entry_view_records(entry)),
                "view_names": [view.get("view_name") for view in self._entry_view_records(entry)],
                "conflict_neighbors": entry.get("conflict_neighbors", []),
                "num_cells": num_cells,
                "cell_member_count": cell_sizes.get(entry.get("cell_id")),
                "assignment_policy": entry.get("assignment_policy"),
                "assignment_action": entry.get("assignment_action"),
                "assignment_conflict": entry.get("assignment_conflict"),
                "assignment_margin": entry.get("assignment_margin"),
            }
            if self.is_v3 and entry.get("shard_id") in self.shard_registry:
                shard = self.shard_registry[entry["shard_id"]]
                row["shard_member_count"] = shard_sizes.get(entry.get("shard_id"))
                row["support_atom_ids"] = list(entry.get("support_atom_ids") or [])
                row["support_amplitudes"] = list(entry.get("support_amplitudes") or [])
                row["shard_atom_count"] = len(self._shard_atoms(shard))
                row["shard_prototype_dispersion"] = shard.get("prototype_dispersion")
                row["write_residual"] = entry.get("write_residual")
            if self.is_v4:
                row["trace_address"] = self._clone_to_json(entry.get("trace_address"))
                row["trace_value_impl"] = entry.get("trace_value_impl")
                row["value_ref"] = entry.get("value_ref")
                row["value_adapter_name"] = entry.get("value_adapter_name")
                row["relation_id"] = entry.get("relation_id")
                row["address_rephrase_prompt"] = entry.get("address_rephrase_prompt")
                if self._use_sparse_address_trace_bank():
                    row["trace_address_impl"] = entry.get("trace_address_impl")
                    row["trace_address_support"] = list(entry.get("trace_address_support") or [])
                    row["trace_positive_support"] = list(entry.get("trace_positive_support") or [])
                    row["trace_family_ids"] = list(entry.get("trace_family_ids") or [])
                    row["trace_primary_family_id"] = entry.get("trace_primary_family_id")
                    row["trace_family_support"] = list(entry.get("trace_family_support") or [])
                    row["trace_family_negative_trace_ids"] = list(entry.get("trace_family_negative_trace_ids") or [])
                    row["trace_irrelevant_negative_trace_ids"] = list(entry.get("trace_irrelevant_negative_trace_ids") or [])
                    row["trace_anchor_names"] = list(entry.get("trace_anchor_names") or [])
                    row["trace_anchor_views"] = self._clone_to_json(entry.get("trace_anchor_views"))
                    row["trace_factor_views"] = self._clone_to_json(entry.get("trace_factor_views"))
                    row["trace_subject_factor_norm"] = None if not isinstance(entry.get("trace_subject_factor"), torch.Tensor) else float(entry["trace_subject_factor"].norm().item())
                    row["trace_relation_factor_norm"] = None if not isinstance(entry.get("trace_relation_factor"), torch.Tensor) else float(entry["trace_relation_factor"].norm().item())
                    row["trace_subject_agreement"] = entry.get("trace_subject_agreement")
                    row["trace_relation_agreement"] = entry.get("trace_relation_agreement")
                    row["trace_address_agreement"] = entry.get("trace_address_agreement")
                    row["trace_address_code_topk"] = entry.get("trace_address_code_topk")
                    row["trace_address_codebook_version"] = entry.get("trace_address_codebook_version")
                    row["trace_min_activation_energy"] = entry.get("trace_min_activation_energy")
                    row["trace_min_activation_margin"] = entry.get("trace_min_activation_margin")
                    row["trace_min_anchor_energy"] = entry.get("trace_min_anchor_energy")
                    row["trace_min_family_margin"] = entry.get("trace_min_family_margin")
                    row["trace_min_locality_margin"] = entry.get("trace_min_locality_margin")
                    row["trace_max_exclusion_score"] = entry.get("trace_max_exclusion_score")
                    row["trace_negative_energy_ceiling"] = entry.get("trace_negative_energy_ceiling")
                    row["trace_family_negative_energy_ceiling"] = entry.get("trace_family_negative_energy_ceiling")
                    row["trace_irrelevant_negative_energy_ceiling"] = entry.get("trace_irrelevant_negative_energy_ceiling")
                    row["trace_positive_energy_floor"] = entry.get("trace_positive_energy_floor")
                    row["trace_positive_energy_mean"] = entry.get("trace_positive_energy_mean")
                    row["trace_positive_margin_floor"] = entry.get("trace_positive_margin_floor")
                    row["trace_positive_margin_mean"] = entry.get("trace_positive_margin_mean")
                    row["trace_positive_locality_margin_floor"] = entry.get("trace_positive_locality_margin_floor")
                    row["trace_positive_locality_margin_mean"] = entry.get("trace_positive_locality_margin_mean")
                    row["trace_positive_exclusion_score_ceiling"] = entry.get("trace_positive_exclusion_score_ceiling")
                    row["trace_negative_exclusion_score_floor"] = entry.get("trace_negative_exclusion_score_floor")
                    row["trace_exclusion_support"] = list(entry.get("trace_exclusion_support") or [])
                    row["trace_exclusion_trace_ids"] = list(entry.get("trace_exclusion_trace_ids") or [])
                    row["trace_shape_width"] = entry.get("trace_shape_width")
            if self.is_v2 and entry.get("cell_id") in self.cell_registry:
                cell = self.cell_registry[entry["cell_id"]]
                row["prototype_view_names"] = [proto.get("view_name") for proto in self._cell_view_records(cell)]
                row["prototype_count_by_view"] = cell.get("prototype_count_by_view")
                row["prototype_dispersion"] = cell.get("prototype_dispersion")
                row["max_intra_cell_prototype_conflict"] = cell.get("max_intra_cell_prototype_conflict")
                row["tier"] = cell.get("tier")
                row["bucket_id"] = cell.get("bucket_id")
                row["state_stability_score"] = cell.get("state_stability_score")
                row["cross_view_route_gap"] = cell.get("cross_view_route_gap")
                row["locality_fragility"] = cell.get("locality_fragility")
                row["state_support_observations"] = cell.get("state_support_observations")
                row["state_gate_score_mean"] = cell.get("state_gate_score_mean")
                if self._use_sparse_slots():
                    row["state_slot_count"] = len(self._state_slots(cell))
                    row["slot_ids"] = [slot.get("slot_id") for slot in self._state_slots(cell)]
                    row["slot_usage_mean"] = cell.get("slot_usage_mean")
            if include_keys:
                row["semantic_key"] = entry["semantic_key"].tolist() if isinstance(entry.get("semantic_key"), torch.Tensor) else entry.get("semantic_key")
                row["activation_key"] = entry["activation_key"].tolist() if isinstance(entry.get("activation_key"), torch.Tensor) else entry.get("activation_key")
            snapshot.append(self._clone_to_json(row))
        return snapshot

    def export_slot_diagnostics(self) -> dict[str, Any]:
        if not self.is_v2 or not self._use_sparse_slots():
            return {
                "applicable": False,
                "slot_count_per_state": {},
                "activated_slot_count_distribution": [],
                "slot_transfer_attempts": 0,
                "accepted_slot_transfers": 0,
                "rejected_slot_transfers_by_reason": {},
                "state_summary": [],
            }
        activated_slot_count_distribution = [len(row.get("selected_slot_ids", [])) for row in self.route_logs if row.get("selected_slot_ids")]
        state_summary = []
        slot_count_per_state = {}
        for cell_id in sorted(self.cell_registry.keys()):
            cell = self.cell_registry[cell_id]
            slots = self._state_slots(cell)
            slot_count_per_state[cell_id] = len(slots)
            state_summary.append(
                {
                    "state_id": cell_id,
                    "slot_count": len(slots),
                    "slot_usage_distribution": [int(slot.get("slot_usage_count") or 0) for slot in slots],
                    "slot_dispersion": [slot.get("slot_dispersion") for slot in slots],
                    "slot_conflict": [slot.get("slot_conflict") for slot in slots],
                }
            )
        return {
            "applicable": True,
            "slot_count_per_state": slot_count_per_state,
            "activated_slot_count_distribution": activated_slot_count_distribution,
            "slot_transfer_attempts": int(self.slot_transfer_attempts),
            "accepted_slot_transfers": int(self.slot_transfer_accepted),
            "rejected_slot_transfers_by_reason": dict(self.slot_transfer_rejected_by_reason),
            "state_summary": state_summary,
        }

    def export_shard_diagnostics(self) -> dict[str, Any]:
        if self._use_sparse_address_trace_bank():
            inference_rows = [row for row in self.route_logs if row.get("route_event") == "inference"]
            candidate_sizes = [float(row["candidate_set_size"]) for row in inference_rows if row.get("candidate_set_size") is not None]
            family_shortlists = [float(row["family_shortlist_size"]) for row in inference_rows if row.get("family_shortlist_size") is not None]
            postings_sizes = [len(trace_ids) for trace_ids in self.address_postings.values()]
            return {
                "applicable": True,
                "memory_unit": "trace",
                "shard_count": len(self.address_dictionary.get("atoms") or []),
                "address_atom_count": len(self.address_dictionary.get("atoms") or []),
                "address_atom_coherence_mean": self.address_dictionary.get("coherence_mean"),
                "address_family_count": len(self.family_postings),
                "candidate_set_size_mean": None if not candidate_sizes else float(sum(candidate_sizes) / len(candidate_sizes)),
                "family_shortlist_size_mean": None if not family_shortlists else float(sum(family_shortlists) / len(family_shortlists)),
                "posting_size_mean": None if not postings_sizes else float(sum(postings_sizes) / len(postings_sizes)),
                "value_cache_size": len(self.trace_value_cache) if self._use_cold_trace_values() else None,
                "shard_summary": [],
            }
        if not self.is_v3:
            return {
                "applicable": False,
                "shard_count": 0,
                "shard_summary": [],
            }
        shard_summary = []
        margins = []
        for shard_id in sorted(self.shard_registry.keys()):
            shard = self.shard_registry[shard_id]
            prototype_history = [float(value) for value in shard.get("prototype_margin_history") or [] if value is not None]
            if prototype_history:
                margins.extend(prototype_history)
            shard_summary.append(
                {
                    "shard_id": shard_id,
                    "occupancy": len(self._shard_atoms(shard)),
                    "member_count": len(shard.get("member_edit_ids") or []),
                    "prototype_dispersion": shard.get("prototype_dispersion"),
                    "prototype_margin_mean": None if not prototype_history else float(sum(prototype_history) / len(prototype_history)),
                    "support_exclusivity_failures": int(shard.get("support_exclusivity_failures") or 0),
                    "base_only_fallbacks": int(shard.get("base_only_fallbacks") or 0),
                }
            )
        return {
            "applicable": True,
            "shard_count": len(self.shard_registry),
            "shard_occupancy_mean": None if not shard_summary else float(sum(row["occupancy"] for row in shard_summary) / len(shard_summary)),
            "shard_prototype_margin_mean": None if not margins else float(sum(margins) / len(margins)),
            "shard_summary": shard_summary,
        }

    def export_support_diagnostics(self) -> dict[str, Any]:
        if self._use_sparse_address_trace_bank():
            support_sizes = []
            overlaps = []
            agreements = []
            family_support_sizes = []
            family_support_overlaps = []
            support_sets: list[set[int]] = []
            family_support_sets: list[set[int]] = []
            reuse_count = 0
            for entry in self.memory_entries:
                support = list(entry.get("trace_address_support") or [])
                if not support:
                    continue
                support_sizes.append(float(len(support)))
                support_set = set(support)
                if any(support_set & existing for existing in support_sets):
                    reuse_count += 1
                support_sets.append(support_set)
                family_support = list(entry.get("trace_family_support") or [])
                if family_support:
                    family_support_sizes.append(float(len(family_support)))
                    family_support_sets.append(set(family_support))
                agreement = entry.get("trace_address_agreement")
                if agreement is not None:
                    agreements.append(float(agreement))
            for left_index in range(len(support_sets)):
                for right_index in range(left_index + 1, len(support_sets)):
                    overlaps.append(self._support_overlap(support_sets[left_index], support_sets[right_index]))
            for left_index in range(len(family_support_sets)):
                for right_index in range(left_index + 1, len(family_support_sets)):
                    family_support_overlaps.append(self._support_overlap(family_support_sets[left_index], family_support_sets[right_index]))
            return {
                "applicable": True,
                "support_size_mean": None if not support_sizes else float(sum(support_sizes) / len(support_sizes)),
                "support_overlap_mean": None if not overlaps else float(sum(overlaps) / len(overlaps)),
                "family_support_size_mean": None if not family_support_sizes else float(sum(family_support_sizes) / len(family_support_sizes)),
                "family_support_overlap_mean": None if not family_support_overlaps else float(sum(family_support_overlaps) / len(family_support_overlaps)),
                "support_reuse_rate": None if not support_sets else float(reuse_count / len(support_sets)),
                "support_exclusivity_failures": 0,
                "whole_model_support_overlap_mean": None if not overlaps else float(sum(overlaps) / len(overlaps)),
                "cross_view_code_agreement_mean": None if not agreements else float(sum(agreements) / len(agreements)),
            }
        if not self.is_v3:
            return {
                "applicable": False,
                "support_size_mean": None,
                "support_overlap_mean": None,
                "support_reuse_rate": None,
                "support_exclusivity_failures": 0,
            }
        support_sizes = []
        reuse_count = 0
        overlaps = []
        support_sets = []
        for entry in self.memory_entries:
            atom_ids = list(entry.get("support_atom_ids") or [])
            if not atom_ids:
                continue
            support_sizes.append(float(len(atom_ids)))
            if entry.get("assignment_action") == "reuse_support":
                reuse_count += 1
            support_sets.append(set(atom_ids))
        for left_index in range(len(support_sets)):
            for right_index in range(left_index + 1, len(support_sets)):
                union = support_sets[left_index] | support_sets[right_index]
                if not union:
                    continue
                overlaps.append(float(len(support_sets[left_index] & support_sets[right_index]) / len(union)))
        return {
            "applicable": True,
            "support_size_mean": None if not support_sizes else float(sum(support_sizes) / len(support_sizes)),
            "support_overlap_mean": None if not overlaps else float(sum(overlaps) / len(overlaps)),
            "support_reuse_rate": None if not self.memory_entries else float(reuse_count / len(self.memory_entries)),
            "support_exclusivity_failures": int(self.support_exclusivity_failures),
            "whole_model_support_overlap_mean": None if not overlaps else float(sum(overlaps) / len(overlaps)),
        }

    def export_realization_diagnostics(self) -> dict[str, Any]:
        if self._use_sparse_address_trace_bank():
            inference_rows = [row for row in self.route_logs if row.get("route_event") == "inference"]
            fallback_values = [1.0 if row.get("base_only_fallback") else 0.0 for row in inference_rows]
            energy_margins = [float(row["trace_energy_margin"]) for row in inference_rows if row.get("trace_energy_margin") is not None]
            locality_margins = [float(row["trace_locality_margin"]) for row in inference_rows if row.get("trace_locality_margin") is not None]
            exclusion_scores = [float(row["trace_exclusion_score"]) for row in inference_rows if row.get("trace_exclusion_score") is not None]
            anchor_energies = [float(row["trace_anchor_energy"]) for row in inference_rows if row.get("trace_anchor_energy") is not None]
            family_margins = [float(row["trace_family_margin"]) for row in inference_rows if row.get("trace_family_margin") is not None]
            cache_hits = [1.0 if row.get("value_cache_hit") else 0.0 for row in inference_rows if row.get("value_cache_hit") is not None]
            return {
                "applicable": True,
                "route_correct_but_behavior_wrong_rate": None,
                "off_target_realization_energy_mean": 0.0 if inference_rows else None,
                "base_only_fallback_rate": None if not fallback_values else float(sum(fallback_values) / len(fallback_values)),
                "trace_energy_margin_mean": None if not energy_margins else float(sum(energy_margins) / len(energy_margins)),
                "trace_locality_margin_mean": None if not locality_margins else float(sum(locality_margins) / len(locality_margins)),
                "trace_exclusion_score_mean": None if not exclusion_scores else float(sum(exclusion_scores) / len(exclusion_scores)),
                "trace_anchor_energy_mean": None if not anchor_energies else float(sum(anchor_energies) / len(anchor_energies)),
                "trace_family_margin_mean": None if not family_margins else float(sum(family_margins) / len(family_margins)),
                "value_cache_hit_rate": None if not cache_hits else float(sum(cache_hits) / len(cache_hits)),
                "value_cache_size": len(self.trace_value_cache) if self._use_cold_trace_values() else None,
                "value_impl": "exact_lora_cold_store" if self._use_cold_trace_values() else "exact_lora",
            }
        if not self.is_v3:
            return {
                "applicable": False,
                "route_correct_but_behavior_wrong_rate": None,
                "off_target_realization_energy_mean": None,
                "base_only_fallback_rate": None,
            }
        inference_rows = [row for row in self.route_logs if row.get("route_event") == "inference"]
        overlaps = [float(row["realization_overlap"]) for row in inference_rows if row.get("realization_overlap") is not None]
        fallbacks = [1.0 if row.get("base_only_fallback") else 0.0 for row in inference_rows]
        return {
            "applicable": True,
            "route_correct_but_behavior_wrong_rate": None,
            "off_target_realization_energy_mean": None if not overlaps else float(sum(overlaps) / len(overlaps)),
            "base_only_fallback_rate": None if not fallbacks else float(sum(fallbacks) / len(fallbacks)),
        }

    def export_factor_space_diagnostics(self) -> dict[str, Any]:
        if self._use_sparse_address_trace_bank():
            inference_rows = [row for row in self.route_logs if row.get("route_event") == "inference"]
            atom_count = max(1, len(self.address_dictionary.get("atoms") or []))
            support_sizes = [float(row["address_support_size"]) for row in inference_rows if row.get("address_support_size") is not None]
            support_overlaps = [float(row["address_support_overlap"]) for row in inference_rows if row.get("address_support_overlap") is not None]
            exclusion_overlaps = [float(row["address_exclusion_overlap"]) for row in inference_rows if row.get("address_exclusion_overlap") is not None]
            family_overlaps = [float(row["address_family_overlap"]) for row in inference_rows if row.get("address_family_overlap") is not None]
            coherences = [float(row["address_atom_coherence"]) for row in inference_rows if row.get("address_atom_coherence") is not None]
            agreements = [float(row["cross_view_code_agreement"]) for row in inference_rows if row.get("cross_view_code_agreement") is not None]
            energy_margins = [float(row["trace_energy_margin"]) for row in inference_rows if row.get("trace_energy_margin") is not None]
            locality_margins = [float(row["trace_locality_margin"]) for row in inference_rows if row.get("trace_locality_margin") is not None]
            exclusion_scores = [float(row["trace_exclusion_score"]) for row in inference_rows if row.get("trace_exclusion_score") is not None]
            anchor_energies = [float(row["trace_anchor_energy"]) for row in inference_rows if row.get("trace_anchor_energy") is not None]
            family_margins = [float(row["trace_family_margin"]) for row in inference_rows if row.get("trace_family_margin") is not None]
            family_overlap_scores = [float(row["trace_family_overlap_score"]) for row in inference_rows if row.get("trace_family_overlap_score") is not None]
            candidate_sizes = [float(row["candidate_set_size"]) for row in inference_rows if row.get("candidate_set_size") is not None]
            family_shortlists = [float(row["family_shortlist_size"]) for row in inference_rows if row.get("family_shortlist_size") is not None]
            subject_margins = [float(row["factor_subject_margin"]) for row in inference_rows if row.get("factor_subject_margin") is not None]
            relation_margins = [float(row["factor_relation_margin"]) for row in inference_rows if row.get("factor_relation_margin") is not None]
            subject_energies = [float(row["factor_subject_energy"]) for row in inference_rows if row.get("factor_subject_energy") is not None]
            relation_energies = [float(row["factor_relation_energy"]) for row in inference_rows if row.get("factor_relation_energy") is not None]
            abstain_rate = None
            locality_veto_rate = None
            failure_partitions = {
                "subject": 0,
                "relation": 0,
                "both": 0,
                "none": 0,
            }
            if inference_rows:
                abstain_rate = float(
                    sum(1.0 if row.get("address_abstained") else 0.0 for row in inference_rows) / len(inference_rows)
                )
                locality_veto_rate = float(
                    sum(1.0 if row.get("address_locality_vetoed") else 0.0 for row in inference_rows) / len(inference_rows)
                )
                for row in inference_rows:
                    failure_key = str(row.get("factor_failure_partition") or "none")
                    if failure_key in failure_partitions:
                        failure_partitions[failure_key] += 1
            return {
                "applicable": True,
                "state_margin_mean": None if not energy_margins else float(sum(energy_margins) / len(energy_margins)),
                "cross_view_gap_mean": None if not agreements else float(1.0 - (sum(agreements) / len(agreements))),
                "realization_overlap_mean": 0.0 if inference_rows else None,
                "code_sparsity_mean": None if not support_sizes else float(sum(size / atom_count for size in support_sizes) / len(support_sizes)),
                "support_size_mean": None if not support_sizes else float(sum(support_sizes) / len(support_sizes)),
                "support_overlap_mean": None if not support_overlaps else float(sum(support_overlaps) / len(support_overlaps)),
                "exclusion_overlap_mean": None if not exclusion_overlaps else float(sum(exclusion_overlaps) / len(exclusion_overlaps)),
                "family_overlap_mean": None if not family_overlaps else float(sum(family_overlaps) / len(family_overlaps)),
                "atom_coherence_mean": None if not coherences else float(sum(coherences) / len(coherences)),
                "code_l1_mean": 1.0 if support_sizes else None,
                "code_l2_mean": None,
                "factor_space_residual_mean": None,
                "residual_after_sparse_pursuit_mean": None,
                "candidate_set_size_mean": None if not candidate_sizes else float(sum(candidate_sizes) / len(candidate_sizes)),
                "family_shortlist_size_mean": None if not family_shortlists else float(sum(family_shortlists) / len(family_shortlists)),
                "cross_view_code_agreement_mean": None if not agreements else float(sum(agreements) / len(agreements)),
                "trace_locality_margin_mean": None if not locality_margins else float(sum(locality_margins) / len(locality_margins)),
                "trace_exclusion_score_mean": None if not exclusion_scores else float(sum(exclusion_scores) / len(exclusion_scores)),
                "trace_anchor_energy_mean": None if not anchor_energies else float(sum(anchor_energies) / len(anchor_energies)),
                "trace_family_margin_mean": None if not family_margins else float(sum(family_margins) / len(family_margins)),
                "trace_family_overlap_score_mean": None if not family_overlap_scores else float(sum(family_overlap_scores) / len(family_overlap_scores)),
                "factor_subject_margin_mean": None if not subject_margins else float(sum(subject_margins) / len(subject_margins)),
                "factor_relation_margin_mean": None if not relation_margins else float(sum(relation_margins) / len(relation_margins)),
                "factor_subject_energy_mean": None if not subject_energies else float(sum(subject_energies) / len(subject_energies)),
                "factor_relation_energy_mean": None if not relation_energies else float(sum(relation_energies) / len(relation_energies)),
                "factored_relation_encoder_impl": str(getattr(self.hparams, "factored_relation_encoder_impl", "identity") or "identity"),
                "factored_relation_encoder_updates": int(self.factored_relation_encoder_updates),
                "factored_relation_encoder_last_loss": self.factored_relation_encoder_last_loss,
                "factored_relation_encoder_checkpoint_loaded": self.factored_relation_encoder_checkpoint_loaded,
                "factored_relation_encoder_checkpoint_metadata": self._clone_to_json(self.factored_relation_encoder_checkpoint_metadata),
                "factored_relation_match_rule": str(getattr(self.hparams, "factored_relation_match_rule", "top1_same_trace") or "top1_same_trace"),
                "factored_relation_storage_transform": self._factored_relation_storage_transform(),
                "factored_relation_score_transform": self._factored_relation_score_transform(),
                "factored_relation_streaming_pc_rank": int(getattr(self.hparams, "factored_relation_streaming_pc_rank", 8) or 8),
                "factored_relation_streaming_pc_min_traces": int(getattr(self.hparams, "factored_relation_streaming_pc_min_traces", 8) or 8),
                "factor_failure_partition_counts": failure_partitions,
                "abstain_rate": abstain_rate,
                "locality_veto_rate": locality_veto_rate,
                "route_summary": {
                    "num_inference_events": len(inference_rows),
                    "num_with_overlap": len(support_overlaps),
                    "num_with_code_sparsity": len(support_sizes),
                    "num_abstained": 0 if abstain_rate is None else int(round(abstain_rate * len(inference_rows))),
                    "num_locality_vetoed": 0 if locality_veto_rate is None else int(round(locality_veto_rate * len(inference_rows))),
                },
                "state_summary": [],
            }
        if self.is_v3:
            support = self.export_support_diagnostics()
            realization = self.export_realization_diagnostics()
            shard = self.export_shard_diagnostics()
            residual_values = [float(entry["write_residual"]) for entry in self.memory_entries if entry.get("write_residual") is not None]
            residual_mean = None if not residual_values else float(sum(residual_values) / len(residual_values))
            return {
                "applicable": True,
                "state_margin_mean": shard.get("shard_prototype_margin_mean"),
                "cross_view_gap_mean": None,
                "realization_overlap_mean": realization.get("off_target_realization_energy_mean"),
                "code_sparsity_mean": None,
                "support_size_mean": support.get("support_size_mean"),
                "support_overlap_mean": support.get("support_overlap_mean"),
                "atom_coherence_mean": None,
                "code_l1_mean": None,
                "code_l2_mean": None,
                "factor_space_residual_mean": residual_mean,
                "residual_after_sparse_pursuit_mean": residual_mean,
                "route_summary": {
                    "num_inference_events": len([row for row in self.route_logs if row.get("route_event") == "inference"]),
                    "num_with_overlap": len([row for row in self.route_logs if row.get("realization_overlap") is not None]),
                    "num_with_code_sparsity": 0,
                },
                "state_summary": shard.get("shard_summary", []),
            }
        if not self.is_v2 or not self._use_sparse_slots():
            return {
                "applicable": False,
                "state_margin_mean": None,
                "cross_view_gap_mean": None,
                "realization_overlap_mean": None,
                "code_sparsity_mean": None,
                "support_size_mean": None,
                "support_overlap_mean": None,
                "atom_coherence_mean": None,
                "code_l1_mean": None,
                "code_l2_mean": None,
                "factor_space_residual_mean": None,
                "residual_after_sparse_pursuit_mean": None,
                "route_summary": {},
                "state_summary": [],
            }
        state_summary = []
        residuals = []
        cross_view_gaps = []
        for cell_id in sorted(self.cell_registry.keys()):
            cell = self.cell_registry[cell_id]
            if self._use_shared_basis_codes() and cell.get("factor_space_residual_mean") is None:
                self._refresh_single_cell_statistics(cell_id, compute_factor_residuals=True)
                cell = self.cell_registry[cell_id]
            residual = cell.get("factor_space_residual_mean")
            if residual is not None:
                residuals.append(float(residual))
            gap = cell.get("cross_view_route_gap")
            if gap is not None:
                cross_view_gaps.append(float(gap))
            state_summary.append(
                {
                    "state_id": cell_id,
                    "factor_space_residual_mean": residual,
                    "residual_after_sparse_pursuit_mean": residual,
                    "cross_view_route_gap": gap,
                    "slot_count": len(self._state_slots(cell)),
                    "prototype_dispersion": cell.get("prototype_dispersion"),
                    "within_state_conflict_mean": cell.get("within_cell_conflict_mean"),
                }
            )
        inference_rows = [row for row in self.route_logs if row.get("route_event") == "inference"]
        state_margins = [float(row["state_margin"]) for row in inference_rows if row.get("state_margin") is not None]
        overlaps = [float(row["realization_overlap"]) for row in inference_rows if row.get("realization_overlap") is not None]
        code_sparsity = [float(row["selected_code_nonzero_fraction"]) for row in inference_rows if row.get("selected_code_nonzero_fraction") is not None]
        support_sizes = [float(row["selected_code_support"]) for row in inference_rows if row.get("selected_code_support") is not None]
        support_overlaps = [float(row["selected_support_overlap"]) for row in inference_rows if row.get("selected_support_overlap") is not None]
        atom_coherences = [float(row["selected_atom_coherence"]) for row in inference_rows if row.get("selected_atom_coherence") is not None]
        code_l1 = [float(row["selected_code_l1_mean"]) for row in inference_rows if row.get("selected_code_l1_mean") is not None]
        code_l2 = [float(row["selected_code_l2_mean"]) for row in inference_rows if row.get("selected_code_l2_mean") is not None]
        return {
            "applicable": True,
            "state_margin_mean": None if not state_margins else float(sum(state_margins) / len(state_margins)),
            "cross_view_gap_mean": None if not cross_view_gaps else float(sum(cross_view_gaps) / len(cross_view_gaps)),
            "realization_overlap_mean": None if not overlaps else float(sum(overlaps) / len(overlaps)),
            "code_sparsity_mean": None if not code_sparsity else float(sum(code_sparsity) / len(code_sparsity)),
            "support_size_mean": None if not support_sizes else float(sum(support_sizes) / len(support_sizes)),
            "support_overlap_mean": None if not support_overlaps else float(sum(support_overlaps) / len(support_overlaps)),
            "atom_coherence_mean": None if not atom_coherences else float(sum(atom_coherences) / len(atom_coherences)),
            "code_l1_mean": None if not code_l1 else float(sum(code_l1) / len(code_l1)),
            "code_l2_mean": None if not code_l2 else float(sum(code_l2) / len(code_l2)),
            "factor_space_residual_mean": None if not residuals else float(sum(residuals) / len(residuals)),
            "residual_after_sparse_pursuit_mean": None if not residuals else float(sum(residuals) / len(residuals)),
            "route_summary": {
                "num_inference_events": len(inference_rows),
                "num_with_overlap": len(overlaps),
                "num_with_code_sparsity": len(code_sparsity),
            },
            "state_summary": state_summary,
        }

    def export_state_diagnostics(self) -> dict[str, Any]:
        if self._use_sparse_address_trace_bank():
            return self.export_shard_diagnostics()
        if self.is_v3:
            return self.export_shard_diagnostics()
        if not self.is_v2:
            return {"applicable": False, "state_summary": []}
        return self.export_stability_diagnostics()

    def export_prototype_diagnostics(self) -> dict[str, Any]:
        if self._use_sparse_address_trace_bank():
            support = self.export_support_diagnostics()
            return {
                "applicable": True,
                "memory_unit": "trace",
                "num_cells": len(self.address_dictionary.get("atoms") or []),
                "prototype_strategy": "sparse_multiview_address",
                "prototype_dispersion_mean": None
                if support.get("cross_view_code_agreement_mean") is None
                else float(1.0 - support["cross_view_code_agreement_mean"]),
                "max_intra_cell_prototype_conflict": self.address_dictionary.get("coherence_mean"),
                "cell_summary": [],
            }
        if self.is_v3:
            shard_summary = self.export_shard_diagnostics()
            return {
                "applicable": True,
                "memory_unit": "shard",
                "num_cells": len(self.shard_registry),
                "prototype_strategy": "side_memory_shards",
                "prototype_dispersion_mean": None
                if not shard_summary.get("shard_summary")
                else float(
                    sum(
                        float(row.get("prototype_dispersion") or 0.0)
                        for row in shard_summary["shard_summary"]
                    )
                    / len(shard_summary["shard_summary"])
                ),
                "max_intra_cell_prototype_conflict": None,
                "cell_summary": [
                    {
                        "cell_id": row["shard_id"],
                        "member_count": row["member_count"],
                        "prototype_count_by_view": {},
                        "prototype_dispersion": row.get("prototype_dispersion"),
                        "max_intra_cell_prototype_conflict": None,
                        "prototype_view_names": [],
                        "slot_count": row.get("occupancy"),
                    }
                    for row in shard_summary.get("shard_summary", [])
                ],
            }
        if not self.is_v2:
            return {
                "applicable": False,
                "memory_unit": "edit",
                "num_cells": 0,
                "prototype_dispersion_mean": None,
                "max_intra_cell_prototype_conflict": None,
                "cell_summary": [],
            }
        cell_summary = []
        dispersions = []
        max_conflicts = []
        for cell_id in sorted(self.cell_registry.keys()):
            cell = self.cell_registry[cell_id]
            dispersion = cell.get("prototype_dispersion")
            max_conflict = cell.get("max_intra_cell_prototype_conflict")
            if dispersion is not None:
                dispersions.append(float(dispersion))
            if max_conflict is not None:
                max_conflicts.append(float(max_conflict))
            cell_summary.append(
                {
                    "cell_id": cell_id,
                    "member_count": cell.get("member_count"),
                    "prototype_count_by_view": cell.get("prototype_count_by_view"),
                    "prototype_dispersion": dispersion,
                    "max_intra_cell_prototype_conflict": max_conflict,
                    "prototype_view_names": [proto.get("view_name") for proto in self._cell_view_records(cell)],
                    "slot_count": len(self._state_slots(cell)) if self._use_sparse_slots() else None,
                }
            )
        return {
            "applicable": True,
            "memory_unit": self.memory_unit,
            "num_cells": len(self.cell_registry),
            "prototype_strategy": "sparse_slots" if self._use_sparse_slots() else getattr(self.hparams, "cell_prototype_strategy", "single"),
            "prototype_dispersion_mean": None if not dispersions else float(sum(dispersions) / len(dispersions)),
            "max_intra_cell_prototype_conflict": max(max_conflicts) if max_conflicts else None,
            "cell_summary": cell_summary,
        }

    def export_stability_diagnostics(self) -> dict[str, Any]:
        if self.is_v3:
            shard_summary = self.export_shard_diagnostics()
            return {
                "applicable": True,
                "stable_state_count": len(self.shard_registry),
                "active_state_count": 0,
                "consolidation_attempts": 0,
                "accepted_merges": 0,
                "rejected_merges_by_reason": {},
                "slot_transfer_attempts": 0,
                "accepted_slot_transfers": 0,
                "rejected_slot_transfers_by_reason": {},
                "state_summary": [
                    {
                        "state_id": row["shard_id"],
                        "tier": "active",
                        "bucket_id": None,
                        "member_count": row["member_count"],
                        "state_stability_score": row.get("prototype_margin_mean"),
                        "cross_view_route_gap": None,
                        "prototype_dispersion": row.get("prototype_dispersion"),
                        "within_state_conflict": None,
                        "locality_fragility": None,
                        "state_support_observations": row.get("member_count"),
                        "state_gate_score_mean": None,
                        "is_stable": False,
                    }
                    for row in shard_summary.get("shard_summary", [])
                ],
            }
        if not self.is_v2:
            return {"applicable": False, "stable_state_count": 0, "active_state_count": 0, "state_summary": []}
        state_summary = []
        stable = 0
        active = 0
        for cell_id in sorted(self.cell_registry.keys()):
            cell = self.cell_registry[cell_id]
            if cell.get("tier") == "consolidated":
                stable += 1
            else:
                active += 1
            state_summary.append(
                {
                    "state_id": cell_id,
                    "tier": cell.get("tier"),
                    "bucket_id": cell.get("bucket_id"),
                    "member_count": cell.get("member_count"),
                    "state_stability_score": cell.get("state_stability_score"),
                    "cross_view_route_gap": cell.get("cross_view_route_gap"),
                    "prototype_dispersion": cell.get("prototype_dispersion"),
                    "within_state_conflict": cell.get("within_cell_conflict_mean"),
                    "locality_fragility": cell.get("locality_fragility"),
                    "state_support_observations": cell.get("state_support_observations"),
                    "state_gate_score_mean": cell.get("state_gate_score_mean"),
                    "is_stable": cell.get("is_stable"),
                }
            )
        return {
            "applicable": True,
            "stable_state_count": stable,
            "active_state_count": active,
            "consolidation_attempts": int(self.consolidation_attempts),
            "accepted_merges": int(self.consolidation_accepted),
            "rejected_merges_by_reason": dict(self.consolidation_rejected_by_reason),
            "slot_transfer_attempts": int(self.slot_transfer_attempts),
            "accepted_slot_transfers": int(self.slot_transfer_accepted),
            "rejected_slot_transfers_by_reason": dict(self.slot_transfer_rejected_by_reason),
            "state_summary": state_summary,
        }

    def export_gate_diagnostics(self) -> dict[str, Any]:
        if not self.is_v2:
            return {"applicable": False, "gate_enabled": False}
        if not self._state_gate_enabled():
            return {"applicable": True, "gate_enabled": False}

        bucket_rows = []
        calibration_bins = [
            {"low": 0.0, "high": 0.2, "scores": [], "labels": []},
            {"low": 0.2, "high": 0.4, "scores": [], "labels": []},
            {"low": 0.4, "high": 0.6, "scores": [], "labels": []},
            {"low": 0.6, "high": 0.8, "scores": [], "labels": []},
            {"low": 0.8, "high": 1.01, "scores": [], "labels": []},
        ]
        for row in self.state_gate_replay_buffer:
            features = row.get("features")
            label = row.get("label")
            if not isinstance(features, torch.Tensor) or label is None:
                continue
            normalized = self._normalize_state_gate_features(features.unsqueeze(0))
            with torch.no_grad():
                score = float(torch.sigmoid(self.state_gate_module(normalized))[0, 0].item())
            for bucket in calibration_bins:
                if bucket["low"] <= score < bucket["high"]:
                    bucket["scores"].append(score)
                    bucket["labels"].append(float(label))
                    break
        for bucket in calibration_bins:
            scores = bucket.pop("scores")
            labels = bucket.pop("labels")
            bucket_rows.append(
                {
                    "score_range": [bucket["low"], min(1.0, bucket["high"])],
                    "count": len(scores),
                    "mean_score": None if not scores else float(sum(scores) / len(scores)),
                    "empirical_positive_rate": None if not labels else float(sum(labels) / len(labels)),
                }
            )
        return {
            "applicable": True,
            "gate_enabled": True,
            "gate_ready": bool(self.state_gate_ready),
            "gate_model": getattr(self.hparams, "state_gate_model", "logistic"),
            "num_gate_examples": int(self.state_gate_seen_examples),
            "warm_start_source": list(self.state_gate_warm_start_source),
            "online_updates_performed": int(self.state_gate_online_updates),
            "direct_accepted_count": int(self.state_gate_runtime_counts.get("direct_accepted_count", 0)),
            "rerank_triggered_count": int(self.state_gate_runtime_counts.get("rerank_triggered_count", 0)),
            "direct_win_count": int(self.state_gate_runtime_counts.get("direct_win_count", 0)),
            "rerank_win_count": int(self.state_gate_runtime_counts.get("rerank_win_count", 0)),
            "calibration": bucket_rows,
        }

    def export_hierarchy_diagnostics(self) -> dict[str, Any]:
        if not self.is_v2:
            return {"applicable": False, "bucket_count": 0, "bucket_summary": []}
        bucket_summary = []
        for bucket_id in sorted(self.bucket_registry.keys()):
            bucket = self.bucket_registry[bucket_id]
            bucket_summary.append(
                {
                    "bucket_id": bucket_id,
                    "state_count": len(bucket.get("state_ids", [])),
                    "bucket_dispersion": bucket.get("bucket_dispersion"),
                    "prototype_view_names": [proto.get("view_name") for proto in bucket.get("bucket_prototypes", [])],
                }
            )
        return {
            "applicable": True,
            "bucket_count": len(self.bucket_registry),
            "bucket_size_distribution": [row["state_count"] for row in bucket_summary],
            "bucket_summary": bucket_summary,
        }

    def _active_runtime_adapter_names(self) -> list[str]:
        if self.is_v3:
            return []
        if self.is_v2:
            if self._use_sparse_slots():
                return []
            return [
                cell["adapter_name"]
                for cell in self._active_cells()
                if cell.get("adapter_name") is not None
            ]
        if self._use_cold_trace_values():
            return list(self.trace_value_cache.values())
        return [entry.get("edit_id") for entry in self.memory_entries if entry.get("edit_id") and entry.get("edit_id") not in self.disabled_adapters]

    def runtime_state_dict(self) -> dict[str, Any]:
        active_adapter = None
        if isinstance(self.model, PeftModel):
            active_adapter = getattr(self.model, "active_adapter", None)
        cell_adapter_weights = None
        if self.is_v2 and not self._use_sparse_slots():
            cell_adapter_weights = {
                cell["adapter_name"]: self._capture_adapter_parameters(cell["adapter_name"])
                for cell in self._active_cells()
                if cell.get("adapter_name") is not None
            }
        return {
            "format_version": 5 if self._use_cold_trace_values() else (4 if self.is_v4 else (3 if self.is_v3 else (2 if self.is_v2 else 1))),
            "hopedit_mode": self.hparams.hopedit_mode,
            "state_memory_impl": getattr(self.hparams, "state_memory_impl", "cell_bank"),
            "slot_realization_impl": getattr(self.hparams, "slot_realization_impl", "concatenated_lora"),
            "state_basis_rank": int(getattr(self.hparams, "state_basis_rank", self._slot_rank()) or self._slot_rank()),
            "edit_index": self.edit_index,
            "cell_index": self.cell_index,
            "shard_index": self.shard_index,
            "atom_index": self.atom_index,
            "disabled_adapters": sorted(self.disabled_adapters),
            "memory_entries": self._clone_to_cpu(self.memory_entries),
            "route_logs": self._clone_to_cpu(self.route_logs),
            "cached_activation_stats": self._clone_to_cpu(self.cached_activation_stats),
            "cell_registry": self._clone_to_cpu(self.cell_registry),
            "shard_registry": self._clone_to_cpu(self.shard_registry),
            "bucket_registry": self._clone_to_cpu(self.bucket_registry),
            "address_dictionary": self._clone_to_cpu(self.address_dictionary),
            "address_postings": self._clone_to_cpu(self.address_postings),
            "family_postings": self._clone_to_cpu(self.family_postings),
            "address_version": int(self.address_version),
            "active_adapter_names": self._active_runtime_adapter_names(),
            "active_adapter": active_adapter,
            "cell_adapter_weights": self._clone_to_cpu(cell_adapter_weights),
            "trace_value_cache_keys": list(self.trace_value_cache.keys()),
            "trace_value_cache_hits": int(self.trace_value_cache_hits),
            "trace_value_cache_misses": int(self.trace_value_cache_misses),
            "factored_relation_encoder_input_dim": self.factored_relation_encoder_input_dim,
            "factored_relation_encoder_state": None
            if self.factored_relation_encoder is None
            else self._clone_to_cpu(self.factored_relation_encoder.state_dict()),
            "factored_relation_encoder_updates": int(self.factored_relation_encoder_updates),
            "factored_relation_encoder_last_loss": self.factored_relation_encoder_last_loss,
            "factored_relation_encoder_checkpoint_loaded": self.factored_relation_encoder_checkpoint_loaded,
            "factored_relation_encoder_checkpoint_metadata": self._clone_to_cpu(self.factored_relation_encoder_checkpoint_metadata),
            "state_gate_state": self._clone_to_cpu(self.state_gate_module.state_dict()),
            "state_gate_optimizer_state": self._clone_to_cpu(self.state_gate_optimizer.state_dict()),
            "state_gate_feature_stats": self._clone_to_cpu(self.state_gate_feature_stats),
            "state_gate_replay_buffer": self._clone_to_cpu(self.state_gate_replay_buffer),
            "state_gate_seen_examples": int(self.state_gate_seen_examples),
            "state_gate_online_updates": int(self.state_gate_online_updates),
            "state_gate_warm_start_source": list(self.state_gate_warm_start_source),
            "state_gate_runtime_counts": dict(self.state_gate_runtime_counts),
            "state_gate_ready": bool(self.state_gate_ready),
            "slot_transfer_attempts": int(self.slot_transfer_attempts),
            "slot_transfer_accepted": int(self.slot_transfer_accepted),
            "slot_transfer_rejected_by_reason": dict(self.slot_transfer_rejected_by_reason),
            "support_exclusivity_failures": int(self.support_exclusivity_failures),
            "base_only_fallback_count": int(self.base_only_fallback_count),
        }

    def save_runtime_checkpoint(self, checkpoint_dir: str) -> str:
        os.makedirs(checkpoint_dir, exist_ok=True)
        adapters_dir = os.path.join(checkpoint_dir, "adapters")
        if os.path.isdir(adapters_dir):
            shutil.rmtree(adapters_dir)
        values_path = os.path.join(checkpoint_dir, "trace_values.pt")
        if os.path.exists(values_path):
            os.remove(values_path)
        if self._use_cold_trace_values():
            torch.save(self._clone_to_cpu(self.trace_value_store), values_path)
        elif not self.is_v2 and not self.is_v3:
            os.makedirs(adapters_dir, exist_ok=True)
            active_adapters = self._active_runtime_adapter_names()
            if isinstance(self.model, PeftModel) and active_adapters:
                try:
                    self.model.save_pretrained(adapters_dir, selected_adapters=active_adapters)
                except TypeError:
                    self.model.save_pretrained(adapters_dir)

        torch.save(self.runtime_state_dict(), os.path.join(checkpoint_dir, "controller_state.pt"))
        return checkpoint_dir

    def _rebuild_auxiliary_state(self, *, refresh_keys: bool = True) -> None:
        self.edit_registry = {
            entry["edit_id"]: entry
            for entry in self.memory_entries
            if isinstance(entry, dict) and entry.get("edit_id") is not None
        }
        if self.is_v3:
            for shard_id in list(self.shard_registry.keys()):
                self._refresh_single_shard_metadata(shard_id)
        if self.is_v2 and not self.cell_registry:
            rebuilt = {}
            for entry in self.memory_entries:
                cell_id = entry.get("cell_id")
                if cell_id is None:
                    continue
                cell = rebuilt.setdefault(
                    cell_id,
                    {
                        "cell_id": cell_id,
                        "adapter_name": cell_id,
                        "member_edit_ids": [],
                        "member_count": 0,
                        "semantic_key": None,
                        "activation_key": None,
                        "within_cell_conflict_mean": None,
                        "within_cell_conflict_max": None,
                        "cell_prototypes": [],
                        "prototype_stats": {},
                        "prototype_dispersion": None,
                        "prototype_count_by_view": {},
                        "max_intra_cell_prototype_conflict": None,
                        "prototype_anchor_text": None,
                        "state_stability_score": None,
                        "cross_view_route_gap": None,
                        "locality_fragility": None,
                        "locality_proxy": None,
                        "tier": "active",
                        "bucket_id": None,
                        "is_stable": False,
                        "state_support_observations": 0,
                        "state_age_edits": 0,
                        "created_at_edit_index": 0,
                        "state_gate_recent_scores": [],
                        "state_gate_score_mean": None,
                        "state_shared_basis": {},
                    },
                )
                cell["member_edit_ids"].append(entry["edit_id"])
                cell["member_count"] = len(cell["member_edit_ids"])
            self.cell_registry = rebuilt
        if refresh_keys:
            self._refresh_processed_memory_keys()
        if self.is_v2:
            has_materialized_state = bool(self.cell_registry) and all(
                isinstance(cell, dict)
                and "cell_prototypes" in cell
                and "tier" in cell
                and "state_stability_score" in cell
                for cell in self.cell_registry.values()
            )
            if has_materialized_state:
                if self._hierarchy_enabled() and not self.bucket_registry:
                    self._rebuild_bucket_registry()
            else:
                self._recompute_cell_statistics()

    def load_runtime_checkpoint(self, checkpoint_dir: str, is_trainable: bool = True) -> None:
        state_path = os.path.join(checkpoint_dir, "controller_state.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Missing HopEdit controller checkpoint: {state_path}")

        state = torch.load(state_path, map_location="cpu")
        state_mode = state.get("hopedit_mode", getattr(self.hparams, "hopedit_mode", "v1_per_edit"))
        self.hparams.state_memory_impl = state.get("state_memory_impl", getattr(self.hparams, "state_memory_impl", "cell_bank"))
        self.hparams.slot_realization_impl = state.get("slot_realization_impl", getattr(self.hparams, "slot_realization_impl", "concatenated_lora"))
        self.hparams.state_basis_rank = int(state.get("state_basis_rank", getattr(self.hparams, "state_basis_rank", self._slot_rank())) or self._slot_rank())
        adapters_dir = os.path.join(checkpoint_dir, "adapters")
        adapter_names = list(state.get("active_adapter_names") or [])
        if state_mode == "v2_cell_bank":
            cell_adapter_weights = state.get("cell_adapter_weights") or {}
            if not self._use_sparse_slots():
                for adapter_name in adapter_names:
                    self._ensure_adapter(adapter_name)
                    weights = cell_adapter_weights.get(adapter_name) or {}
                    self._load_adapter_parameters(adapter_name, weights)
                active_adapter = state.get("active_adapter") or (adapter_names[-1] if adapter_names else None)
                if active_adapter is not None and hasattr(self.model, "set_adapter") and active_adapter in getattr(self.model, "peft_config", {}):
                    self.model.set_adapter(active_adapter)
        elif adapter_names and os.path.isdir(adapters_dir):
            first_adapter = adapter_names[0]
            if not isinstance(self.model, PeftModel):
                self.model = PeftModel.from_pretrained(
                    self.model,
                    adapters_dir,
                    adapter_name=first_adapter,
                    is_trainable=is_trainable,
                    subfolder=first_adapter,
                )
            elif first_adapter not in getattr(self.model, "peft_config", {}):
                self.model.load_adapter(
                    adapters_dir,
                    first_adapter,
                    is_trainable=is_trainable,
                    subfolder=first_adapter,
                )

            for adapter_name in adapter_names[1:]:
                if adapter_name in getattr(self.model, "peft_config", {}):
                    continue
                self.model.load_adapter(
                    adapters_dir,
                    adapter_name,
                    is_trainable=is_trainable,
                    subfolder=adapter_name,
                )

            active_adapter = state.get("active_adapter") or adapter_names[-1]
            if hasattr(self.model, "set_adapter") and active_adapter in getattr(self.model, "peft_config", {}):
                self.model.set_adapter(active_adapter)

        self.edit_index = int(state.get("edit_index", len(adapter_names)))
        self.cell_index = int(state.get("cell_index", 0))
        self.shard_index = int(state.get("shard_index", self.shard_index))
        self.atom_index = int(state.get("atom_index", self.atom_index))
        self.disabled_adapters = set(state.get("disabled_adapters") or [])
        self.memory_entries = state.get("memory_entries") or []
        self.route_logs = state.get("route_logs") or []
        self.cached_activation_stats = state.get("cached_activation_stats")
        self.cell_registry = state.get("cell_registry") or {}
        self.shard_registry = state.get("shard_registry") or {}
        self.bucket_registry = state.get("bucket_registry") or {}
        self.address_dictionary = state.get("address_dictionary") or {"atoms": [], "usage_counts": []}
        self.address_postings = state.get("address_postings") or {}
        self.family_postings = state.get("family_postings") or {}
        self.address_version = int(state.get("address_version", self.address_version))
        self.trace_value_cache = OrderedDict()
        self.trace_value_cache_hits = int(state.get("trace_value_cache_hits", 0))
        self.trace_value_cache_misses = int(state.get("trace_value_cache_misses", 0))
        relation_encoder_state = state.get("factored_relation_encoder_state")
        relation_encoder_input_dim = state.get("factored_relation_encoder_input_dim")
        self.factored_relation_encoder_updates = int(state.get("factored_relation_encoder_updates", 0))
        self.factored_relation_encoder_last_loss = state.get("factored_relation_encoder_last_loss")
        self.factored_relation_encoder_checkpoint_loaded = state.get("factored_relation_encoder_checkpoint_loaded")
        self.factored_relation_encoder_checkpoint_metadata = state.get("factored_relation_encoder_checkpoint_metadata") or {}
        if relation_encoder_state and relation_encoder_input_dim:
            encoder = self._ensure_factored_relation_encoder(int(relation_encoder_input_dim))
            if encoder is not None:
                encoder.load_state_dict(relation_encoder_state)
                encoder.eval()
        if self._use_cold_trace_values():
            values_path = os.path.join(checkpoint_dir, "trace_values.pt")
            self.trace_value_store = torch.load(values_path, map_location="cpu") if os.path.exists(values_path) else {}
        else:
            self.trace_value_store = {}
        gate_state = state.get("state_gate_state") or {}
        if gate_state:
            self.state_gate_module.load_state_dict(gate_state)
        gate_optimizer_state = state.get("state_gate_optimizer_state") or {}
        if gate_optimizer_state:
            self.state_gate_optimizer.load_state_dict(gate_optimizer_state)
        self.state_gate_feature_stats = state.get("state_gate_feature_stats") or self.state_gate_feature_stats
        self.state_gate_replay_buffer = state.get("state_gate_replay_buffer") or []
        self.state_gate_seen_examples = int(state.get("state_gate_seen_examples", self.state_gate_seen_examples))
        self.state_gate_online_updates = int(state.get("state_gate_online_updates", self.state_gate_online_updates))
        self.state_gate_warm_start_source = list(state.get("state_gate_warm_start_source") or [])
        self.state_gate_runtime_counts = dict(state.get("state_gate_runtime_counts") or self.state_gate_runtime_counts)
        self.state_gate_ready = bool(state.get("state_gate_ready", self.state_gate_ready))
        self.slot_transfer_attempts = int(state.get("slot_transfer_attempts", self.slot_transfer_attempts))
        self.slot_transfer_accepted = int(state.get("slot_transfer_accepted", self.slot_transfer_accepted))
        self.slot_transfer_rejected_by_reason = dict(state.get("slot_transfer_rejected_by_reason") or self.slot_transfer_rejected_by_reason)
        self.support_exclusivity_failures = int(state.get("support_exclusivity_failures", self.support_exclusivity_failures))
        self.base_only_fallback_count = int(state.get("base_only_fallback_count", self.base_only_fallback_count))
        self.hparams.hopedit_mode = state_mode
        preserve_overlap_address_state = bool(
            (self._use_overlap_aware_anchor_trace_bank() or self._use_factored_address_trace_bank())
            and self.address_dictionary.get("atoms")
            and (self.address_postings or not self.memory_entries)
        )
        self._rebuild_auxiliary_state(refresh_keys=not preserve_overlap_address_state)


def load_hopedit_runtime_checkpoint(
    model: AutoModelForCausalLM | HopEditController,
    tok: AutoTokenizer,
    hparams: HopEditHyperParams,
    checkpoint_dir: str,
    is_trainable: bool = True,
) -> HopEditController:
    controller = model if isinstance(model, HopEditController) else HopEditController(model=model, tok=tok, hparams=hparams)
    controller.load_runtime_checkpoint(checkpoint_dir, is_trainable=is_trainable)
    return controller


def apply_hopedit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: list[dict[str, Any]],
    hparams: HopEditHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    keep_original_weight: bool = False,
    **kwargs: Any,
) -> tuple[HopEditController, Any]:
    if isinstance(model, HopEditController):
        controller = model
    else:
        controller = HopEditController(model=model, tok=tok, hparams=hparams)

    added_edit_ids = []
    for request in requests:
        print(
            f"Executing HOPEDIT ({hparams.hopedit_mode}) for: "
            f"[{controller._format_prompt(request)}] -> [{request['target_new']}]"
        )
        added_edit_ids.append(controller.add_edit(request))

    def reset_new_edits():
        for edit_id in reversed(added_edit_ids):
            controller.rollback_edit(edit_id)

    return controller, reset_new_edits
