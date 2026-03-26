from __future__ import annotations

import json
import os
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hopedit_hparams import HopEditHyperParams


class HopEditController(nn.Module):
    def __init__(self, model: AutoModelForCausalLM, tok: AutoTokenizer, hparams: HopEditHyperParams):
        super().__init__()
        self.model = model
        self.tok = tok
        self.hparams = hparams
        self.memory_entries: list[dict[str, Any]] = []
        self.edit_registry: dict[str, dict[str, Any]] = {}
        self.route_logs: list[dict[str, Any]] = []
        self.disabled_adapters: set[str] = set()
        self.edit_index = 0
        if self.hparams.route_log_dir:
            os.makedirs(self.hparams.route_log_dir, exist_ok=True)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

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

    def _adapter_disabled(self):
        if hasattr(self.model, "disable_adapter"):
            return self.model.disable_adapter()
        return nullcontext()

    def _next_edit_id(self) -> str:
        edit_id = f"hopedit_{self.edit_index:05d}"
        self.edit_index += 1
        return edit_id

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

    def _make_lora_config(self) -> LoraConfig:
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.hparams.rank,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_modules=self.hparams.target_modules,
        )

    def _ensure_adapter(self, adapter_name: str) -> None:
        self.model.config.use_cache = False
        if hasattr(self.model, "supports_gradient_checkpointing"):
            self.model.supports_gradient_checkpointing = True
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        if isinstance(self.model, PeftModel):
            if adapter_name not in getattr(self.model, "peft_config", {}):
                self.model.add_adapter(adapter_name, self._make_lora_config())
        else:
            self.model = get_peft_model(self.model, self._make_lora_config(), adapter_name=adapter_name)
            self.model.is_parallelizable = True
            self.model.model_parallel = True

    def _configure_trainable_adapter(self, adapter_name: str) -> list[torch.nn.Parameter]:
        if hasattr(self.model, "set_adapter"):
            self.model.set_adapter(adapter_name)
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

    def _collect_raw_activation_keys(self, entries: list[dict[str, Any]] | None = None) -> list[torch.Tensor]:
        source_entries = self.memory_entries if entries is None else entries
        raw_keys = []
        for entry in source_entries:
            for view in self._entry_view_records(entry):
                raw_key = view.get("raw_activation_key")
                if isinstance(raw_key, torch.Tensor):
                    raw_keys.append(raw_key)
        return raw_keys

    def _refresh_processed_memory_keys(self) -> None:
        activation_stats = self._activation_stats_from_raw_keys(self._collect_raw_activation_keys())
        for entry in self.memory_entries:
            entry["semantic_key"] = self._normalize_semantic_key(entry["raw_semantic_key"])
            entry["activation_key"] = self._normalize_activation_key(entry["raw_activation_key"], activation_stats)
            for view in self._entry_view_records(entry):
                if isinstance(view.get("raw_semantic_key"), torch.Tensor):
                    view["semantic_key"] = self._normalize_semantic_key(view["raw_semantic_key"])
                if isinstance(view.get("raw_activation_key"), torch.Tensor):
                    view["activation_key"] = self._normalize_activation_key(view["raw_activation_key"], activation_stats)

    def _extract_keys(self, semantic_texts: list[str], activation_texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_tokens = self._tokenize(semantic_texts)
        with torch.no_grad():
            input_embeddings = self.model.get_input_embeddings()(semantic_tokens["input_ids"])
        semantic_key = self._mean_pool(input_embeddings, semantic_tokens.get("attention_mask")).mean(dim=0)

        activation_tokens = self._tokenize(activation_texts)
        with torch.no_grad():
            with self._adapter_disabled():
                outputs = self.model(**activation_tokens, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        probe_layers = self._resolve_probe_layers(len(hidden_states))
        pooled_layers = [
            self._mean_pool(hidden_states[layer_idx], activation_tokens.get("attention_mask")).mean(dim=0)
            for layer_idx in probe_layers
        ]
        activation_key = torch.stack(pooled_layers, dim=0).mean(dim=0)
        return semantic_key.detach().float().cpu(), activation_key.detach().float().cpu()

    def _append_route_log(self, metadata: dict[str, Any], decision: dict[str, Any]) -> None:
        entry = {
            "case_id": metadata.get("case_id"),
            "prompt": metadata.get("prompt"),
            "subject": metadata.get("subject"),
            "chosen_edit_id": decision["chosen_edit_id"],
            "top_edit_ids": decision["top_edit_ids"],
            "top_scores": decision["top_scores"],
            "top_view_names": decision.get("top_view_names", []),
            "route_margin": decision["route_margin"],
            "route_stage": decision.get("route_stage", "multiview"),
            "target_edit_id": metadata.get("target_edit_id"),
            "route_match": None,
            "route_event": metadata.get("route_event", "inference"),
        }
        if metadata.get("target_edit_id") is not None and decision["chosen_edit_id"] is not None:
            entry["route_match"] = decision["chosen_edit_id"] == metadata["target_edit_id"]
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
                "chosen_edit_id": None,
                "top_edit_ids": [],
                "top_scores": [],
                "route_margin": 0.0,
                "top_view_names": [],
                "route_stage": route_stage,
            }

        top_ranking = ranking[: min(self.hparams.top_k, len(ranking))]
        values = torch.tensor([row["combined_conflict"] for row in top_ranking], dtype=torch.float32)
        probs = F.softmax(values * self.hparams.beta, dim=0)
        top_scores = [float(score) for score in probs.tolist()]
        route_margin = top_scores[0] if len(top_scores) == 1 else top_scores[0] - top_scores[1]
        chosen_edit_id = top_ranking[0]["entry"]["edit_id"]
        if top_scores[0] < min_route_prob or route_margin < min_route_margin:
            chosen_edit_id = None
        return {
            "chosen_edit_id": chosen_edit_id,
            "top_edit_ids": [row["entry"]["edit_id"] for row in top_ranking],
            "top_scores": top_scores,
            "route_margin": float(route_margin),
            "top_view_names": [row.get("best_view_name") for row in top_ranking],
            "route_stage": route_stage,
        }

    def _route_from_keys(self, semantic_key: torch.Tensor, activation_key: torch.Tensor, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
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
        if decision["chosen_edit_id"] is not None and hasattr(self.model, "set_adapter"):
            self.model.set_adapter(decision["chosen_edit_id"])
        return decision

    def _call_model(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            return self.model(**args[0])
        if "batch" in kwargs and isinstance(kwargs["batch"], dict):
            batch = kwargs.pop("batch")
            return self.model(**batch, **kwargs)
        return self.model(*args, **kwargs)

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
        input_ids, attention_mask = self._infer_inputs(args, kwargs)
        if input_ids is None or not self.memory_entries:
            with self._adapter_disabled():
                return self._call_model(*args, **kwargs)
        decision = self._select_adapter_for_inputs(input_ids, attention_mask)
        if decision["chosen_edit_id"] is None:
            with self._adapter_disabled():
                return self._call_model(*args, **kwargs)
        return self._call_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        input_ids, attention_mask = self._infer_inputs(args, kwargs)
        if input_ids is None or not self.memory_entries:
            with self._adapter_disabled():
                return self.model.generate(*args, **kwargs)
        decision = self._select_adapter_for_inputs(input_ids, attention_mask)
        if decision["chosen_edit_id"] is None:
            with self._adapter_disabled():
                return self.model.generate(*args, **kwargs)
        return self.model.generate(*args, **kwargs)

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
        if self.hparams.use_rephrase_prompt and request.get("rephrase_prompt"):
            views.append(str(request["rephrase_prompt"]))
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
        view_records = []
        seen = set()
        candidate_views = [("prompt", prompt)]
        if self.hparams.use_rephrase_prompt and request.get("rephrase_prompt"):
            candidate_views.append(("rephrase", str(request["rephrase_prompt"])))
        if self.hparams.use_subject_prompt:
            candidate_views.append(("subject", subject_prompt))
        for view_name, text in candidate_views:
            normalized_text = " ".join(str(text).split())
            if not normalized_text or normalized_text in seen:
                continue
            seen.add(normalized_text)
            raw_semantic_key, raw_activation_key = self._extract_keys([normalized_text], [normalized_text])
            view_records.append(
                {
                    "view_name": view_name,
                    "text": normalized_text,
                    "raw_semantic_key": raw_semantic_key,
                    "raw_activation_key": raw_activation_key,
                    "semantic_key": raw_semantic_key.clone(),
                    "activation_key": raw_activation_key.clone(),
                }
            )
        return view_records

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
        for _ in range(self.hparams.num_steps):
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

            loss.backward()
            optimizer.step()

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def add_edit(self, request: dict[str, Any]) -> str:
        prompt = self._format_prompt(request)
        subject_prompt = self._subject_conditioned_prompt(request, prompt)
        raw_semantic_key, raw_activation_key = self._extract_keys([prompt], [subject_prompt])
        conflict_ranking = self._rank_conflicts_from_keys(raw_semantic_key, raw_activation_key)
        negative_entries = self._select_negative_entries(conflict_ranking)
        view_key_records = self._build_view_key_records(request, prompt, subject_prompt)

        edit_id = self._next_edit_id()
        self._ensure_adapter(edit_id)
        self._train_adapter(edit_id, request, prompt, subject_prompt, negative_entries)

        entry = {
            "edit_id": edit_id,
            "prompt": prompt,
            "subject": request.get("subject"),
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
        }
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
                "route_event": "post_edit",
            },
        )
        return edit_id

    def rollback_edit(self, edit_id: str) -> None:
        self.memory_entries = [entry for entry in self.memory_entries if entry["edit_id"] != edit_id]
        self.edit_registry.pop(edit_id, None)
        if hasattr(self.model, "delete_adapter"):
            try:
                self.model.delete_adapter(edit_id)
                return
            except Exception:
                pass
        self.disabled_adapters.add(edit_id)

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
        for entry in self.memory_entries:
            row = {
                "edit_id": entry.get("edit_id"),
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
            }
            if include_keys:
                row["semantic_key"] = entry["semantic_key"].tolist() if isinstance(entry.get("semantic_key"), torch.Tensor) else entry.get("semantic_key")
                row["activation_key"] = entry["activation_key"].tolist() if isinstance(entry.get("activation_key"), torch.Tensor) else entry.get("activation_key")
            snapshot.append(row)
        return snapshot


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
        print(f"Executing HOPEDIT algo for: [{controller._format_prompt(request)}] -> [{request['target_new']}]")
        added_edit_ids.append(controller.add_edit(request))

    def reset_new_edits():
        for edit_id in reversed(added_edit_ids):
            controller.rollback_edit(edit_id)

    return controller, reset_new_edits
