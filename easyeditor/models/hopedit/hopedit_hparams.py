from dataclasses import dataclass
from typing import List, Optional
import yaml

from ...util.hparams import HyperParams


@dataclass
class HopEditHyperParams(HyperParams):
    rank: int
    lora_alpha: float
    lora_dropout: float
    target_modules: List[str]
    num_steps: int
    lr: float
    weight_decay: float
    top_k: int
    beta: float
    semantic_weight: float
    activation_weight: float
    probe_layer: int
    route_log_dir: str
    device: int
    alg_name: str
    model_name: str
    probe_layers: Optional[List[int]] = None
    route_strategy: str = "multiview"
    min_route_prob: float = 0.0
    min_route_margin: float = 0.0
    fallback_min_route_prob: Optional[float] = None
    fallback_min_route_margin: Optional[float] = None
    activation_center: bool = True
    activation_whiten: bool = True
    key_l2_normalize: bool = True
    use_rephrase_prompt: bool = True
    use_subject_prompt: bool = True
    negative_top_k: int = 0
    negative_weight: float = 0.0
    negative_margin: float = 0.2
    negative_min_conflict: float = 0.0
    log_training_loss: bool = False
    loss_log_every: int = 1
    batch_size: int = 1
    max_length: int = 512
    model_parallel: bool = False
    use_chat_template: bool = False
    fp16: bool = False
    bf16: bool = True
    save_path: str = None
    hopedit_mode: str = "v1_per_edit"
    cell_budget: int = 0
    cell_assignment_policy: str = "conflict_aware"
    cell_conflict_threshold: float = 0.65
    cell_conflict_stat: str = "mean"
    cell_merge_rule: str = "weighted_delta_average"
    cell_merge_trim_quantile: float = 0.0
    cell_router_topk: int = 4
    cell_min_assignment_margin: float = 0.0
    cell_warmup_edits: int = 0
    cell_metadata_backend: str = "inline_json"
    cell_prototype_strategy: str = "single"
    cell_prototype_rerank_topk: int = 4
    cell_rerank_dispersion_penalty: float = 0.05
    cell_direct_accept_min_prob: float = 0.52
    cell_direct_accept_min_margin: float = 0.05
    state_memory_impl: str = "cell_bank"
    state_slot_capacity: int = 8
    slot_rank: int = 2
    state_basis_rank: int = 8
    slot_realization_impl: str = "concatenated_lora"
    slot_topk: int = 2
    atom_sparse_topk: int = 2
    atom_write_topk: Optional[int] = None
    atom_coherence_penalty: float = 0.25
    atom_support_exclusivity_penalty: float = 0.35
    atom_sparse_ridge: float = 1.0e-4
    atom_sparse_min_abs_affinity: float = 1.0e-6
    slot_activation_min_weight: float = 0.15
    state_finalist_margin: float = 0.05
    locality_risk_threshold: float = 0.35
    state_gate_enable: bool = False
    state_gate_model: str = "logistic"
    state_gate_warm_start_from_replay: bool = True
    state_gate_replay_paths: Optional[List[str]] = None
    state_gate_replay_buffer_size: int = 4096
    state_gate_online_update_interval: int = 16
    state_gate_lr: float = 1.0e-2
    state_gate_batch_size: int = 128
    state_gate_warm_start_epochs: int = 8
    state_gate_direct_threshold: float = 0.70
    state_gate_promotion_threshold: float = 0.80
    state_gate_merge_threshold: float = 0.75
    memory_tier_warmup_edits: int = 256
    stability_min_observations: int = 8
    stability_max_cross_view_gap: float = 0.10
    stability_max_prototype_dispersion: float = 0.30
    stability_max_within_state_conflict: float = 0.30
    stability_min_locality: float = 0.95
    consolidation_interval_edits: int = 16
    consolidation_max_pairs_per_pass: int = 16
    hierarchy_enable: bool = True
    hierarchy_start_edit: int = 1024
    bucket_topk: int = 2
    bucket_max_size: int = 256
    bucket_split_dispersion: float = 0.35
    atoms_per_shard: int = 32
    atom_rank: int = 4
    read_support_topk: int = 2
    write_support_topk: int = 2
    max_new_atoms_per_edit: int = 1
    module_grouping: str = "attention"
    side_memory_shard_shortlist: int = 2
    side_memory_energy_margin: float = 0.05
    support_usage_penalty: float = 0.35
    support_overlap_veto: float = 0.75
    new_shard_residual_threshold: float = 1.0
    side_memory_write_steps: int = 2
    side_memory_loss_threshold: float = 2.5
    base_only_fallback_threshold: float = 0.0
    shard_budget_microbench: int = 32
    address_encoder_impl: str = "deterministic_topk"
    address_num_atoms: int = 256
    address_code_topk: int = 4
    address_candidate_budget: int = 64
    address_atom_merge_threshold: float = 0.92
    address_code_min_affinity: float = 1.0e-6
    address_coherence_penalty: float = 0.05
    trace_energy_beta: float = 12.0
    trace_energy_temperature: float = 1.0
    trace_abstain_margin: float = 0.03
    trace_abstain_min_energy: float = 0.10
    trace_use_calibrated_thresholds: bool = True
    trace_calibration_negative_topk: int = 8
    trace_calibration_energy_blend: float = 0.60
    trace_calibration_margin_scale: float = 0.50
    trace_calibration_min_margin: float = 0.03
    trace_shape_energy_relax_scale: float = 0.35
    trace_shape_margin_relax_scale: float = 0.20
    trace_shape_agreement_weight: float = 0.50
    trace_use_locality_veto: bool = True
    trace_locality_margin_scale: float = 0.50
    trace_locality_min_margin: float = 0.02
    trace_locality_relax_scale: float = 0.10
    trace_exclusion_code_topk: int = 8
    trace_exclusion_threshold_blend: float = 0.50
    trace_exclusion_relax_scale: float = 0.05
    trace_family_budget: int = 6
    trace_family_boost: float = 0.35
    trace_anchor_weight: float = 0.25
    trace_anchor_energy_floor: float = 0.05
    trace_family_negative_topk: int = 8
    trace_irrelevant_negative_topk: int = 4
    trace_family_margin_scale: float = 0.60
    trace_anchor_margin_scale: float = 0.60
    trace_value_cache_size: int = 8
    factored_address_layer: Optional[int] = None
    factored_subject_layer: Optional[int] = None
    factored_relation_layer: Optional[int] = None
    factored_subject_weight: float = 0.5
    factored_relation_weight: float = 0.5
    factored_subject_margin_threshold: float = 0.03
    factored_relation_margin_threshold: float = 0.03
    factored_subject_energy_threshold: float = 0.0
    factored_relation_energy_threshold: float = 0.0
    factored_use_subject_metadata: bool = True
    factored_subject_resolution: str = "metadata_or_substring"
    factored_subject_pooling: str = "last"
    log_full_factor_scores: bool = False
    factored_relation_encoder_impl: str = "identity"
    factored_relation_encoder_hidden_dim: int = 512
    factored_relation_encoder_steps: int = 0
    factored_relation_encoder_lr: float = 1.0e-3
    factored_relation_encoder_temperature: float = 0.05
    factored_relation_encoder_relation_weight: float = 0.25
    factored_relation_encoder_min_examples: int = 4
    factored_relation_encoder_rebuild_on_train: bool = True
    factored_relation_encoder_checkpoint: Optional[str] = None
    factored_relation_encoder_freeze_checkpoint: bool = True
    factored_relation_match_rule: str = "top1_same_trace"
    factored_relation_exclude_same_relation_id_from_margin: bool = True
    factored_relation_storage_transform: str = "identity"
    factored_relation_score_transform: str = "identity"
    factored_relation_whiten_eps: float = 1.0e-4
    factored_relation_whiten_min_traces: int = 2
    factored_relation_streaming_pc_rank: int = 8
    factored_relation_streaming_pc_min_traces: int = 8
    factored_relation_streaming_pc_eps: float = 1.0e-6
    factored_capsule_enable: bool = False
    factored_capsule_config_path: Optional[str] = None
    factored_capsule_score_family: str = "min_z"
    factored_capsule_theta_accept: Optional[float] = None
    factored_capsule_conflict_margin: float = 0.0
    factored_capsule_feature_weights: Optional[List[float]] = None
    factored_capsule_feature_bias: float = 0.0

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, 'r') as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'HOPEDIT'), (
            f'HopEditHyperParams can not load from {hparams_name_or_path}. '
            f"alg_name is {config['alg_name']}"
        )
        return cls(**config)
