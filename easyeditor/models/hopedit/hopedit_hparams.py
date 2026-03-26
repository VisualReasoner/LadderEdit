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
    batch_size: int = 1
    max_length: int = 512
    model_parallel: bool = False
    use_chat_template: bool = False
    fp16: bool = False
    bf16: bool = True
    save_path: str = None

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
