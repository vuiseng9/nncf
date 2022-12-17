"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data

from nncf import NNCFConfig
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.nncf_network import NNCFNetwork
from nncf.experimental.torch.sparsity.movement.scheduler import MovementSchedulerParams
from tests.torch.sparsity.movement.helpers.config import MovementAlgoConfig

from datasets import Dataset  # pylint: disable=no-name-in-module
from transformers import AutoModelForAudioClassification
from transformers import AutoModelForImageClassification
from transformers import AutoModelForSequenceClassification
from transformers import BertConfig
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from transformers import SwinConfig
from transformers import Wav2Vec2Config


@dataclass
class TransformerBlockInfo:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    dim_per_head: int


@dataclass
class TransformerBlockItem:
    mhsa_q: Any
    mhsa_k: Any
    mhsa_v: Any
    mhsa_o: Any
    ffn_i: Any
    ffn_o: Any


class BaseMockRunRecipe(ABC):
    model_family: str
    supports_structured_masking: bool
    default_model_config = PretrainedConfig()
    default_algo_config = MovementAlgoConfig()

    def __init__(self, model_config: Optional[PretrainedConfig] = None,
                 algo_config: Optional[MovementAlgoConfig] = None,
                 log_dir=None) -> None:
        self.model_config = model_config or deepcopy(self.default_model_config)
        self.algo_config = algo_config or deepcopy(self.default_algo_config)
        self.set_log_dir(log_dir)

    @property
    @abstractmethod
    def model_input_info(self) -> List[ModelInputInfo]:
        pass

    @property
    @abstractmethod
    def transformer_block_info(self) -> List[TransformerBlockInfo]:
        pass

    @property
    def scheduler_params(self) -> MovementSchedulerParams:
        return self.algo_config.scheduler_params

    def model(self) -> torch.nn.Module:
        torch_model = self._create_model()
        g = torch.Generator()
        g.manual_seed(42)
        with torch.no_grad():
            for _, parameter in torch_model.named_parameters():
                parameter.normal_(generator=g)
        return torch_model

    def nncf_config(self) -> NNCFConfig:
        config_dict = {
            'input_info': self._get_model_input_info_dict(self.model_input_info),
            'compression': self.algo_config.to_dict()}
        if self.log_dir is not None:
            config_dict['log_dir'] = str(self.log_dir)
        return NNCFConfig.from_dict(config_dict)

    def set_log_dir(self, log_dir=None):
        if log_dir is None:
            self.log_dir = None
        else:
            self.log_dir = str(log_dir)
            Path(log_dir).mkdir(exist_ok=True, parents=True)

    def generate_mock_dataset(self, num_samples: int = 16,
                              float_low: float = -1., float_high: float = 1.,
                              int_low: int = 0, int_high: int = 2,
                              seed: int = 42) -> Dataset:
        g = torch.Generator()
        g.manual_seed(seed)
        input_dict = {}
        for input_info in self.model_input_info:
            shape = list(input_info.shape)
            keyword = input_info.keyword
            if input_info.type == torch.float32:
                tensor = torch.rand((num_samples, *shape[1:]), dtype=torch.float32, generator=g) \
                    * (float_high - float_low) + float_low
            else:
                tensor = torch.randint(int_low, int_high, (num_samples, *shape[1:]), generator=g)
            input_dict[keyword] = tensor
        input_dict['labels'] = torch.arange(self.model_config.num_labels).repeat(
            num_samples // self.model_config.num_labels + 1)[:num_samples]
        return Dataset.from_dict(input_dict)

    @staticmethod
    @abstractmethod
    def get_nncf_modules_in_transformer_block_order(
            compressed_model: NNCFNetwork) -> List[TransformerBlockItem]:
        pass

    @abstractmethod
    def _create_model(self) -> torch.nn.Module:
        pass

    @staticmethod
    def _get_model_input_info_dict(model_input_infos: List[ModelInputInfo]) -> List[Dict[str, Any]]:
        result = []
        for info in model_input_infos:
            item = {'sample_size': info.shape,
                    'type': info.torch_type_to_string(info.type)}
            if info.keyword is not None:
                item['keyword'] = info.keyword
            if info.filler is not None:
                item['filler'] = info.filler
            result.append(item)
        return result


class Wav2Vec2ModelConfig(Wav2Vec2Config):
    hidden_size = 4
    num_hidden_layers = 1
    num_attention_heads = 2
    intermediate_size = 3
    conv_dim = (4, 4)
    conv_stride = (1, 1)
    conv_kernel = (3, 3)
    num_conv_pos_embeddings = 3
    num_conv_pos_embedding_groups = 1
    proj_codevector_dim = 4
    classifier_proj_size = 3
    num_labels = 2


class Wav2Vec2RunRecipe(BaseMockRunRecipe):
    model_family = 'huggingface_wav2vec2'
    supports_structured_masking = True
    default_model_config = Wav2Vec2Config(
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=3,
        conv_dim=(4, 4),
        conv_stride=(1, 1),
        conv_kernel=(3, 3),
        num_conv_pos_embeddings=3,
        num_conv_pos_embedding_groups=1,
        proj_codevector_dim=4,
        classifier_proj_size=3,
        num_labels=2,
    )

    default_algo_config = MovementAlgoConfig(
        sparse_structure_by_scopes=[
            {'mode': 'block', 'sparse_factors': [2, 2], 'target_scopes': '{re}Wav2Vec2Attention'},
            {'mode': 'per_dim', 'axis': 0, 'target_scopes': '{re}intermediate_dense'},
            {'mode': 'per_dim', 'axis': 1, 'target_scopes': '{re}output_dense'},
        ],
        ignored_scopes=['{re}feature_extractor'],
    )

    def _create_model(self) -> torch.nn.Module:
        return AutoModelForAudioClassification.from_config(self.model_config)

    @property
    def model_input_info(self) -> List[ModelInputInfo]:
        return [ModelInputInfo(shape=[1, 32], keyword='input_values')]

    @property
    def transformer_block_info(self) -> List[TransformerBlockInfo]:
        model_config = self.model_config
        return [TransformerBlockInfo(
            num_hidden_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            dim_per_head=model_config.hidden_size // model_config.num_attention_heads
        )]

    @staticmethod
    def get_nncf_modules_in_transformer_block_order(
            compressed_model: NNCFNetwork) -> List[TransformerBlockItem]:
        modules = []
        for block in compressed_model.nncf_module.wav2vec2.encoder.layers:
            modules.append(TransformerBlockItem(
                block.attention.q_proj,
                block.attention.k_proj,
                block.attention.v_proj,
                block.attention.out_proj,
                block.feed_forward.intermediate_dense,
                block.feed_forward.output_dense)
            )
        return modules


class BertRunRecipe(BaseMockRunRecipe):
    model_family = 'huggingface_bert'
    supports_structured_masking = True
    default_model_config = BertConfig(
        hidden_size=4,
        intermediate_size=3,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_hidden_layers=1,
        vocab_size=10,
        num_labels=2,
        mhsa_qkv_bias=True,
        mhsa_o_bias=True,
        ffn_bias=True
    )
    default_algo_config = MovementAlgoConfig(
        sparse_structure_by_scopes=[
            {'mode': 'block', 'sparse_factors': [2, 2], 'target_scopes': '{re}attention'},
            {'mode': 'per_dim', 'axis': 0, 'target_scopes': '{re}BertIntermediate'},
            {'mode': 'per_dim', 'axis': 1, 'target_scopes': '{re}BertOutput'},
        ],
        ignored_scopes=['{re}embedding', '{re}pooler', '{re}classifier'],
    )

    def __init__(self, model_config: BertConfig,
                 algo_config: MovementAlgoConfig,
                 log_dir=None) -> None:
        super().__init__(model_config, algo_config, log_dir)
        extra_model_keys = {'mhsa_qkv_bias', 'mhsa_o_bias', 'ffn_bias'}
        for key in extra_model_keys:
            value = getattr(self.model_config, key, True)
            setattr(self.model_config, key, value)

    def _create_model(self) -> torch.nn.Module:
        model = AutoModelForSequenceClassification.from_config(self.model_config)
        for block in model.bert.encoder.layer:
            if not self.model_config.mhsa_qkv_bias:
                block.attention.self.query.bias = None
                block.attention.self.key.bias = None
                block.attention.self.value.bias = None
            if not self.model_config.mhsa_o_bias:
                block.attention.output.dense.bias = None
            if not self.model_config.ffn_bias:
                block.intermediate.dense.bias = None
                block.output.dense.bias = None
        return model

    @property
    def model_input_info(self) -> List[ModelInputInfo]:
        dim = self.model_config.max_position_embeddings
        return [
            ModelInputInfo(shape=[1, dim], type_str='long', keyword='input_ids'),
            ModelInputInfo(shape=[1, dim], type_str='long', keyword='attention_mask'),
            ModelInputInfo(shape=[1, dim], type_str='long', keyword='token_type_ids'),
            ModelInputInfo(shape=[1, dim], type_str='long', keyword='position_ids'),
        ]

    @property
    def transformer_block_info(self) -> List[TransformerBlockInfo]:
        model_config = self.model_config
        return [TransformerBlockInfo(
            num_hidden_layers=model_config.num_hidden_layers,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            dim_per_head=model_config.hidden_size // model_config.num_attention_heads
        )]

    @staticmethod
    def get_nncf_modules_in_transformer_block_order(
            compressed_model: NNCFNetwork) -> List[TransformerBlockItem]:
        modules = []
        for block in compressed_model.nncf_module.bert.encoder.layer:
            modules.append(TransformerBlockItem(
                block.attention.self.query,
                block.attention.self.key,
                block.attention.self.value,
                block.attention.output.dense,
                block.intermediate.dense,
                block.output.dense)
            )
        return modules


class SwinRunRecipe(BaseMockRunRecipe):
    model_family = 'huggingface_swin'
    supports_structured_masking = True
    default_model_config = SwinConfig(
        image_size=4,
        patch_size=1,
        num_channels=3,
        embed_dim=4,
        depths=[1],
        num_heads=[2],
        window_size=2,
        mlp_ratio=3 / 4,
        num_labels=2,
        qkv_bias=True,
    )
    default_algo_config = MovementAlgoConfig(
        sparse_structure_by_scopes=[
            {'mode': 'block', 'sparse_factors': [2, 2], 'target_scopes': '{re}attention'},
            {'mode': 'per_dim', 'axis': 0, 'target_scopes': '{re}SwinIntermediate'},
            {'mode': 'per_dim', 'axis': 1, 'target_scopes': '{re}SwinOutput'},
        ],
        ignored_scopes=['{re}embedding', '{re}pooler', '{re}classifier'],
    )

    def _create_model(self) -> torch.nn.Module:
        return AutoModelForImageClassification.from_config(self.model_config)

    @property
    def model_input_info(self) -> List[ModelInputInfo]:
        img_size = self.model_config.image_size
        return [
            ModelInputInfo(shape=[1, self.model_config.num_channels, img_size, img_size],
                           keyword='pixel_values')
        ]

    @property
    def transformer_block_info(self) -> List[TransformerBlockInfo]:
        model_config = self.model_config
        info_list = []
        for i, (depth, num_head) in enumerate(zip(model_config.depths, model_config.num_heads)):
            hidden_size = model_config.embed_dim * int(2 ** i)
            intermediate_size = int(hidden_size * model_config.mlp_ratio)
            dim_per_head = hidden_size // num_head
            info_list.append(TransformerBlockInfo(
                num_hidden_layers=depth,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dim_per_head=dim_per_head
            ))
        return info_list

    @staticmethod
    def get_nncf_modules_in_transformer_block_order(
            compressed_model: NNCFNetwork) -> List[TransformerBlockItem]:
        modules = []
        for layer in compressed_model.nncf_module.swin.encoder.layers:
            for block in layer.blocks:
                modules.append(TransformerBlockItem(
                    block.attention.self.query,
                    block.attention.self.key,
                    block.attention.self.value,
                    block.attention.output.dense,
                    block.intermediate.dense,
                    block.output.dense)
                )
        return modules


class LinearForClassification(PreTrainedModel):

    def __init__(self, input_size: int = 4, bias: bool = True, num_labels: int = 2):
        super().__init__(PretrainedConfig())
        self.model = torch.nn.Linear(input_size, num_labels, bias=bias)

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}


class Conv2dForClassification(LinearForClassification):

    def __init__(self, input_size: int = 4, bias: bool = True, num_labels: int = 2):
        super().__init__(input_size, bias, num_labels)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, num_labels, kernel_size=3, stride=1, padding=1, bias=bias),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )


class Conv2dPlusLinearForClassification(LinearForClassification):

    def __init__(self, input_size: int = 4, bias: bool = True, num_labels: int = 2):
        super().__init__(input_size, bias, num_labels)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, num_labels, kernel_size=3, stride=1, padding=1, bias=bias),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(num_labels, num_labels, bias=bias)
        )


class LinearRunRecipe(BaseMockRunRecipe):
    model_family = 'linear'
    supports_structured_masking = False
    default_model_config = PretrainedConfig(
        num_labels=2,
        input_size=4,
        bias=True
    )
    default_algo_config = MovementAlgoConfig(
        enable_structured_masking=False
    )

    def _create_model(self) -> torch.nn.Module:
        model_config = self.model_config
        return LinearForClassification(input_size=model_config.input_size,
                                       bias=model_config.bias,
                                       num_labels=model_config.num_labels)

    @property
    def model_input_info(self) -> List[ModelInputInfo]:
        return [ModelInputInfo(shape=[1, self.model_config.input_size], keyword='tensor')]

    @property
    def transformer_block_info(self) -> List[TransformerBlockInfo]:
        return []

    @staticmethod
    def get_nncf_modules_in_transformer_block_order(
            compressed_model: NNCFNetwork) -> List[TransformerBlockItem]:
        return []


class Conv2dRunRecipe(LinearRunRecipe):
    model_family = 'conv2d'
    supports_structured_masking = False
    default_model_config = PretrainedConfig(
        num_labels=2,
        input_size=4,
        bias=True
    )
    default_algo_config = MovementAlgoConfig(
        enable_structured_masking=False
    )

    def _create_model(self) -> torch.nn.Module:
        model_config = self.model_config
        return Conv2dForClassification(input_size=model_config.input_size,
                                       bias=model_config.bias,
                                       num_labels=model_config.num_labels)

    @property
    def model_input_info(self) -> List[ModelInputInfo]:
        return [
            ModelInputInfo(shape=[1, 3, self.model_config.input_size, self.model_config.input_size],
                           keyword='tensor')
        ]


class Conv2dPlusLinearRunRecipe(LinearRunRecipe):
    model_family = 'conv2d+linear'
    supports_structured_masking = False
    default_model_config = PretrainedConfig(
        num_labels=2,
        input_size=4,
        bias=True
    )
    default_algo_config = MovementAlgoConfig(
        enable_structured_masking=False
    )

    def _create_model(self) -> torch.nn.Module:
        model_config = self.model_config
        return Conv2dPlusLinearForClassification(input_size=model_config.input_size,
                                                 bias=model_config.bias,
                                                 num_labels=model_config.num_labels)

    @property
    def model_input_info(self) -> List[ModelInputInfo]:
        return [
            ModelInputInfo(shape=[1, 3, self.model_config.input_size, self.model_config.input_size],
                           keyword='tensor')
        ]
