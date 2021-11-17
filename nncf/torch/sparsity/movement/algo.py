"""
 Copyright (c) 2019-2020 Intel Corporation
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
from copy import deepcopy
from typing import List

import torch
import torch.distributed as dist

from nncf import NNCFConfig
from nncf.config.extractors import extract_algo_specific_config
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.api.compression import CompressionStage
from nncf.common.graph import NNCFNode
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.sparsity.base_algo import BaseSparsityAlgoBuilder, BaseSparsityAlgoController, SparseModuleInfo
from nncf.torch.sparsity.movement.layers import MovementSparsifyingWeight
from nncf.torch.sparsity.movement.loss import ImportanceLoss, SparseLossForPerLayerSparsity
from nncf.torch.utils import get_world_size
from nncf.common.accuracy_aware_training.training_loop import ADAPTIVE_COMPRESSION_CONTROLLERS
from nncf.torch.sparsity.collector import PTSparseModelStatisticsCollector
from nncf.common.sparsity.schedulers import SPARSITY_SCHEDULERS
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.sparsity.statistics import MovementSparsityStatistics
from nncf.common.statistics import NNCFStatistics


@PT_COMPRESSION_ALGORITHMS.register('movement_sparsity')
class MovementSparsityBuilder(BaseSparsityAlgoBuilder):
    def create_weight_sparsifying_operation(self, target_module_node: NNCFNode, compression_lr_multiplier: float):
        return MovementSparsifyingWeight(target_module_node.layer_attributes.get_weight_shape(), frozen=False,
                                   compression_lr_multiplier=compression_lr_multiplier)

    def _build_controller(self, model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return MovementSparsityController(model, self._sparsified_module_info, self.config)


@ADAPTIVE_COMPRESSION_CONTROLLERS.register('pt_movement_sparsity')
class MovementSparsityController(BaseSparsityAlgoController):
    def __init__(self, target_model: NNCFNetwork, sparsified_module_info: List[SparseModuleInfo],
                 config: NNCFConfig):
        super().__init__(target_model, sparsified_module_info)
        algo_config = extract_algo_specific_config(config, 'movement_sparsity')
        params = deepcopy(algo_config.get('params', {}))

        self._distributed = False
        self._mode = params.get('sparsity_level_setting_mode', 'global')
        self._check_sparsity_masks = params.get('check_sparsity_masks', False)

        sparsify_operations = [m.operand for m in self.sparsified_module_info]
        if self._mode == 'local':
            # TODO: make sure we test this loop out
            self._loss = SparseLossForPerLayerSparsity(sparsify_operations)
            self._scheduler = StubCompressionScheduler()
        else:
            scheduler_cls = SPARSITY_SCHEDULERS.get(params.get('schedule', 'exponential')) #TODO: can we actually map to other scheduler in current implementation
            self._scheduler = scheduler_cls(self, params)
            self._loss = ImportanceLoss(sparsify_operations, self.scheduler)

    def compression_stage(self) -> CompressionStage:
        if self._mode == 'local':
            return CompressionStage.FULLY_COMPRESSED

        if self.scheduler.current_sparsity_level == 0:
            return CompressionStage.UNCOMPRESSED
        if self.scheduler.current_sparsity_level >= self.scheduler.target_level:
            return CompressionStage.FULLY_COMPRESSED
        return CompressionStage.PARTIALLY_COMPRESSED

    def freeze(self):
        self._loss.disable()

    def distributed(self):
        if not dist.is_initialized():
            raise KeyError('Could not set distributed mode for the compression algorithm '
                           'because the default process group has not been initialized.')

        if next(self._model.parameters()).is_cuda:
            state = torch.cuda.get_rng_state()
            if dist.get_backend() == dist.Backend.NCCL:
                state = state.cuda()
            torch.distributed.broadcast(state, src=0)
            torch.cuda.set_rng_state(state.cpu())
        else:
            state = torch.get_rng_state()
            torch.distributed.broadcast(state, src=0)
            torch.set_rng_state(state)

        self._distributed = True

    def _check_distributed_masks(self):
        if not self._distributed or get_world_size() == 1:
            return 1

        nvalues = 0
        ncor_values = 0
        eps = 1e-4
        for minfo in self.sparsified_module_info:
            mask = minfo.operand.mask

            mask_list = [torch.empty_like(mask) for _ in range(get_world_size())]
            # nccl does not support gather, send, recv operations
            dist.all_gather(mask_list, mask)

            for i in range(1, len(mask_list)):
                rel_error = (mask_list[0] - mask_list[i]) / mask_list[0]
                ncor_values = ncor_values + (rel_error.abs() < eps).sum(dtype=mask.dtype)
                nvalues = nvalues + mask_list[i].numel()

        return ncor_values / nvalues

    def statistics(self, quickly_collected_only=False) -> NNCFStatistics:
        collector = PTSparseModelStatisticsCollector(self.model, self.sparsified_module_info)
        model_statistics = collector.collect()

        stats = MovementSparsityStatistics(model_statistics,
                                           self.scheduler.current_importance_threshold,
                                           self.scheduler.current_importance_lambda)

        nncf_stats = NNCFStatistics()
        nncf_stats.register('movement_sparsity', stats)
        return nncf_stats

    @property
    def compression_rate(self):
        return self.statistics().movement_sparsity.model_statistics.sparsity_level
