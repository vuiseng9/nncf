"""
 Copyright (c) 2019-2022 Intel Corporation
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
from nncf.torch.sparsity.hs.layers import HSSparsifyingWeight
from nncf.torch.sparsity.hs.loss import HSLoss
from nncf.torch.utils import get_model_device
from nncf.torch.utils import get_world_size
from nncf.common.accuracy_aware_training.training_loop import ADAPTIVE_COMPRESSION_CONTROLLERS
from nncf.torch.sparsity.collector import PTSparseModelStatisticsCollector, PTWBSparseModelStatisticsCollector
from nncf.common.sparsity.schedulers import SPARSITY_SCHEDULERS
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.sparsity.statistics import RBSparsityStatistics
from nncf.common.statistics import NNCFStatistics


@PT_COMPRESSION_ALGORITHMS.register('hs_sparsity')
class HSSparsityBuilder(BaseSparsityAlgoBuilder):
    def create_weight_sparsifying_operation(self, target_module_node: NNCFNode, compression_lr_multiplier: float):
        return HSSparsifyingWeight(target_module_node, frozen=False,
                                   compression_lr_multiplier=compression_lr_multiplier)

    def _build_controller(self, model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return HSSparsityController(model, self._sparsified_module_info, self.config)


@ADAPTIVE_COMPRESSION_CONTROLLERS.register('pt_hs_sparsity')
class HSSparsityController(BaseSparsityAlgoController):
    def __init__(self, target_model: NNCFNetwork, sparsified_module_info: List[SparseModuleInfo],
                 config: NNCFConfig):
        super().__init__(target_model, sparsified_module_info)
        self.sparsified_module_info = sparsified_module_info
        algo_config = extract_algo_specific_config(config, 'hs_sparsity')
        params = deepcopy(algo_config.get('params', {}))

        self._distributed = False
        self._mode = params.get('sparsity_level_setting_mode', 'global')
        self._check_sparsity_masks = params.get('check_sparsity_masks', False)

        sparsify_operations = [m.operand for m in self.sparsified_module_info]
        self.hs_weight_penalty = params.get('weight_penalty', 1.0)
        self.hs_bias_penalty = params.get('bias_penalty', 1.0)

        # By design, weight_penalty is only applied after bias is frozen.
        self._loss = HSLoss(sparsify_operations,
                                weight_penalty=1.0,
                                bias_penalty=self.hs_bias_penalty)

        scheduler_cls = SPARSITY_SCHEDULERS.get('freezer')
        self._scheduler = scheduler_cls(self, params)


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

    def freeze_bias_mask(self):
        for sminfo in self.sparsified_module_info:
            sminfo.operand.frozen_mask_b = True
            for n, p in sminfo.module.named_parameters():
                if n == 'bias':
                    p.requires_grad = False
            self._loss.reset_active_layer_count()
    
    def freeze_weight_mask(self):
        for sminfo in self.sparsified_module_info:
            sminfo.operand.frozen_mask_w = True
            for n, p in sminfo.module.named_parameters():
                if n == 'weight':
                    p.requires_grad = False
            self._loss.reset_active_layer_count()

    def apply_weight_penalty(self):
        self._loss.weight_penalty = self.hs_weight_penalty

    def distributed(self):
        if not dist.is_initialized():
            raise KeyError('Could not set distributed mode for the compression algorithm '
                           'because the default process group has not been initialized.')

        if 'cuda' in get_model_device(self._model).type:
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
        collector = PTWBSparseModelStatisticsCollector(self.model, self.sparsified_module_info)
        model_statistics = collector.collect()

        # target_sparsity_level = self.scheduler.current_sparsity_level if self._mode == 'global' else None
        # mean_sparse_prob = 1.0 - self.loss.mean_sparse_prob

        # stats = RBSparsityStatistics(model_statistics, 0, 0)

        nncf_stats = NNCFStatistics()
        nncf_stats.register('hs_sparsity', model_statistics)
        return nncf_stats

    def prepare_for_export(self):
        """
        Applies pruning masks to layer weights before exporting the model to ONNX.
        """
        self._propagate_masks()

    def _propagate_masks(self):
        def calc_sparsity(tensor):
            return 1-tensor.count_nonzero()/tensor.numel()
      
        from collections import OrderedDict
        sparse_sd = OrderedDict()
        with torch.no_grad():    
            for sparse_info in self.sparsified_module_info:
                for modn, m in self.model.named_modules():
                    if m == sparse_info.module:
                        for n, p in sparse_info.module.named_parameters():
                            if n == 'weight':
                                sparse_sd[modn+'.weight'] = sparse_info.operand.weight_binary_mask*p
                            if n == 'bias':
                                sparse_sd[modn+'.bias'] = sparse_info.operand.bias_binary_mask*p

                        # # print("- SparseModule: {} -".format(n))
                        # # print("\tw_mask sparsity: {:.3f}".format(calc_sparsity(sparse_info.operand.weight_ctx.binary_mask)))
                        # # print("\tw_sd   sparsity: {:.3f}".format(calc_sparsity(m.weight)))
                        # sparse_sd[n+'.weight'] = sparse_info.operand.apply_binary_mask(m.weight)
                        # # print("\t*w_sd  sparsity: {:.3f}".format(calc_sparsity(sparse_sd[n+'.weight'])))

                        # if hasattr(m, 'bias'):
                        #     # print("\tb_mask sparsity: {:.3f}".format(calc_sparsity(sparse_info.operand.bias_ctx.binary_mask)))
                        #     # print("\tb_sd   sparsity: {:.3f}".format(calc_sparsity(m.bias)))
                        #     sparse_sd[n+'.bias'] = sparse_info.operand.apply_binary_mask(m.bias, isbias=True)
                        #     # print("\t*w_sd  sparsity: {:.3f}".format(calc_sparsity(sparse_sd[n+'.bias'])))

        model_sd = self.model.state_dict()
        for k, v in sparse_sd.items():
            assert k in model_sd, "key not exists!"
            model_sd[k] = sparse_sd[k]
        self.model.load_state_dict(model_sd)
