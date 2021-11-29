"""
 Copyright (c) 2019 Intel Corporation
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
from typing import List

import torch

from nncf.torch.sparsity.layers import BinaryMask
from nncf.torch.sparsity.movement.functions import binary_mask_by_threshold
from nncf.torch.functions import logit
from nncf.torch.layer_utils import COMPRESSION_MODULES, CompressionParameter
from enum import Enum
from typing import Dict, List, Optional, Any
from copy import deepcopy

class SparseStructure(str, Enum):
    FINE = "fine"
    BLOCK = "block"
    PER_DIM = "per_dim"

class SparseConfig:
    def __init__(self, mode: SparseStructure = SparseStructure.FINE, sparse_args=None):
        self.mode = SparseStructure(mode)
        self.sparse_args = sparse_args
        self.sparse_factors = None


@COMPRESSION_MODULES.register()
class MovementSparsifyingWeight(BinaryMask):
    def __init__(self, 
                 weight_shape: List[int], 
                 frozen=True, 
                 compression_lr_multiplier=None, 
                 eps=1e-6, 
                 sparse_cfg=None):
        super().__init__(weight_shape)

        self.frozen = frozen
        self.eps = eps
        
        self.sparse_cfg = sparse_cfg
        self._importance_shape, self._bool_expand_importance = self._get_importance_shape(weight_shape)
        self._importance = CompressionParameter(
                                torch.zeros(self._importance_shape),
                                requires_grad=not self.frozen,
                                compression_lr_multiplier=compression_lr_multiplier)

        self.lmbd = 0.5 # module_level_loss_weightage
        
        self.masking_threshold = 0.0
        self.binary_mask = binary_mask_by_threshold(self._importance, self._masking_threshold)

        self.mask_calculation_hook = MaskCalculationHook(self)

    @property
    def importance(self):
        return self._importance.data

    @property
    def masking_threshold(self):
        return self._masking_threshold
    
    @masking_threshold.setter
    def masking_threshold(self, threshold_value):
        self._masking_threshold = threshold_value

    @property
    def lmbd(self):
        return self._lmbd
    
    @lmbd.setter
    def lmbd(self, module_level_loss_weightage):
        self._lmbd = module_level_loss_weightage

    def freeze_importance(self):
        self.frozen = True
        self._importance.requires_grad=False

    def unfreeze_importance(self):
        self.frozen = False
        self._importance.requires_grad=True

    def extra_repr(self):
        return '{}, {}'.format(
            self.sparse_cfg.mode, self.sparse_cfg.sparse_args)

    def _get_importance_shape(self, weight_shape):
        #TODO:remove  weight_shape, r=32, c=32):
        # Default to fine_grained sparsity
        if self.sparse_cfg is None:
            self.sparse_cfg = SparseConfig(
                SparseStructure("fine"),
                (1,1)
            )
            self.sparse_cfg.sparse_factors = (1, 1)

        if self.sparse_cfg.mode == SparseStructure.FINE:
            self.sparse_cfg.sparse_factors = (1, 1)
            return weight_shape, False

        if self.sparse_cfg.mode == SparseStructure.BLOCK:
            r, c = self.sparse_cfg.sparse_args
            assert weight_shape[0] % r == 0, "r: {} is not a factor of dim axes 0".format(r)
            assert weight_shape[1] % c == 0, "c: {} is not a factor of dim axes 1".format(c)
            self.sparse_cfg.sparse_factors = (r, c)
            return (weight_shape[0]//r, weight_shape[1]//c), True

        if self.sparse_cfg.mode == SparseStructure.PER_DIM:
            if len(self.sparse_cfg.sparse_args) != 1 or not isinstance(self.sparse_cfg.sparse_args[0], int):
                raise ValueError("Invalid sparse_arg {}, per_dim expects a single digit that indicates axes".format(self.sparse_cfg.sparse_args))

            if self.sparse_cfg.sparse_args[0] < 0 or self.sparse_cfg.sparse_args[0] >= len(weight_shape):
                raise ValueError("Invalid axes id {}, axes range {}".format(
                                                                        self.sparse_cfg.sparse_args[0],
                                                                        list(range(len(weight_shape)))))
            self.sparse_cfg.sparse_factors = deepcopy(weight_shape)
            self.sparse_cfg.sparse_factors[self.sparse_cfg.sparse_args[0]] = 1
            self.sparse_cfg.sparse_factors = tuple(self.sparse_cfg.sparse_factors)

            score_shape = []
            for axes, (dim, factor) in enumerate(zip(weight_shape, self.sparse_cfg.sparse_factors)):
                assert dim % factor == 0, "{} is not a factor of axes {} with dim size {}".format(factor, axes, dim)
                score_shape.append(dim//factor)
            return score_shape, True


    def _expand_importance(self, importance):
        #TODO only works dense layer for now
        if self._bool_expand_importance:
            return importance.repeat_interleave(
                self.sparse_cfg.sparse_factors[0], dim=0).repeat_interleave(
                self.sparse_cfg.sparse_factors[1], dim=1)
        return importance

    def _calc_training_binary_mask(self, weight):
        if self.training and not self.frozen:
            _mask = binary_mask_by_threshold(
                self._expand_importance(self._importance), 
                self._masking_threshold
            )
            self.binary_mask = _mask
            return _mask
        else:
            return self.binary_mask

    def loss(self):
        return self.lmbd * (torch.norm(
                torch.sigmoid(
                    self._expand_importance(self._importance)
                ), p=1) / self._importance.numel())


class MaskCalculationHook():
    def __init__(self, module):
        # pylint: disable=protected-access
        self.hook = module._register_state_dict_hook(self.hook_fn)

    def hook_fn(self, module, destination, prefix, local_metadata):
        module.binary_mask = binary_mask_by_threshold(
                                module._expand_importance(module.importance), 
                                module.masking_threshold
                             )
        destination[prefix + '_binary_mask'] = module.binary_mask
        return destination

    def close(self):
        self.hook.remove()
