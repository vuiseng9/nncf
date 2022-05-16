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
from typing import List

import torch
import torch.nn as nn

from nncf.torch.sparsity.hs.functions import make_binary_mask
from nncf.torch.layer_utils import COMPRESSION_MODULES, CompressionParameter
from nncf.torch.utils import is_tracing_state, no_jit_trace

@COMPRESSION_MODULES.register()
class HSSparsifyingWeight(nn.Module):
    def __init__(self, target_module_node, frozen=True, 
                       compression_lr_multiplier=None, eps=1e-6):
        super().__init__()

        self.target_module_node = target_module_node
        self.prune_bias = target_module_node.layer_attributes.bias

        self.frozen = frozen
        self.eps = eps
        
        # self._mask = CompressionParameter(logit(torch.ones(weight_shape) * 0.99), requires_grad=not self.frozen,
        #                                   compression_lr_multiplier=compression_lr_multiplier)

        weight_shape = target_module_node.layer_attributes.get_weight_shape()
        self.register_buffer("_weight_binary_mask", torch.zeros(weight_shape))
        self.weight_binary_mask = torch.ones(weight_shape)

        if self.prune_bias is True:
            bias_shape = target_module_node.layer_attributes.get_bias_shape()
            self.register_buffer("_bias_binary_mask", torch.zeros(bias_shape))
            self.bias_binary_mask = torch.ones(bias_shape)

        # self.mask_calculation_hook = MaskCalculationHook(self)
        self.hs_w = 0.0
        self.hs_b = 0.0

    @property
    def weight_binary_mask(self):
        return self._weight_binary_mask

    @weight_binary_mask.setter
    def weight_binary_mask(self, tensor):
        with torch.no_grad():
            self._weight_binary_mask.set_(tensor)

    @property
    def bias_binary_mask(self):
        return self._bias_binary_mask

    @bias_binary_mask.setter
    def bias_binary_mask(self, tensor):
        with torch.no_grad():
            self._bias_binary_mask.set_(tensor)

    def _calc_hoyer_square(self, p):
        denom = (p**2).sum()
        numer = (p.abs().sum())**2
        hs = 0.0
        if denom == 0 or numer == 0:
            pass
        else:
            hs += (numer/denom)/p.numel()
        return hs

    def forward(self, weight, bias):
        if is_tracing_state():
            with no_jit_trace():
                return weight.mul_(self.binary_mask)

        self.weight_binary_mask = make_binary_mask(weight)
        self.hs_w = self._calc_hoyer_square(weight)
        
        if self.prune_bias is True:
            self.bias_binary_mask = make_binary_mask(bias)
            self.hs_b = self._calc_hoyer_square(bias)

        return weight, bias

    def loss(self):
        return self.hs_w + self.hs_b

    def apply_binary_mask(self, param_tensor, isbias=False):
        # TODO param_tensor is dummy, this is workaround
        if isbias is True:
            return self.bias_binary_mask
        return self.weight_binary_mask
        
#TODO: why do we need this?
class MaskCalculationHook():
    def __init__(self, module):
        # pylint: disable=protected-access
        self.hook = module._register_state_dict_hook(self.hook_fn)

    def hook_fn(self, module, destination, prefix, local_metadata):
        module.binary_mask = binary_mask(module.mask)
        destination[prefix + '_binary_mask'] = module.binary_mask
        return destination

    def close(self):
        self.hook.remove()
