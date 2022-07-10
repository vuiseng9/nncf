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

import torch

from nncf.torch.compression_method_api import PTCompressionLoss

# Actually in responsible to lean density to target value
class HSLoss(PTCompressionLoss):
    def __init__(self, sparse_layers=None, weight_penalty=1.0, bias_penalty=1.0):
        super().__init__()
        self._sparse_layers = sparse_layers
        self.weight_penalty = weight_penalty
        self.bias_penalty = bias_penalty
        self.disabled = False
        self.reset_active_layer_count()

    def reset_active_layer_count(self):
        n_active_bias=0
        n_active_weight=0
        for l in self._sparse_layers:
            if l.frozen_mask_w is False:
                n_active_weight += 1

            if l.prune_bias is True:
                if l.frozen_mask_b is False:
                    n_active_bias += 1

        self.n_active_weight = n_active_weight
        self.n_active_bias = n_active_bias
            
    def set_layers(self, sparse_layers):
        self._sparse_layers = sparse_layers

    def disable(self):
        if not self.disabled:
            self.disabled = True

            for sparse_layer in self._sparse_layers:
                sparse_layer.frozen_mask_w = True
                sparse_layer.frozen_mask_b = True

    def calculate(self) -> torch.Tensor:
        if self.disabled:
            return 0

        loss_w = 0
        loss_b = 0

        for sparse_layer in self._sparse_layers:           
            hs_w, hs_b = sparse_layer.loss()
            loss_w += hs_w
            loss_b += hs_b

        loss = 0.0
        if self.n_active_weight != 0:
            loss += self.weight_penalty*loss_w/self.n_active_weight
        if self.n_active_bias != 0:
            loss += self.bias_penalty*loss_b/self.n_active_bias
        
        return loss