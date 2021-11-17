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



@COMPRESSION_MODULES.register()
class MovementSparsifyingWeight(BinaryMask):
    def __init__(self, weight_shape: List[int], frozen=True, compression_lr_multiplier=None, eps=1e-6):
        super().__init__(weight_shape)
        self.frozen = frozen
        self.eps = eps
        self.lmbd = 0.5 # module_level_loss_weightage
        self.masking_threshold = 0.0
        self._importance = CompressionParameter(
                                torch.zeros(weight_shape), 
                                requires_grad=not self.frozen,
                                compression_lr_multiplier=compression_lr_multiplier)
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

    def _calc_training_binary_mask(self, weight):
        if self.training and not self.frozen:
            _mask = binary_mask_by_threshold(self._importance, self._masking_threshold)
            self.binary_mask = _mask
            #TODO: remove
            # if (_mask.numel() - _mask.count_nonzero()) > 0:
            #     print("yay")
            return _mask
        else:
            return self.binary_mask

    def loss(self):
        return self.lmbd * (torch.norm(torch.sigmoid(self._importance), p=1) / self._importance.numel())


class MaskCalculationHook():
    def __init__(self, module):
        # pylint: disable=protected-access
        self.hook = module._register_state_dict_hook(self.hook_fn)

    def hook_fn(self, module, destination, prefix, local_metadata):
        module.binary_mask = binary_mask_by_threshold(module.importance, module.masking_threshold)
        destination[prefix + '_binary_mask'] = module.binary_mask
        return destination

    def close(self):
        self.hook.remove()
