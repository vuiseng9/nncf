"""
 Copyright (c) 2021 Intel Corporation
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

from typing import Optional
import numpy as np
import scipy.optimize

from nncf.common.utils.registry import Registry
from nncf.common.schedulers import ExponentialDecaySchedule, BaseCompressionScheduler
from copy import deepcopy
from collections import OrderedDict
from nncf.common.utils.logger import logger as nncf_logger

PRUNING_SCHEDULERS = Registry("pruning_schedulers")


class PruningScheduler(BaseCompressionScheduler):
    """
    This is the class from which all pruning schedulers inherit.

    A pruning scheduler is an object which specifies the pruning
    level at each training epoch. It involves a scheduling algorithm,
    defined in the `_calculate_pruning_level()` method and a state
    (some parameters required for current pruning level calculation)
    defined in the `__init__()` method.

    :param initial_level: Pruning level which already has been
        applied to the model. It is the level at which the schedule begins.
    :param target_level: Pruning level at which the schedule ends.
    :param num_warmup_epochs: Number of epochs for model pre-training before pruning.
    :param num_pruning_epochs: Number of epochs during which the pruning level
        is increased from `initial_level` to `target_level`.
    :param freeze_epoch: Zero-based index of the epoch from which the pruning
        mask will be frozen and will not be trained.
    """

    def __init__(self, controller, params: dict):
        """
        Initializes the internal state of the pruning scheduler.

        :param controller: Pruning algorithm controller.
        :param params: Parameters of the scheduler.
        """
        super().__init__()
        self._controller = controller
        self.initial_level = self._controller.pruning_init

        if self._controller.prune_flops:
            self.target_level = params.get('pruning_flops_target')
        else:
            self.target_level = params.get('pruning_target', 0.5)

        self.num_warmup_epochs = params.get('num_init_steps', 0)
        self.num_pruning_epochs = params.get('pruning_steps', 100)
        self.freeze_epoch = self.num_warmup_epochs + self.num_pruning_epochs
        self._current_level = self.initial_level

    def _calculate_pruning_level(self) -> float:
        """
        Calculates a pruning level that should be applied to the model
        for the `current_epoch`.

        :return: Pruning level that should be applied to the model.
        """
        raise NotImplementedError(
            'PruningScheduler implementation must override _calculate_pruning_level method.')

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        """
        Should be called at the beginning of each training epoch to prepare
        the pruning method to continue training the model in the `next_epoch`.

        :param next_epoch: The epoch index for which the pruning scheduler
            will update the state of the pruning method.
        """
        super().epoch_step(next_epoch)
        self._current_level = self._calculate_pruning_level()
        self._controller.set_pruning_rate(self.current_pruning_level)
        if self.current_epoch >= self.freeze_epoch:
            self._controller.freeze()

    def step(self, next_step: Optional[int] = None) -> None:
        """
        Should be called at the beginning of each training step to prepare
        the pruning method to continue training the model in the `next_step`.

        :param next_step: The global step index for which the pruning scheduler
            will update the state of the pruning method.
        """
        super().step(next_step)
        self._controller.step(next_step)

    @property
    def current_pruning_level(self) -> float:
        """
        Returns pruning level for the `current_epoch`.

        :return: Current sparsity level.
        """
        if self.current_epoch >= self.num_warmup_epochs:
            return self._current_level
        return 0


@PRUNING_SCHEDULERS.register("baseline")
class BaselinePruningScheduler(PruningScheduler):
    """
    Pruning scheduler which applies the same pruning level for each epoch.

    The model is trained without pruning during `num_warmup_epochs` epochs.
    Then scheduler sets `target_level` and freezes the algorithm.
    """

    def __init__(self, controller, params: dict):
        super().__init__(controller, params)
        self.freeze_epoch = self.num_warmup_epochs

    def _calculate_pruning_level(self) -> float:
        return self.target_level


@PRUNING_SCHEDULERS.register("exponential")
class ExponentialPruningScheduler(PruningScheduler):
    """
    Pruning scheduler with an exponential decay schedule.

    This scheduler applies exponential decay to the density level
    to calculate the pruning level for the `current_epoch`.
    The density level for the `current_epoch` is calculated as

        current_density = 1.0 - current_level
    """

    def __init__(self, controller, params: dict):
        """
        Initializes a pruning scheduler with an exponential decay schedule.

        :param controller: Sparsity algorithm controller.
        :param params: Parameters of the scheduler.
        """
        super().__init__(controller, params)
        initial_density = 1.0 - self.initial_level
        target_density = 1.0 - self.target_level
        target_epoch = self.num_pruning_epochs - 1
        self.schedule = ExponentialDecaySchedule(initial_density, target_density, target_epoch)

    def _calculate_pruning_level(self) -> float:
        current_density = self.schedule(self.current_epoch - self.num_warmup_epochs)
        current_level = 1.0 - current_density
        return min(current_level, self.target_level)


@PRUNING_SCHEDULERS.register("exponential_with_bias")
class ExponentialWithBiasPruningScheduler(PruningScheduler):
    """
    Pruning scheduler which calculates pruning rate for the current epoch
    according to the formula:

        current_level = a * exp(-k * epoch_idx) + b,

    where a, b, k is a params.
    """

    def __init__(self, controller, params: dict):
        """
        Initializes a pruning scheduler with an exponential (with bias) decay schedule.

        :param controller: Sparsity algorithm controller.
        :param params: Parameters of the scheduler.
        """
        super().__init__(controller, params)
        target_epoch = self.num_pruning_epochs - 1
        self.a, self.b, self.k = self._init_exp(target_epoch, self.initial_level, self.target_level)

    def _calculate_pruning_level(self) -> float:
        current_level = self.a * np.exp(-self.k * (self.current_epoch - self.num_warmup_epochs)) + self.b
        return min(current_level, self.target_level)

    @staticmethod
    def _init_exp(epoch_idx, p_min, p_max, factor=0.125):
        """
        Finds parameters a, b, k from the system:
            p_min = a + b
            p_max = a * exp(-k * epoch_idx) + b
            3/4 * p_max = a * exp(-k * factor * epoch_idx) + b
        For more details see [paper](https://arxiv.org/pdf/1808.07471.pdf).

        :param epoch_idx: Zero-based index of the epoch for which the a * exp(-k * epoch_idx) + b = p_max.
        :param p_min: Initial pruning level at which the schedule begins.
        :param p_max: Target pruning level at which the schedule ends.
        :param factor: Hyperparameter.
        """
        def get_b(a):
            return p_min - a

        def get_a(k):
            return (p_max - p_min) / (np.exp(-k * epoch_idx) - 1)

        def f_to_solve(x):
            c = (0.75 * p_max - p_min) / (p_max - p_min)
            y = np.exp(-x * epoch_idx)
            return y ** factor - c * y + c - 1

        k = scipy.optimize.fsolve(f_to_solve, [1])[0]
        a = get_a(k)
        b = get_b(a)
        return a, b, k


@PRUNING_SCHEDULERS.register("paas_ft")
class PaasFTScheduler(PruningScheduler):
    """
    NEMO/PAAS Fine-tuning on frozen groupwise pruning ratio
    """

    def __init__(self, controller, params: dict):
        super().__init__(controller, params)
        self.freeze_epoch = self.num_warmup_epochs
        self.num_warmup_epochs = 0 # Freezing these per paas use-cases
        self.num_pruning_epochs = 1000 # Freezing these per paas use-cases
        group_id_list = list(map(lambda x: x.id, controller.pruned_module_groups_info.get_all_clusters()))
        groupwise_pruning_cfg = params.get('groupwise_pruning_cfg', None)
        assert len(group_id_list) == len(groupwise_pruning_cfg)
        
        self.groupwise_pruning_cfg= {int(gid): ratio for gid, ratio in groupwise_pruning_cfg.items()}

    def _calculate_pruning_level(self) -> float:
        return self.groupwise_pruning_cfg

@PRUNING_SCHEDULERS.register("paas_gradual_ft")
class PaasGradualFTScheduler(PruningScheduler):
    """
    NEMO/PAAS Fine-tuning with gradual pruning with mask
    """

    def __init__(self, controller, params: dict):
        super().__init__(controller, params)
        self.freeze_epoch = self.num_warmup_epochs
        self.num_warmup_epochs = 0 # Freezing these per paas use-cases
        self.num_pruning_epochs = 1000 # Freezing these per paas use-cases
        self.ft_step_per_group = params.get('ft_step_per_group', 1)
        group_id_list = list(map(lambda x: x.id, controller.pruned_module_groups_info.get_all_clusters()))
        groupwise_pruning_cfg = params.get('groupwise_pruning_cfg', None)
        assert len(group_id_list) == len(groupwise_pruning_cfg)
        
        self.groupwise_pruning_cfg = {int(gid): ratio for gid, ratio in groupwise_pruning_cfg.items()}
        self.groupwise_pruning_cfg = OrderedDict(sorted(self.groupwise_pruning_cfg.items(), key=lambda item: item[1], reverse=params.get("large_groupwise_ratio_first", False)))

        # Mask extraction
        # self.groupwise_pruning_current_mask will be unpruned
        self.groupwise_pruning_current_mask = deepcopy(self._controller.pruned_module_groups_info.get_all_clusters())

        # Use the following function to just get mask computed and prevet mask being applied to parameters
        self._controller._set_binary_masks_for_pruned_modules_groupwise(self.groupwise_pruning_cfg)
        self.groupwise_pruning_target_mask = deepcopy(self._controller.pruned_module_groups_info.get_all_clusters())

        # reset mask
        self._controller._set_binary_masks_for_pruned_modules_groupwise(self.groupwise_pruning_current_mask)
        # IMPORTANT, flops are calculated based on mask in _set_binary_masks_for_pruned_modules_groupwise
        # Ensure this is being validated

        # Template to access mask
        # for group in self._controller.pruned_module_groups_info.get_all_clusters():
        #     for minfo in group.nodes:
        #         pruning_module = minfo.operand
        #         pruning_module.binary_filter_pruning_mask = torch.ones_like(pruning_module.binary_filter_pruning_mask)

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        """
        Should be called at the beginning of each training epoch to prepare
        the pruning method to continue training the model in the `next_epoch`.

        :param next_epoch: The epoch index for which the pruning scheduler
            will update the state of the pruning method.
        """
        if next_epoch is None:
            next_epoch = self.current_epoch + 1
        self.current_epoch = next_epoch
        
        sequence_id = (self.current_epoch//self.ft_step_per_group) % len(self.groupwise_pruning_cfg)
        tuple_list = list(self.groupwise_pruning_cfg.items())
        group_id = tuple_list[sequence_id][0]

        nncf_logger.info("#PAAS# backend: PaasGradualFTScheduler, prune filter group {}".format(group_id))
        for current_node, target_node in zip(self.groupwise_pruning_current_mask[group_id].nodes, self.groupwise_pruning_target_mask[group_id].nodes):
            assert current_node.nncf_node_id == target_node.nncf_node_id, "Logical Bug"
            current_node.operand.binary_filter_pruning_mask = target_node.operand.binary_filter_pruning_mask

        self._controller.set_pruning_rate(self.groupwise_pruning_current_mask)
        # if self.current_epoch >= self.freeze_epoch:
        #     self._controller.freeze()