import types
import math
from torch._six import inf
from functools import wraps
import warnings
import weakref
from collections import Counter
from bisect import bisect_right

from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

class PAASStepLR(StepLR):
    """behave as native torch StepLR except LR can be cyclical
    
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, hyperstep_per_cycle=5, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        self.hyperstep_per_cycle = hyperstep_per_cycle
        super(StepLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** ((self.last_epoch // self.step_size)%self.hyperstep_per_cycle)
                for base_lr in self.base_lrs]
