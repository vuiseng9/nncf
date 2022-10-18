from pathlib import Path
from typing import List, Optional, Union

import nncf
import pytest
import torch
from nncf.common.sparsity.schedulers import PolynomialThresholdScheduler
from nncf.common.sparsity.statistics import MovementSparsityStatistics
from nncf.common.utils.helpers import matches_any, should_consider_scope
from nncf.torch import create_compressed_model
from nncf.torch.layer_utils import CompressionParameter
from nncf.torch.layers import NNCFLinear
from nncf.torch.module_operations import UpdateWeightAndBias
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.sparsity.layers import BinaryMask
from nncf.torch.sparsity.movement.algo import (ImportanceLoss,
                                               MovementSparsifier,
                                               MovementSparsityController,
                                               SparseConfig, SparseStructure)
from onnx import numpy_helper
from pytest import approx
from tests.torch.sparsity.movement.helpers import (BaseCallback, ConfigBuilder,
                                                   bert_tiny_torch_model,
                                                   bert_tiny_unpretrained,
                                                   run_movement_pipeline)
from transformers import (AutoModelForSequenceClassification, BertConfig,
                          TrainingArguments)
from transformers.trainer_callback import TrainerControl, TrainerState


@pytest.mark.parametrize('description', [
    # TODO: check fill operation cases
    dict(unstructured_masks=([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [0, 0, 0, 0],  # mhsa query
                             [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [1, 0, 0, 0],  # mhsa key
                             [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [0, 1, 0, 0],  # mhsa value
                             [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [0, 0, 0, 0],  # mhsa output
                             [[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]], [1, 0, 0],  # ffn intermediate
                             [[0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]], [0, 0, 0, 0]),  # ffn output
         ref_structured_masks=([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], [1, 1, 0, 0],
                               [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], [1, 1, 0, 0],
                               [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], [1, 1, 0, 0],
                               [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]], [1, 1, 1, 1],
                               [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]], [1, 1, 0],
                               [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]], [1, 1, 1, 1])),
])
def test_controller_structured_mask_filling(tmp_path, description):
    sparse_structures = [
        ["block", [1, 1], "{re}.*attention*"],
        ["per_dim", [0], "{re}.*BertIntermediate.*"],
        ["per_dim", [1], "{re}.*BertOutput.*"],
    ]
    nncf_config = ConfigBuilder(sparse_structure_by_scopes=sparse_structures).build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_unpretrained(), nncf_config)
    compressed_model.train()

    state_keys = []
    for keyword in ["attention.self.query", 'attention.self.key', 'attention.self.value',
                    'attention.output.dense', 'intermediate.dense', 'output.dense']:
        for attr in ['weight', 'bias']:
            state_keys.append(f"nncf_module.bert.encoder.layer.0.{keyword}"
                              f".pre_ops.0.op.{attr}_ctx._binary_mask")

    unstructued_state_dict = dict(zip(state_keys, map(torch.FloatTensor, description['unstructured_masks'])))
    compressed_model.load_state_dict(unstructued_state_dict, strict=False)
    compression_ctrl.reset_independent_structured_mask()
    compression_ctrl.resolve_structured_mask()
    compression_ctrl.populate_structured_mask()
    structured_state_dict = compressed_model.state_dict()
    ref_structured_state_dict = dict(zip(state_keys, map(torch.FloatTensor, description['ref_structured_masks'])))
    for key in state_keys:
        assert torch.allclose(structured_state_dict[key], ref_structured_state_dict[key])
        # print(structured_state_dict[key].numpy().astype(np.int).tolist(), ',')