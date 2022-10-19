
import pytest
import torch
from nncf.torch import create_compressed_model
from nncf.torch.sparsity.movement.algo import StructuredMask
from nncf.torch.sparsity.movement.functions import binary_mask_by_threshold
from tests.torch.sparsity.movement.helpers import (ConfigBuilder,
                                                   bert_tiny_unpretrained)


def test_structured_mask_setter(tmp_path):
    nncf_config = ConfigBuilder(sparse_structure_by_scopes=[]).build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_unpretrained(), nncf_config)
    ctx: StructuredMask = compression_ctrl.structured_ctx_by_group[0][0]  # pick one structured mask
    # check independent mask
    ref_mask = ctx.sparse_module_info.operand.get_structured_mask((2, 4))
    assert torch.allclose(ctx.independent_structured_mask, ref_mask)
    ref_mask = ref_mask * 2. + 1.
    ctx.independent_structured_mask = ref_mask
    assert torch.allclose(ctx.independent_structured_mask, ref_mask)
    with pytest.raises(ValueError):
        ctx.independent_structured_mask = ref_mask[:1, :1]
    # check dependent mask
    assert ctx.dependent_structured_mask is None
    ref_mask = torch.ones_like(ref_mask)
    ctx.dependent_structured_mask = ref_mask
    assert torch.allclose(ctx.dependent_structured_mask, ref_mask)
    with pytest.raises(ValueError):
        ctx.dependent_structured_mask = ref_mask[:1, :1]


@pytest.mark.parametrize(("input_tensor", "threshold", "max_percentile", "ref_output_tensor"), [
    (torch.tensor([1., 2, 3, 4]), 0., 0.9, torch.tensor([1., 1, 1, 1])),
    (torch.tensor([1., 2, 3, 4]), 2.5, 0.8, torch.tensor([0., 0, 1, 1])),
    (torch.tensor([1., 2, 3, 4]), 2.5, 0.2, torch.tensor([0., 1, 1, 1])),
    (torch.tensor([1., 2, 3, 4]), 5., 0.8, torch.tensor([0., 0, 0, 1])),
    (torch.tensor([1., 1, 1, 1]), 5., 0.8, torch.tensor([0., 0, 0, 0])),
])
def test_binary_mask_by_threshold(input_tensor, threshold, max_percentile, ref_output_tensor):
    for requires_grad in [True, False]:
        input_tensor.requires_grad_(requires_grad)
        output_tensor = binary_mask_by_threshold(input_tensor, threshold, max_percentile)
        assert torch.allclose(output_tensor, ref_output_tensor)
        assert output_tensor.requires_grad is requires_grad