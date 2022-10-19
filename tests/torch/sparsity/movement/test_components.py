
import pytest
import torch
from nncf.torch import create_compressed_model
from nncf.torch.sparsity.movement.algo import StructuredMask
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
