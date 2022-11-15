from unittest.mock import Mock

import pytest
import torch
from tests.torch.sparsity.movement.helpers import BertRunRecipe
from tests.torch.sparsity.movement.helpers import Wav2Vec2RunRecipe
from tests.torch.sparsity.movement.helpers import mock_linear_nncf_node
from tests.torch.sparsity.movement.helpers import ensure_tensor
from tests.torch.sparsity.movement.helpers import ParamDict
from nncf.torch import create_compressed_model
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlockType
from nncf.experimental.torch.sparsity.movement.layers import MovementSparsifier, SparseConfig, SparseStructure
from nncf.experimental.torch.sparsity.movement.structured_mask_handler import StructuredMaskContextGroup, StructuredMaskHandler, StructuredMaskContext
from nncf.experimental.torch.sparsity.movement.structured_mask_strategy import STRUCTURED_MASK_STRATEGY


desc_test_update_independent_structured_mask = {
    "prune1row": dict(
        weight_binary_mask=ensure_tensor([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
        bias_binary_mask=ensure_tensor([1, 0, 0]),
        prune_grid=(1, 3),
        ref_independent_structured_mask=ensure_tensor([[1], [1], [0]])
    ),
    "prune1col": dict(
        weight_binary_mask=ensure_tensor([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
        bias_binary_mask=ensure_tensor([1, 0, 0]),
        prune_grid=(3, 1),
        ref_independent_structured_mask=ensure_tensor([[1, 1, 0]])
    ),
    "prune1col_nobias": dict(
        weight_binary_mask=ensure_tensor([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
        bias_binary_mask=None,
        prune_grid=(3, 1),
        ref_independent_structured_mask=ensure_tensor([[1, 1, 0]])
    ),
    "not_pruneable": dict(
        weight_binary_mask=ensure_tensor([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
        bias_binary_mask=ensure_tensor([1, 1, 1]),
        prune_grid=(1, 3),
        ref_independent_structured_mask=ensure_tensor([[1], [1], [1]])
    )
}


class TestStructuredMaskContext:
    @pytest.mark.parametrize(('input_grid', 'ref_resolved_grid'), [
        ((1, 4), (1, 4)),
        ((-1, 2), (4, 2)),
        ((4, -1), (4, 4))
    ])
    def test_can_resolve_prune_grid_size(self, input_grid, ref_resolved_grid):
        operand = MovementSparsifier(
            mock_linear_nncf_node(4, 4),
            SparseConfig(SparseStructure.FINE),
        )
        ctx = StructuredMaskContext(operand, 'linear', input_grid)
        assert ctx.grid_size == ref_resolved_grid

    @pytest.mark.parametrize(('structure_grid_size', 'ref_mask_shape'), [
        ((2, 4), torch.Size((2, 1))),
        ((4, 1), torch.Size((1, 4))),
    ])
    @pytest.mark.parametrize('is_dependent_mask', [True, False],
                             ids=['dependent', 'independent'])
    def test_structured_mask_setter(self, is_dependent_mask: bool, structure_grid_size, ref_mask_shape):
        mask_name = 'dependent_structured_mask' if is_dependent_mask else 'independent_structured_mask'
        operand = MovementSparsifier(
            mock_linear_nncf_node(4, 4),
            SparseConfig(SparseStructure.FINE),
        )
        ctx = StructuredMaskContext(operand, 'linear', structure_grid_size)
        assert getattr(ctx, mask_name) is None
        # initialize
        ref_mask1 = torch.ones(ref_mask_shape)
        setattr(ctx, mask_name, ref_mask1)
        assert torch.equal(getattr(ctx, mask_name), ref_mask1)
        assert getattr(ctx, mask_name).requires_grad is False
        assert getattr(ctx, mask_name).device == ref_mask1.device
        id_on_creation = id(getattr(ctx, mask_name))
        assert id_on_creation != id(ref_mask1)

        # reset value
        ref_mask2 = torch.zeros(ref_mask_shape, requires_grad=True)
        setattr(ctx, mask_name, ref_mask2)
        assert torch.equal(getattr(ctx, mask_name), ref_mask2)
        assert getattr(ctx, mask_name).requires_grad is False
        assert id(getattr(ctx, mask_name)) == id_on_creation

        # set with wrong shape
        ref_mask3 = torch.ones(ref_mask_shape).unsqueeze(0)
        with pytest.raises(ValueError, match='Wrong shape'):
            setattr(ctx, mask_name, ref_mask3)

    @pytest.mark.parametrize('desc', desc_test_update_independent_structured_mask.values(),
                             ids=desc_test_update_independent_structured_mask.keys())
    def test_update_independent_structured_mask(self, desc):
        sparsifier = Mock()
        sparsifier.prune_bias = (desc['bias_binary_mask'] is not None)
        sparsifier.weight_ctx.binary_mask = desc['weight_binary_mask']
        if sparsifier.prune_bias:
            sparsifier.bias_ctx.binary_mask = desc['bias_binary_mask']
        ctx = StructuredMaskContext(sparsifier, 'linear', desc['prune_grid'])
        ctx.update_independent_structured_mask()
        assert torch.equal(ctx.independent_structured_mask,
                           desc['ref_independent_structured_mask'])

    @pytest.mark.parametrize('desc', [
        dict(mask=ensure_tensor([[1, 0, 1]]),
             prune_grid=(2, 1),
             ref_binary_mask=ensure_tensor([[1, 0, 1], [1, 0, 1]])),
        dict(mask=ensure_tensor([[1], [0], [1]]),
             prune_grid=(1, 2),
             ref_binary_mask=ensure_tensor([[1, 1], [0, 0], [1, 1]]))
    ])
    def test_populate_dependent_structured_mask(self, desc):
        sparsifier = Mock()
        sparsifier.prune_bias = True
        sparsifier.weight_ctx.binary_mask = torch.zeros_like(desc['ref_binary_mask'])
        ctx = StructuredMaskContext(sparsifier, 'linear', desc['prune_grid'])
        ctx.dependent_structured_mask = desc['mask']
        ctx.populate_dependent_structured_mask_to_operand()
        assert torch.equal(sparsifier.weight_ctx.binary_mask, desc['ref_binary_mask'])
        assert torch.equal(sparsifier.bias_ctx.binary_mask, desc['ref_binary_mask'].amax(dim=1))


class TransformerLayerMaskParam:
    def __init__(self, MHSA_Q: torch.Tensor,
                 MHSA_K: torch.Tensor,
                 MHSA_V: torch.Tensor,
                 MHSA_O: torch.Tensor,
                 FFN_I: torch.Tensor,
                 FFN_O: torch.Tensor):
        self.MHSA_Q = MHSA_Q
        self.MHSA_K = MHSA_K
        self.MHSA_V = MHSA_V
        self.MHSA_O = MHSA_O
        self.FFN_I = FFN_I
        self.FFN_O = FFN_O

    @property
    def params_in_transformer_block_order(self):
        return [self.MHSA_Q, self.MHSA_K, self.MHSA_V,
                self.MHSA_O, self.FFN_I, self.FFN_O]


desc_test_resolve_dependent_structured = {
    "prune_1head_1channel": dict(
        independent_structured=TransformerLayerMaskParam(
            MHSA_Q=ensure_tensor([[1], [0]]),
            MHSA_K=ensure_tensor([[1], [0]]),
            MHSA_V=ensure_tensor([[1], [0]]),
            MHSA_O=ensure_tensor([[1, 0]]),
            FFN_I=ensure_tensor([[1], [1], [0]]),
            FFN_O=ensure_tensor([[1, 1, 0]]),
        ),
        dependent_structured=TransformerLayerMaskParam(
            MHSA_Q=ensure_tensor([[1], [0]]),
            MHSA_K=ensure_tensor([[1], [0]]),
            MHSA_V=ensure_tensor([[1], [0]]),
            MHSA_O=ensure_tensor([[1, 0]]),
            FFN_I=ensure_tensor([[1], [1], [0]]),
            FFN_O=ensure_tensor([[1, 1, 0]]),
        ),
    ),
    "prune_0head_0channel": dict(
        independent_structured=TransformerLayerMaskParam(
            MHSA_Q=ensure_tensor([[1], [0]]),
            MHSA_K=ensure_tensor([[1], [0]]),
            MHSA_V=ensure_tensor([[0], [1]]),
            MHSA_O=ensure_tensor([[1, 0]]),
            FFN_I=ensure_tensor([[1], [1], [0]]),
            FFN_O=ensure_tensor([[1, 0, 1]]),
        ),
        dependent_structured=TransformerLayerMaskParam(
            MHSA_Q=ensure_tensor([[1], [1]]),
            MHSA_K=ensure_tensor([[1], [1]]),
            MHSA_V=ensure_tensor([[1], [1]]),
            MHSA_O=ensure_tensor([[1, 1]]),
            FFN_I=ensure_tensor([[1], [1], [1]]),
            FFN_O=ensure_tensor([[1, 1, 1]]),
        ),
    )
}

run_recipes = [BertRunRecipe(), Wav2Vec2RunRecipe()]


@pytest.mark.parametrize('run_recipe', run_recipes,
                         ids=[r.model_family for r in run_recipes])
class TestStructuredMaskHandler:
    @pytest.fixture(autouse=True)
    def setup(self, run_recipe):
        self.model = run_recipe.model
        self.nncf_config = run_recipe.nncf_config
        self.compression_ctrl, self.compressed_model = create_compressed_model(self.model, self.nncf_config,
                                                                               dump_graphs=False)
        strategy = STRUCTURED_MASK_STRATEGY.get(run_recipe.model_family).from_compressed_model(self.compressed_model)
        self.handler = StructuredMaskHandler(self.compression_ctrl.prunable_sparsified_module_info_groups, strategy)
        self.run_recipe = run_recipe
        self.all_ctxes = []
        for group in self.handler._structured_mask_ctx_groups:
            self.all_ctxes.extend(group.structured_mask_context_list)

    def test_create_ctx_groups(self):
        handler = self.handler
        run_recipe = self.run_recipe
        assert len(handler._structured_mask_ctx_groups) == 2
        handler._structured_mask_ctx_groups.sort(key=lambda group: group.group_type.value)
        group_ff = handler._structured_mask_ctx_groups[0]
        assert isinstance(group_ff, StructuredMaskContextGroup)
        assert group_ff.group_type == BuildingBlockType.FF
        assert len(group_ff.structured_mask_context_list) == 2
        group_mhsa = handler._structured_mask_ctx_groups[1]
        assert isinstance(group_mhsa, StructuredMaskContextGroup)
        assert group_mhsa.group_type == BuildingBlockType.MSHA
        assert len(group_mhsa.structured_mask_context_list) == 4

    def test_update_independent_structured_mask(self, mocker):
        handler = self.handler
        mock_methods = [mocker.patch.object(ctx, 'update_independent_structured_mask') for ctx in self.all_ctxes]
        handler.update_independent_structured_mask()
        for mock_method in mock_methods:
            mock_method.assert_called_once()

    @pytest.mark.parametrize('desc', desc_test_resolve_dependent_structured.values(),
                             ids=desc_test_resolve_dependent_structured.keys())
    def test_resolve_dependent_structured_mask(self, desc):
        handler = self.handler
        run_recipe = self.run_recipe
        modules = run_recipe.get_nncf_modules_in_transformer_block_order(self.compressed_model)[0]
        module_2_node_name = {minfo.module: minfo.module_node_name for minfo in self.compression_ctrl.sparsified_module_info}
        node_name_2_context = {ctx.module_node_name: ctx for ctx in self.all_ctxes}
        ctxes = [node_name_2_context[module_2_node_name[m]] for m in modules]
        for ctx, param in zip(ctxes, desc['independent_structured'].params_in_transformer_block_order):
            ctx.independent_structured_mask = param

        handler.resolve_dependent_structured_mask()
        for ctx, ref_param in zip(ctxes, desc['dependent_structured'].params_in_transformer_block_order):
            assert torch.allclose(ctx.dependent_structured_mask, ref_param)

    def test_populate_dependent_structured_mask_to_operand(self, mocker):
        handler = self.handler
        mock_methods = [mocker.patch.object(ctx, 'populate_dependent_structured_mask_to_operand') for ctx in self.all_ctxes]
        handler.populate_dependent_structured_mask_to_operand()
        for mock_method in mock_methods:
            mock_method.assert_called_once()
