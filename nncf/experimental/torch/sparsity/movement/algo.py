"""
 Copyright (c) 2019-2020 Intel Corporation
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
from copy import deepcopy
from typing import DefaultDict, List, OrderedDict, Optional
from collections import OrderedDict
import torch
import torch.distributed as dist

from nncf import NNCFConfig
from nncf.config.extractors import extract_algo_specific_config
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.api.compression import CompressionStage
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.sparsity.base_algo import BaseSparsityAlgoBuilder, BaseSparsityAlgoController, SparseModuleInfo
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import TransformationPriority
from nncf.experimental.torch.sparsity.movement.layers import MovementSparsifier, SparseConfig, SparseStructure
from nncf.experimental.torch.sparsity.movement.layers import SparseConfigByScope
from nncf.experimental.torch.sparsity.movement.loss import ImportanceLoss, SparseLossForPerLayerSparsity
from nncf.experimental.torch.sparsity.movement.structured_mask_handler import StructuredMaskHandler, SparsifiedModuleInfoGroup
from nncf.torch.module_operations import UpdateWeightAndBias
from nncf.torch.utils import get_world_size, get_model_device
from nncf.common.utils.helpers import matches_any
from nncf.common.accuracy_aware_training.training_loop import ADAPTIVE_COMPRESSION_CONTROLLERS
from nncf.torch.sparsity.collector import PTSparseModelStatisticsCollector
from nncf.common.sparsity.schedulers import SPARSITY_SCHEDULERS
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.sparsity.statistics import MovementSparsityStatistics
from nncf.common.statistics import NNCFStatistics
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlock, get_building_blocks, BuildingBlockType, BlockFilteringStrategy
from collections import defaultdict, namedtuple
from nncf.torch.dynamic_graph.operation_address import OperationAddress
import networkx as nx
from nncf.torch.layers import NNCF_MODULES_OP_NAMES, NNCFLinear
import os
import numpy as np
import pandas as pd
from nncf.experimental.torch.sparsity.movement.structured_mask_strategy import STRUCTURED_MASK_STRATEGY, detect_supported_model_family

SUPPORTED_NNCF_MODULES = [NNCFLinear]


@PT_COMPRESSION_ALGORITHMS.register('movement_sparsity')
class MovementSparsityBuilder(BaseSparsityAlgoBuilder):
    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)
        configs = self._algo_config.get('sparse_structure_by_scopes', [])
        self._sparse_configs_by_scopes = [SparseConfigByScope.from_config(c) for c in configs]

    def _sparsify_weights(self, target_model: NNCFNetwork) -> List[PTInsertionCommand]:
        device = get_model_device(target_model)
        sparsified_module_nodes = target_model.get_weighted_original_graph_nodes(
            nncf_module_names=[m.__name__ for m in SUPPORTED_NNCF_MODULES])
        insertion_commands = []
        for module_node in sparsified_module_nodes:
            node_name = module_node.node_name

            if not self._should_consider_scope(node_name):
                nncf_logger.info("Ignored adding Weight Sparsifier in scope: {}".format(node_name))
                continue

            nncf_logger.info("Adding Weight Sparsifier in scope: {}".format(node_name))
            compression_lr_multiplier = \
                self.config.get_redefinable_global_param_value_for_algo('compression_lr_multiplier',
                                                                        self.name)
            sparsifying_operation = self.create_weight_sparsifying_operation(module_node, compression_lr_multiplier)
            hook = UpdateWeightAndBias(sparsifying_operation).to(device)
            insertion_commands.append(
                PTInsertionCommand(
                    PTTargetPoint(TargetType.PRE_LAYER_OPERATION, target_node_name=node_name),
                    hook,
                    TransformationPriority.SPARSIFICATION_PRIORITY
                )
            )
            sparsified_module = target_model.get_containing_module(node_name)
            self._sparsified_module_info.append(
                SparseModuleInfo(node_name, sparsified_module, sparsifying_operation))

        return insertion_commands

    def create_weight_sparsifying_operation(self, target_module_node: NNCFNode, compression_lr_multiplier: float):
        sparse_cfg = SparseConfig(SparseStructure.FINE)
        node_name = target_module_node.node_name
        for configs_per_scopes in self._sparse_configs_by_scopes:
            target_scopes = configs_per_scopes.target_scopes
            if matches_any(node_name, target_scopes):
                sparse_cfg = configs_per_scopes.sparse_config
                break

        return MovementSparsifier(target_module_node, sparse_cfg=sparse_cfg, frozen=False,
                                  compression_lr_multiplier=compression_lr_multiplier)

    def _build_controller(self, model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return MovementSparsityController(model, self._sparsified_module_info, self.config)


class PrunableOp:
    def __init__(self, op_addr: OperationAddress, op_mod: Optional[torch.nn.Module]):
        self.op_addr = op_addr
        self.op_mod = op_mod


@ADAPTIVE_COMPRESSION_CONTROLLERS.register('pt_movement_sparsity')
class MovementSparsityController(BaseSparsityAlgoController):
    def __init__(self, target_model: NNCFNetwork, sparsified_module_info: List[SparseModuleInfo],
                 config: NNCFConfig):
        super().__init__(target_model, sparsified_module_info)
        algo_config = extract_algo_specific_config(config, 'movement_sparsity')
        self._distributed = False
        sparsify_operations = [m.operand for m in self.sparsified_module_info]
        params = deepcopy(algo_config.get('params', {}))
        scheduler_cls = SPARSITY_SCHEDULERS.get('threshold_polynomial_decay')  # TODO(yujie): hard coded this scheduler name
        self._scheduler = scheduler_cls(self, params)
        self._loss = ImportanceLoss(sparsify_operations, self.scheduler)

        # TODO: review - perhaps not the right place
        self.config = config
        # self.visualize_groups_of_prunables()
        # self.create_structured_sparsity_context()
        # self.prunable_sparsified_module_info_groups = self._get_group_of_prunable_sparsified_module_info()

        if self._scheduler.enable_structured_masking:
            model_family = detect_supported_model_family(self.model)
            if model_family not in STRUCTURED_MASK_STRATEGY.registry_dict:
                raise RuntimeError("You set `enable_structured_masking=True`, but no supported model is detected. "
                                   "Supported model families: {}".format(list(STRUCTURED_MASK_STRATEGY.keys())))
            strategy_cls = STRUCTURED_MASK_STRATEGY.get(model_family)
            strategy = strategy_cls.from_compressed_model(self.model)
            self._structured_mask_handler = StructuredMaskHandler(self.model,
                                                                  self.sparsified_module_info,
                                                                  strategy)

    def compression_stage(self) -> CompressionStage:
        # if self._mode == 'local':
        #     return CompressionStage.FULLY_COMPRESSED
        if self.scheduler.current_epoch < self.scheduler.warmup_start_epoch:
            return CompressionStage.UNCOMPRESSED
        if self.scheduler.current_sparsity_level >= self.scheduler.warmup_end_epoch:
            return CompressionStage.FULLY_COMPRESSED
        return CompressionStage.PARTIALLY_COMPRESSED

    def freeze(self):
        self._loss.disable()

    def distributed(self):
        if not dist.is_initialized():
            raise KeyError('Could not set distributed mode for the compression algorithm '
                           'because the default process group has not been initialized.')

        if next(self._model.parameters()).is_cuda:
            state = torch.cuda.get_rng_state()
            if dist.get_backend() == dist.Backend.NCCL:
                state = state.cuda()
            torch.distributed.broadcast(state, src=0)
            torch.cuda.set_rng_state(state.cpu())
        else:
            state = torch.get_rng_state()
            torch.distributed.broadcast(state, src=0)
            torch.set_rng_state(state)

        self._distributed = True

    def _check_distributed_masks(self):
        if not self._distributed or get_world_size() == 1:
            return 1

        nvalues = 0
        ncor_values = 0
        eps = 1e-4
        for minfo in self.sparsified_module_info:
            mask = minfo.operand.mask

            mask_list = [torch.empty_like(mask) for _ in range(get_world_size())]
            # nccl does not support gather, send, recv operations
            dist.all_gather(mask_list, mask)

            for i in range(1, len(mask_list)):
                rel_error = (mask_list[0] - mask_list[i]) / mask_list[0]
                ncor_values = ncor_values + (rel_error.abs() < eps).sum(dtype=mask.dtype)
                nvalues = nvalues + mask_list[i].numel()

        return ncor_values / nvalues

    def statistics(self, quickly_collected_only=False) -> NNCFStatistics:
        collector = PTSparseModelStatisticsCollector(self.model, self.sparsified_module_info)
        model_statistics = collector.collect()

        stats = MovementSparsityStatistics(model_statistics,
                                           self.scheduler.current_importance_threshold,
                                           self.scheduler.current_importance_lambda)

        nncf_stats = NNCFStatistics()
        nncf_stats.register('movement_sparsity', stats)
        return nncf_stats

    def reset_independent_structured_mask(self):
        assert self._scheduler.enable_structured_masking is True
        self._structured_mask_handler.update_independent_structured_mask()

    def resolve_structured_mask(self):
        assert self._scheduler.enable_structured_masking is True
        self._structured_mask_handler.resolve_dependent_structured_mask()

    def populate_structured_mask(self):
        assert self._scheduler.enable_structured_masking is True
        self._structured_mask_handler.populate_dependent_structured_mask_to_operand()

    @property
    def compression_rate(self):
        return self.statistics().movement_sparsity.model_statistics.sparsity_level

    def prepare_for_export(self):
        """
        Applies pruning masks to layer weights before exporting the model to ONNX.
        """
        self._propagate_masks()

    def _propagate_masks(self):
        sparse_sd = OrderedDict()
        with torch.no_grad():
            for sparse_info in self.sparsified_module_info:
                for n, m in self.model.named_modules():
                    if m == sparse_info.module:
                        sparse_sd[n + '.weight'] = sparse_info.operand.apply_binary_mask(m.weight)
                        if hasattr(m, 'bias'):
                            sparse_sd[n + '.bias'] = sparse_info.operand.apply_binary_mask(m.bias, isbias=True)

        model_sd = self.model.state_dict()
        for k, v in sparse_sd.items():
            assert k in model_sd, "key not exists!"
            model_sd[k] = sparse_sd[k]
        self.model.load_state_dict(model_sd)

    def __delete_propagate_masks(self):
        def calc_sparsity(tensor):
            return 1 - tensor.count_nonzero() / tensor.numel()
        # nncf_logger.debug("MVMT - Propagating pruning masks")
        # 1. Propagate masks for all modules
        from collections import OrderedDict
        sparse_sd = OrderedDict()
        with torch.no_grad():
            for sparse_info in self.sparsified_module_info:
                for n, m in self.model.named_modules():
                    if m == sparse_info.module:
                        # print("- SparseModule: {} -".format(n))
                        # print("\tw_mask sparsity: {:.3f}".format(calc_sparsity(sparse_info.operand.weight_ctx.binary_mask)))
                        # print("\tw_sd   sparsity: {:.3f}".format(calc_sparsity(m.weight)))
                        sparse_sd[n + '.weight'] = sparse_info.operand.apply_binary_mask(m.weight)
                        # print("\t*w_sd  sparsity: {:.3f}".format(calc_sparsity(sparse_sd[n+'.weight'])))

                        if hasattr(m, 'bias'):
                            # print("\tb_mask sparsity: {:.3f}".format(calc_sparsity(sparse_info.operand.bias_ctx.binary_mask)))
                            # print("\tb_sd   sparsity: {:.3f}".format(calc_sparsity(m.bias)))
                            sparse_sd[n + '.bias'] = sparse_info.operand.apply_binary_mask(m.bias, isbias=True)
                            # print("\t*w_sd  sparsity: {:.3f}".format(calc_sparsity(sparse_sd[n+'.bias'])))

        model_sd = self.model.state_dict()
        for k, v in sparse_sd.items():
            assert k in model_sd, "key not exists!"
            model_sd[k] = sparse_sd[k]
        self.model.load_state_dict(model_sd)

    def print_prunableops_per_group(self):
        for group, op_list in self.prunableops_per_group.items():
            print("= Group {} ======".format(group))
            print('\n'.join(list(map(lambda x: '{:12} | {}'.format(str(list(x.op_mod.weight.shape)), str(x.op_addr)), op_list))))

    def _get_all_node_op_addresses_in_block(self, nncf_network, blocks):
        graph = nncf_network.get_original_graph()
        all_nodes_per_skipped_block_idxs = {}
        for idx, block in enumerate(blocks):
            start_node, end_node = block.start_node_name, block.end_node_name
            start_node_key, end_node_key = None, None
            for node in graph._nx_graph._node.values():
                if start_node == str(node['node_name']):
                    start_node_key = node['key']
                if end_node == str(node['node_name']):
                    end_node_key = node['key']
            simple_paths = nx.all_simple_paths(graph._nx_graph, start_node_key, end_node_key)
            all_nodes_in_block = set()
            for node_keys_in_path in simple_paths:
                for node_key in node_keys_in_path:
                    all_nodes_in_block.add(str(graph._nx_graph._node[node_key]['node_name']))
            start_op_address = str(graph._nx_graph._node[start_node_key]['node_name'])
            all_nodes_in_block.remove(start_op_address)
            all_nodes_per_skipped_block_idxs[idx] = list(all_nodes_in_block)
        return all_nodes_per_skipped_block_idxs

    def visualize_groups_of_prunables(self, path=None):
        import networkx as nx
        from nncf.torch.graph.graph import PTNNCFGraph
        from networkx.drawing.nx_agraph import to_agraph
        import matplotlib._color_data as mcd
        import matplotlib.pyplot as plt
        import numpy as np
        palette = np.array(list(mcd.CSS4_COLORS.keys())).reshape(-1, 4).transpose().reshape(-1).tolist()

        from matplotlib.colors import to_hex
        palette = np.array([to_hex(c) for c in plt.get_cmap("tab20b").colors]).reshape(-1, 5).transpose().reshape(-1).tolist()

        learnable_node_color_map = dict()
        opbook = dict()

        for group_id, op_list in self.prunableops_per_group.items():
            color = palette[group_id % len(palette)]
            for op in op_list:
                learnable_node_color_map[str(op.op_addr)] = color
                opbook[str(op.op_addr)] = op

        building_blocks = get_building_blocks(self.model, allow_nested_blocks=False)
        node_op_address_per_block = self._get_all_node_op_addresses_in_block(self.model, building_blocks)
        node_color_map = dict()
        for group_id, op_list in node_op_address_per_block.items():
            color = palette[group_id % len(palette)]
            for op in op_list:
                node_color_map[op] = color

        g = self.model.get_graph()

        out_graph = nx.DiGraph()
        for node_name, node in g._nx_graph.nodes.items():
            # ia_op_exec_context = node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR]

            attrs_node = {}
            label = node['key']
            # label = str(node[PTNNCFGraph.ID_NODE_ATTR]) + ' ' + str(ia_op_exec_context)
            # if 'conv2d' in label.lower():
            #     label = "*prunable*\n" + label
            tokens = label.split("/")
            new_tokens = []
            for i, token in enumerate(tokens):
                if (i + 1) % 2 == 0:
                    token += "\n"
                new_tokens.append(token)
            attrs_node['label'] = '/'.join(new_tokens)

            if node['node_name'] in node_color_map:
                # cluster_id = self.df.cluster_id[self.df.node_name == node_name].values[0]
                # attrs_node['label'] += "\n(cluster {})".format(cluster_id)
                # mcd.CSS4_COLORS
                # attrs_node['color'] = mcd.CSS4_COLORS[node_color_map[node['node_name']]]

                attrs_node['color'] = node_color_map[node['node_name']]
                if node['node_name'] in learnable_node_color_map:
                    attrs_node['label'] += "\n{}\n".format(str(tuple(opbook[node['node_name']].op_mod.weight.shape)))
                    attrs_node['style'] = 'filled'
                else:
                    attrs_node['style'] = 'diagonals'
                    # At present, there are 8 style values recognized: filled , invisible , diagonals , rounded . dashed , dotted , solid and bold

            out_graph.add_node(node_name, **attrs_node)

        for u, v in g._nx_graph.edges:
            out_graph.add_edge(u, v, label=g._nx_graph.edges[u, v][PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])

        mapping = {k: v["label"] for k, v in out_graph.nodes.items()}
        out_graph = nx.relabel_nodes(out_graph, mapping)
        for node in out_graph.nodes.values():
            node.pop("label")

        if path is None:
            path = 'mvmt_prunableops_group_viz.dot'
        path = os.path.join(self.config.get("log_dir", "."), path)

        nx.drawing.nx_pydot.write_dot(out_graph, path)

        try:
            A = to_agraph(out_graph)
            A.layout('dot')
            png_path = os.path.splitext(path)[0] + '.png'
            A.draw(png_path)
        except ImportError:
            print("Graphviz is not installed - only the .dot model visualization format will be used. "
                  "Install pygraphviz into your Python environment and graphviz system-wide to enable "
                  "PNG rendering.")
