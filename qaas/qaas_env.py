import os
import re
import networkx as nx
from nncf.torch.graph.graph import PTNNCFGraph
from networkx.drawing.nx_agraph import to_agraph
import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
import numpy as np
# PALETTE = np.array(list(mcd.CSS4_COLORS.keys())).reshape(-1, 4).transpose().reshape(-1).tolist()
from matplotlib.colors import to_hex
PALETTE = np.array([to_hex(c) for c in plt.get_cmap("tab20b").colors]).reshape(-1, 5).transpose().reshape(-1).tolist()
from collections import Counter
from collections import defaultdict
from copy import deepcopy
import json

from nncf.torch.quantization.algo import QuantizationController
from nncf.torch.automl.environment.quantization_env import QuantizationEnv, QuantizationEnvParams
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.common.quantization.structs import NonWeightQuantizerId
from nncf.common.quantization.structs import WeightQuantizerId

class Qaas:
    def __init__(self, 
                 quantization_controller, 
                 quantized_model, 
                 nncf_cfg, eval_fn, val_loader, test_loader):

        if isinstance(quantization_controller, QuantizationController):
            self.qctrl = quantization_controller #FIXME this controller is just a handle to access qenv, its context is not corresponding qenv
            self.qmodel = quantized_model #FIXME do we need this?
            self.qenv = quantization_controller.qenv
            self.nncf_cfg = nncf_cfg
            self.qenv.nncf_config = self.nncf_cfg # This is for batchnorm adaptation! qenv.qctrl.config will be lost due to regeneration of experimental qctrl
            self.bool_perf_bw = self.nncf_cfg.get("quantizer_coupling", False)
            
            self.qenv.eval_fn = eval_fn # override autoq eval fn that only return top5, the new eval_fn should be returning top1, and top5 and loss for imagenet case
            self.val_loader = val_loader
            self.test_loader = test_loader
        else:
            raise ValueError("Qaas requires a quantization wrapped controller and model")
        self.base_ft_cfg = self.get_quantization_algo_cfg()
        assert self.base_ft_cfg is not None, "Qaas does not instantiated with a valid nncf cfg that contains quantization"
        
        self.g = self.qenv.qctrl.model.get_graph()
        self.qid_to_nncfnode_map, self.nncfnode_to_qid_map = self.create_qid_to_nncfnode_map()
        self.visualize_adjacent_quantizers()
        self.feature_df = self.extract_quantizable_layer_features()
        self.node_name_to_qenv_index = self.feature_df['node_name'].reset_index().set_index('node_name').to_dict()['index']
        self.node_type_lut, self.connectivity_lut = self.extract_graph_connectivity()
        # self.print_groupwise_nodes()

    @property
    def bw_space(self):
        return self.qenv.model_bitwidth_space

    @property
    def bool_perf_bw(self):
        return self.qenv.performant_bw

    @bool_perf_bw.setter
    def bool_perf_bw(self, bool_val):
        self.qenv.performant_bw = bool_val

    @property
    def original_model_size(self):
        return int(self.qenv.orig_model_size)

    @property
    def original_bop(self):
        return int(self.qenv.compression_ratio_calculator.maximum_bits_complexity * 4) # orignally 8bit as baseline, normalized to 32bit

    def get_quantization_algo_cfg(self):
        def _finditem(obj, key):
            if isinstance(obj, list):
                for e in obj:
                    item = _finditem(e, key) 
                    if item is not None:
                        return item
            elif isinstance(obj, dict):
                if key in obj: return obj
                for k, v in obj.items():
                    item = _finditem(v, key)
                    if item is not None:
                        return item
            return None
        compression = _finditem(self.nncf_cfg, 'algorithm')
        if compression is not None and compression['algorithm'] == 'quantization':
            base_ft_cfg =  deepcopy(self.nncf_cfg)
        return base_ft_cfg

    def generate_ft_cfg(self):
        # this generates based on what is captured in action of master_df

        bitwidth_per_scope = [[bw, qp] for qp, bw in self.qenv.master_df['action'].to_dict().items()]

        # bitwidth_per_scope = []
        # for qp in self.feature_df.index.to_list():
        #     bw = input_bw_cfg[qp]
        #     assert bw in self.bw_space, "invalid bitwidth"
        #     bitwidth_per_scope.append([int(bw), qp])
        # assert len(bitwidth_per_scope) == len(self.feature_df), "unexpected length of input_bw_cfg"

        ft_cfg = deepcopy(self.base_ft_cfg)

        if 'log_dir' in ft_cfg:
            service_str = '\n\n// QAAS service on | {} | at log path: | {} |'.format(os.uname().nodename, ft_cfg['log_dir'])
        else:
            service_str = None
            
        for key in ['log_dir', 'episodic_nncfcfg', 'restful', 'eval_cache', 'quantizer_coupling']:
            if key in ft_cfg:
                del ft_cfg[key]

        ft_cfg['compression']['initializer']['precision'] = dict(bitwidth_per_scope=bitwidth_per_scope)

        bw_dist_comment = "\n// " + self.qenv.qctrl.statistics().to_str().replace("\n","\n// ")
        if service_str is not None:
            bw_dist_comment += service_str

        ft_cfg_str = json.dumps(ft_cfg, indent=4)
        index_of_last_closing_curly_bracket = ft_cfg_str.rfind('}') 
        return ft_cfg_str[0:index_of_last_closing_curly_bracket]+bw_dist_comment+"\n}"

    def generate_sample_request(self):
        def rand_bw():
            return int(np.random.choice(self.bw_space, size=1, replace=True, p=None)[0])

        bw_cfg = dict()
        for node, _ in self.node_type_lut.items():
            if node in self.node_name_to_qenv_index:
                bw_cfg[node] = rand_bw()
            else:
                bw_cfg[node] = 0
        
        sample_config = dict(bw_cfg=bw_cfg, bool_bnadap=True)
        return sample_config

    def _setup_bnadap_pipeline(self, bool_bnadap):
        if bool_bnadap is True:
            self.qenv.nncf_config = self.nncf_cfg 
        else:
            self.qenv.nncf_config = None

    def evaluate_valset(self, bw_cfg, bool_bnadap):
        self._setup_bnadap_pipeline(bool_bnadap)
        self.qenv.eval_loader=self.val_loader
        strategy = [bw_cfg[id] for id in self.qenv.master_df.index]
        return self.qenv.evaluate_strategy(strategy, skip_constraint=True) # always skip constraints

    def evaluate_testset(self, bw_cfg, bool_bnadap):
        self._setup_bnadap_pipeline(bool_bnadap)
        self.qenv.eval_loader=self.test_loader
        strategy = [bw_cfg[id] for id in self.qenv.master_df.index]
        return self.qenv.evaluate_strategy(strategy, skip_constraint=True) # always skip constraints

    def _to_nx_node(self, nncfnode):
        nx_node = '{} {}'.format(str(nncfnode.node_id), nncfnode.node_name)
        return nx_node
    
    def _qid_to_node_name(self, qid_obj):
        target_nncfnode = self.g.get_node_by_name(qid_obj.target_node_name)
        return self._to_nx_node(target_nncfnode)

    def extract_quantizable_layer_features(self):
        # ['qid', 'gid', 'qconf_space', 'qp_id_set', 'state_scope', 'qid_obj',
        # 'qmodule', 'is_wt_quantizer', 'state_module', 'cin', 'conv_dw', 'cout',
        # 'ifm_size', 'kernel', 'param', 'prev_action', 'stride', 'layer_idx',
        # 'weight_quantizer', 'n_op', 'action', 'unconstrained_action']
        feature_cols = ['gid', 'is_wt_quantizer', 'cin', 'conv_dw', 'cout', 'ifm_size', 'kernel', 'param', 'stride']
        feature_df = self.qenv.master_df[feature_cols]
        # node_name is actually nx_node where nncfnode is prefixed with node id
        feature_df['node_name'] = self.qenv.master_df['qid_obj'].apply(lambda x: self._qid_to_node_name(x))
        return feature_df

    def extract_graph_connectivity(self):       
        g = self.g
        # nx_digraph = g.get_graph_for_structure_analysis()

        node_type = dict()
        for nncfnode in g.get_all_nodes():
            node_type[self._to_nx_node(nncfnode)] = nncfnode.node_type

        edge_connectivity = defaultdict(list) # key: src_node, val: set(dst_nodes)
        for nncfedge in g.get_all_edges():
            src_node = self._to_nx_node(nncfedge.from_node)
            dst_node = self._to_nx_node(nncfedge.to_node)
            edge_connectivity[src_node].append(dst_node)
        return node_type, edge_connectivity

    def create_qid_to_nncfnode_map(self):
        g = self.g
        qid_to_nncfnode_map = dict()
        
        for qid, qmod in self.qenv.qctrl.all_quantizations.items():
            target_nncfnode = g.get_node_by_name(qid.target_node_name)
            if isinstance(qid, WeightQuantizerId):
                wquantize_nncfnode = g.get_node_by_id(target_nncfnode.node_id-1)
                assert 'UpdateWeight' in wquantize_nncfnode.node_name, "Logical Bug"
                qid_to_nncfnode_map[qid] = g.get_node_by_name(wquantize_nncfnode.node_name)
            elif isinstance(qid, NonWeightQuantizerId):
                aquantize_nncfnode = g.get_node_by_id(target_nncfnode.node_id+1)
                assert 'quantize' in aquantize_nncfnode.node_type, "Logical Bug"
                qid_to_nncfnode_map[qid] = g.get_node_by_name(aquantize_nncfnode.node_name)
            else:
                raise NotImplementedError
            # qid_target_opaddr = OperationAddress.from_str(qid.target_node_name)

            # pattern = '/'.join(qid.target_node_name.split('/')[0:-1])
            # pattern = '.+{}.+quantize.+{}$'.format(pattern, qid_target_opaddr.call_order)

            # for node_name, node in g._nx_graph.nodes.items():
            #     # if all(map(node_name.__contains__, [pattern, "quantize_{}".format(qid_target_opaddr.call_order)])):
            #     # if re.search(pattern, node_name):
                
            #     if all(map(node_name.__contains__, [pattern, 'quantize'])):
            #         if node_name in qid_to_nncfnode_map:
            #             raise KeyError("{} key exists! This should not happen, pls debug".format(node_name))
            #         qid_to_nncfnode_map[qid] = node
            nncfnode_to_qid_map = {v:k for k,v in qid_to_nncfnode_map.items()}
        return qid_to_nncfnode_map, nncfnode_to_qid_map
        
    def visualize_adjacent_quantizers(self, path=None):
        node_color_map = dict()
        node_style_map = dict()
        # At present, there are 8 style values recognized: filled , invisible , diagonals , rounded . dashed , dotted , solid and bold

        for group_id, adjq_group in enumerate(self.qctrl.qenv.qctrl.groups_of_adjacent_quantizers):
            color = PALETTE[group_id % len(PALETTE)]
            for wqid, wqmod in adjq_group.weight_quantizers:
                node_color_map[wqid.target_node_name] = color
                node_style_map[wqid.target_node_name] = "diagonals"
                node_color_map[
                    self.qid_to_nncfnode_map[wqid].node_name
                ] = color
                node_style_map[
                    self.qid_to_nncfnode_map[wqid].node_name
                ] = "filled"

            for aqid, aqmod in adjq_group.activation_quantizers:
                if 'nncf_model_input' in aqid.target_node_name:
                    continue
                node_color_map[aqid.target_node_name] = color
                node_style_map[aqid.target_node_name] = "diagonals"
                node_color_map[
                    self.qid_to_nncfnode_map[aqid].node_name
                ] = color
                node_style_map[
                    self.qid_to_nncfnode_map[aqid].node_name
                ] = "filled"

        g = self.qenv.qctrl.model.get_graph()

        out_graph = nx.DiGraph()
        for node_name, node in g._nx_graph.nodes.items():
            attrs_node = {}
            label = node['key']

            tokens=label.split("/")
            new_tokens=[]
            for i, token in enumerate(tokens):
                if (i+1)%2==0:
                    token += "\n"
                new_tokens.append(token)
            attrs_node['label'] = '/'.join(new_tokens)

            if node['node_name'] in node_color_map:               
                attrs_node['color'] = node_color_map[node['node_name']]

            if node['node_name'] in node_style_map:               
                attrs_node['style'] = node_style_map[node['node_name']]
                                
            out_graph.add_node(node_name, **attrs_node)

        for u, v in g._nx_graph.edges:
            out_graph.add_edge(u, v, label=g._nx_graph.edges[u, v][PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])

        mapping = {k: v["label"] for k, v in out_graph.nodes.items()}
        out_graph = nx.relabel_nodes(out_graph, mapping)
        for node in out_graph.nodes.values():
            node.pop("label")

        if path is None:
            path = 'adjq_group_viz.dot'
        path = os.path.join(self.nncf_cfg.get("log_dir", "."), path)
        
        nx.drawing.nx_pydot.write_dot(out_graph, path)

        try:
            A = to_agraph(out_graph)
            A.layout('dot')
            png_path = os.path.splitext(path)[0]+'.png'
            A.draw(png_path)
        except ImportError:
            print("Graphviz is not installed - only the .dot model visualization format will be used. "
                                "Install pygraphviz into your Python environment and graphviz system-wide to enable "
                                "PNG rendering.")

