import os,sys
from typing import Protocol
from flask import Flask, g
from flask import request, send_file, abort
import json, random
import numpy as np
import gc
import time
from multiprocessing import Lock, Semaphore
from multiprocessing.sharedctypes import Value
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from natsort import natsorted
from collections import OrderedDict
import hashlib
lock = Semaphore(1)
mutex= Lock()

from .qaas_env import Qaas
from examples.torch.classification.main import main as imgnet
from run_qa import main as qa
from copy import deepcopy
import logging, pandas
from datetime import datetime
log = logging.getLogger('werkzeug')
log.setLevel(logging.INFO)
def prRed(prt): print("\033[91m {}\033[00m".format(prt),flush=True)
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt),flush=True)
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt),flush=True)

def init_workload():
    if os.environ['workload'] == 'imgnet':
        _args = [
            '--gpu-id', '0', 
            '--workers', '6', 
            '--log-dir', '/tmp/qaas-imgnet-log/',  
            '--config', os.environ['config'],
            '--data',   os.environ['data']]
        return Qaas(*imgnet(_args))
    elif os.environ['workload'] == 'bert-squad':
        _args = [
            "--model_name_or_path",
            "vuiseng9/bert-base-uncased-squad", 
            "--dataset_name", "squad",
            "--do_eval",
            "--do_predict",
            "--per_device_eval_batch_size", "240",
            "--max_seq_length", "384",
            "--doc_stride", "128",
            "--nncf_config", os.environ['config'],
            "--output_dir", '/tmp/qaas-bert-squad-log/',
            "--overwrite_output_dir"
        ]
        # handling for qa evaluate with val and testset
        compression_ctrl, model, nncf_config, autoq_validate, autoq_predict = qa(_args)
        env = Qaas(compression_ctrl, model, nncf_config, None, None, None)
        env.validate_fn = autoq_validate
        env.test_fn = autoq_predict
        return env
        
    else:
        raise ValueError("Environment variable workload is not valid")

def dummy_response():
    return 	{
        'rc': 0, 
        'done': 1, 
        'reward': random.uniform(0.0, 1.4), 
        'meta_data': {
            'acccuracy': random.uniform(0.0, 0.95), 
            'modelsize': random.uniform(0.0625, 0.25)
            }
        }

def acquire_lock(calling_method):
    global concurrent_requests_value
    global last_lock_time

    try:
        if lock.acquire(block=False):
            last_lock_time = time.time()
            concurrent_requests_value += 1
            prCyan("acquired_lock - " + calling_method+", concurrent_requests_value="+str(concurrent_requests_value))
            return True
        else:
            prRed("Blocked since: " + str(time.time()-last_lock_time))
            return  False
    except:
        prRed("BlockedY")
        pass

def release_lock(calling_method):
    global concurrent_requests_value
    global last_lock_time
    lock.release()
    last_lock_time = time.time()
    concurrent_requests_value -= 1
    prCyan("release_lock - " + calling_method+", concurrent_requests="+str(concurrent_requests_value))
    return  True
        
def create_app() -> Flask:
        
    global concurrent_requests_value
    global max_thread_time
    global hostname
    hostname = os.uname().nodename
    bEnvReady = False
    concurrent_requests_value = 0

    '''Create an app by initializing components'''
    app = Flask(__name__)
    acquire_lock("WorkloadInit")
    t1 = time.time()
    env = init_workload()
    max_thread_time = int(5*(time.time()-t1))
    prRed("max_thread_time="+str(max_thread_time)+" sec.")
    release_lock("WorkloadInit")

    bEnvReady = True
    print("{} Quantization Service initialized".format(os.environ['workload']), flush=True)

    @app.route('/ready_state')
    def readystate():
        if bEnvReady is True:
            return {'method':'ready_state', 'rc': 0, 'msg':"Environment {} initialized".format(os.environ['workload']), 'config':os.environ['config']}
        return {'method': 'ready_state', 'rc': 1, 'msg': "Environment {} not yet initialzed".format(os.environ['workload']), 'config':os.environ['config']}

    #FIXME Doesnt seem to be used anywhere or client, do we need this? Good to keep for now
    @app.route('/ready')
    def ready():

        global concurrent_requests_value
        global last_lock_time
        global max_thread_time
        from flask import jsonify
        bOK = True
        if bEnvReady:
            if concurrent_requests_value>0:
                bOK = False
        else:
            prRed("Service yet not initialized")
            bOK = False
        if bOK:
            return jsonify(success=True, concurrent_requests_value=concurrent_requests_value, lock_time=time.time()-last_lock_time), 200
        else:
            if time.time() - last_lock_time > 400:
                prRed("1 Service blocked for {} min.".format((time.time() - last_lock_time) / 60.))
            if (time.time() - last_lock_time) > max_thread_time:
                while (concurrent_requests_value > 0):
                    release_lock("Force release lock")
            return jsonify(success=False, concurrent_requests_value=concurrent_requests_value, lock_time=time.time()-last_lock_time), 300

    @app.route('/layer_graph_data')
    def layer_graph():
        if acquire_lock("creating_layer_wise_graph"):
            try:
                jsonresponse = dict()
                jsonresponse['edges'] = env.connectivity_lut

                quantizable_attr = env.feature_df.to_dict()
                quantizable_df = pd.DataFrame.from_dict(quantizable_attr)
                node2gid = quantizable_df[['node_name', 'gid']].set_index('node_name').to_dict()

                node_type = env.node_type_lut

                optype_encoder = OneHotEncoder()
                gid_encoder = OneHotEncoder()

                one_hot_encoded_gid = gid_encoder.fit_transform(np.array(list(node2gid['gid'].values())).reshape(-1,1)).toarray()
                one_hot_encoded_gid_map = dict(zip(node2gid['gid'].keys(), one_hot_encoded_gid))
                gid_df = pd.Series(data=one_hot_encoded_gid_map.values(), index=one_hot_encoded_gid_map.keys(), name="gid").to_frame()

                one_hot_encoded_optype = optype_encoder.fit_transform(np.array(list(node_type.values())).reshape(-1,1)).toarray()
                one_hot_encoded_optype_map = dict(zip(node_type.keys(), one_hot_encoded_optype))
                optype_df = pd.Series(data=one_hot_encoded_optype_map.values(), index=one_hot_encoded_optype_map.keys(), name="optype").to_frame()
                optype_df = optype_df.reindex(natsorted(optype_df.index))

                quantizable_features = quantizable_df.drop(columns=['gid']).set_index('node_name')
                features_per_node = pd.concat([optype_df, quantizable_features], axis=1).fillna(0)
                features_per_node = pd.concat([features_per_node, gid_df], axis=1)

                features_per_node['gid'] = \
                    [np.zeros_like(one_hot_encoded_gid[0]) if isinstance(val, float) else val for val in features_per_node['gid'].to_list()]

                for ii, item in enumerate(features_per_node['gid']):
                    features_per_node['gid'][ii] = features_per_node['gid'][ii].tolist()
                    features_per_node['optype'][ii] = features_per_node['optype'][ii].tolist()

                if 'target_optype' in features_per_node.columns:
                    null_target_optype = [0.0] * len(quantizable_df.target_optype[0])
                    d = OrderedDict()
                    for iii, id in enumerate(features_per_node.index):
                        if features_per_node.loc[id, 'target_optype'] == 0.0:
                            d[id] = null_target_optype
                        else:
                            d[id] = features_per_node.loc[id, 'target_optype']
                    features_per_node['target_optype'] = pd.Series(d)

                feature_dict = features_per_node.to_dict()

                jsonresponse['node_features'] = feature_dict                
                jsonresponse['action_space'] = {'discrete': env.bw_space}
                jsonresponse['rc'] = 1

                prRed("Sending connectivity and nodes features...")
            finally:
                gc.collect()
                release_lock("creating_layer_wise_graph")
            return jsonresponse
        else:
            jsonresponse = {'rc': -1, 'msg': 'Server busy'}
            return jsonresponse


    @app.route('/get_model_graph_viz')
    def get_model_graph_viz():
        graph_imgpth = os.path.join(env.nncf_cfg['log_dir'],'adjq_group_viz.png')
        try:
            return send_file(graph_imgpth, mimetype='image/png')
        except FileNotFoundError:
            if bEnvReady is True:
                return {'method':'get_model_graph_viz', 'msg':"Env ready, model graph image not found"}
            return {'method': 'get_model_graph_viz', 'msg': "Env not ready, model graph image not found"}

    @app.route('/get_node2optype_map')
    def get_node2optype_map():
        if acquire_lock("get_node2optype_map"):
            try:
                prRed("Sending dictionary of nodetype per nodes...")
                jsonresponse= env.node_type_lut
            finally:
                gc.collect()
                release_lock("get_node2optype_map")
            return jsonresponse
        else:
            jsonresponse = {'rc': -1, 'msg': 'Server busy'}
            return jsonresponse
    
    @app.route('/get_connectivity_map')
    def get_connectivity_map():
        if acquire_lock("get_node_connectivity_map"):
            try:
                prRed("Sending connectivity per source nodes...")
                jsonresponse= env.connectivity_lut
            finally:
                gc.collect()
                release_lock("get_connectivity_map")
            return jsonresponse
        else:
            jsonresponse = {'rc': -1, 'msg': 'Server busy'}
            return jsonresponse

    @app.route('/get_quantizable_attr')
    def get_quantizable_attr():
        if acquire_lock("get_quantizable_attr"):
            try:
                prRed("Sending attributes of quantizable nodes...")
                jsonresponse= env.feature_df.to_dict()
            finally:
                gc.collect()
                release_lock("get_quantizable_attr")
            return jsonresponse
        else:
            jsonresponse = {'rc': -1, 'msg': 'Server busy'}
            return jsonresponse

    @app.route('/sample_eval_req')
    def sample_eval_req():
        if acquire_lock("sample_eval_req"):
            try:
                prRed("Sending qaas sample request...")
                jsonresponse= env.generate_sample_request()
            finally:
                gc.collect()
                release_lock("sample_eval_req")
            return jsonresponse
        else:
            jsonresponse = {'rc': -1, 'msg': 'Server busy'}
            return jsonresponse

    @app.route('/evaluate', methods=['POST'])
    def validate_cfg():
        return evaluate(eval_type="evaluate")

    @app.route('/test', methods=['POST'])
    def test_cfg():
        return evaluate(eval_type="test")

    def evaluate(eval_type):
        #Code snippet for debugging purposes only
        start_time = time.time()
        #=================
        #Validate format of incoming request
        # =================
        try:
            assert eval_type == "evaluate" or eval_type == "test"
            content = request.get_json()
            input_req = content
            bool_bnadap = input_req.get('bool_bnadap', False)
            input_bw_cfg = input_req['bw_cfg']
            bw_cfg = dict()
            for node, bw in input_bw_cfg.items():
                if node in env.node_name_to_qenv_index: #Filter away unquantizable node prediction
                    assert bw in env.bw_space, "invalid bitwidth"
                    bw_cfg[env.node_name_to_qenv_index[node]]=bw
                    # print(node)
                    # print(env.node_name_to_qenv_index[node])
                    # print()
            assert len(bw_cfg) == len(env.node_name_to_qenv_index), "unexpected length of bw cfg"

        except:
            prRed("Exception in parsing http request")
            return {'rc': -3, 'msg': 'Exception in parsing http request'}
        
        # =================
        # For debugging purposes return a random response immediately
        # =================
        if 'debug' in content:
            if content['debug']:
                return dummy_response()
        
        if env.nncf_cfg.get('eval_cache', True) is True:
            cache_dir = os.path.join(env.nncf_cfg['log_dir'], 'qaas_eval_cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            input_cfg_hash = hashlib.md5(json.dumps(input_req).encode('utf-8')).hexdigest()
            input_cfg_hash_filename = os.path.join(cache_dir, eval_type+"_"+input_cfg_hash+".json")
            
            if os.path.exists(input_cfg_hash_filename):
                prRed("Recycling: {}".format(bw_cfg))
                prCyan("Recycling results in "+input_cfg_hash_filename)
                with open(input_cfg_hash_filename, 'r') as f:
                    jsonresponse = json.load(f)
                    jsonresponse['cached_eval'] = True
                return jsonresponse

        if acquire_lock("evaluate"):
            try:
                prRed("Evaluating: bnadap:{} === {}".format(str(bool_bnadap), bw_cfg))
                if eval_type == "evaluate":
                    retval = env.evaluate_valset(bw_cfg, bool_bnadap)
                elif eval_type == "test":
                    retval = env.evaluate_testset(bw_cfg, bool_bnadap)
                end_time = time.time()

                jsonresponse = dict()
                jsonresponse['rc'] = 0
                jsonresponse['msg'] = 'Quantize and inference completed'
                jsonresponse['meta_data'] = {
                    'hostname': hostname,
                    'eval_type': eval_type,
                    'task_metric': retval[-1]['accuracy'],
                    'original_model_size': env.original_model_size,
                    'original_bop': env.original_bop,
                    'bop_ratio': 1/(4*retval[-1]['bop_ratio']),
                    'size_ratio': retval[-1]['model_ratio'],
                    'input_req': input_req,
                    'executed_bw_cfg': bw_cfg,
                    'executed_wt_bnadap': bool_bnadap,
                    'quantizer_coupling': env.bool_perf_bw,
                    'ft_cfg_str': env.generate_ft_cfg(retval[-1])
                    }

                jsonresponse['processing_time'] = str(end_time - start_time)

                if env.nncf_cfg.get('eval_cache', True) is True:                  
                    evaluated_cfg_hash = hashlib.md5(json.dumps(bw_cfg).encode('utf-8')).hexdigest()
                    evaluated_cfg_hash_filename = os.path.join(cache_dir, eval_type+"_bnadap_"+str(bool_bnadap)+"_"+evaluated_cfg_hash+".json")

                    prCyan("Writing input cfg to " + input_cfg_hash_filename)
                    with open(input_cfg_hash_filename, 'w') as f:
                        json.dump(jsonresponse, f, indent=4)
                    prCyan("Writing evaluated cfg to " + evaluated_cfg_hash_filename)
                    with open(evaluated_cfg_hash_filename, 'w') as f:
                        json.dump(jsonresponse, f, indent=4)

                # restoration of dense state dict must be after stats collection
                # as statistics are only extracted upon collection
                # env.restore_dense_model() # not needed because qenv restores prior to evaluation
            finally:
                gc.collect()
                release_lock("evaluate")
                jsonresponse['cached_eval'] = False #TODO return cache path!
            return jsonresponse
        else:
            jsonresponse = {'rc': -1, 'msg': 'Server busy'}
            return jsonresponse

    return app