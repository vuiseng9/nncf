#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging as log
import os
import sys
import time
import torch
import pandas as pd
import numpy as np
from collections import OrderedDict

import cv2
from openvino.inference_engine import IECore, StatusCode

from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', '{:20,.2f}'.format)
pd.set_option('display.max_colwidth', None)

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    args.add_argument('-m', '--model', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument('-i', '--input', required=True, type=str, nargs='+', help='Required. Path to an image file(s). If a data folder is provided, it assumes torchvision folder structures')
    args.add_argument('-l', '--extension', type=str, default=None,
                      help='Optional. Required by the CPU Plugin for executing the custom operation on a CPU. '
                      'Absolute path to a shared library with the kernels implementations.')
    args.add_argument('-c', '--config', type=str, default=None,
                      help='Optional. Required by GPU or VPU Plugins for the custom operation kernel. '
                      'Absolute path to operation description file (.xml).')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, MYRIAD, HDDL or HETERO: '
                      'is acceptable. The sample will look for a suitable plugin for device specified. '
                      'Default value is CPU.')
    args.add_argument('--labels', default=None, type=str, help='Optional. Path to a labels mapping file.')
    args.add_argument('-nt', '--number_top', default=10, type=int, help='Optional. Number of top results.')
    parser.add_argument('--log-freq', default=50, type=int,
                    metavar='N', help='batch logging frequency (default: 50), this is used only if input is imagenet dir')
    # fmt: on
    return parser.parse_args()


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = parse_args()

    # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
    log.info('Creating Inference Engine')
    ie = IECore()

    if args.extension and args.device == 'CPU':
        log.info(f'Loading the {args.device} extension: {args.extension}')
        ie.add_extension(args.extension, args.device)

    if args.config and args.device in ('GPU', 'MYRIAD', 'HDDL'):
        log.info(f'Loading the {args.device} configuration: {args.config}')
        ie.set_config({'CONFIG_FILE': args.config}, args.device)

    # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
    log.info(f'Reading the network: {args.model}')
    # (.xml and .bin files) or (.onnx file)
    net = ie.read_network(model=args.model)

    if len(net.input_info) != 1:
        log.error('Sample supports only single input topologies')
        return -1
    if len(net.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1

    # ---------------------------Step 3. Configure input & output----------------------------------------------------------
    log.info('Configuring input and output blobs')
    # Get names of input and output blobs
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # Get a number of input images
    num_of_input = len(args.input)
    datadir = None
    if num_of_input == 1:
        if os.path.isdir(args.input[0]):
            datadir = args.input[0]# assume torchvision datadir structure

    # Set input and output precision manually
    if datadir is not None:
        net.input_info[input_blob].precision = 'FP32'
    else:
        net.input_info[input_blob].precision = 'U8'
    net.outputs[out_blob].precision = 'FP32'

    # Get a number of classes recognized by a model
    num_of_classes = max(net.outputs[out_blob].shape)

    # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    log.info('Loading the model to the plugin')
    exec_net = ie.load_network(network=net, device_name=args.device, num_requests=num_of_input)

    # ---------------------------Step 5. Create infer request--------------------------------------------------------------
    # load_network() method of the IECore class with a specified number of requests (default 1) returns an ExecutableNetwork
    # instance which stores infer requests. So you already created Infer requests in the previous step.

    # ---------------------------Step 6. Prepare input---------------------------------------------------------------------
    if datadir is not None:
        
        from openvino.runtime import Core
        pyie = Core()
        pyir = pyie.read_model(args.model)

        pyir_sd = OrderedDict()
        for op in pyir.get_ordered_ops():
            if 'constant' in op.type_info.name.lower():
                # print("const. {} | {} ".format(op.get_name(), str(op.get_output_shape(0))))
                pyir_sd[op.get_name()] = op.get_vector()

        df = per_item_sparsity(pyir_sd)

        dataset = create_dataset(
        root=datadir, name='', split='validation',
        download=False, load_bytes=False, class_map='')

        crop_pct = 0.9
        bs = 1
        n_worker = 8
        data_config = {'input_size': (3, 224, 224), 'interpolation': 'bilinear', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'crop_pct': 0.875}
        
        loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=bs,
        use_prefetcher=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=8,
        crop_pct=data_config['crop_pct'],
        pin_memory=False,
        tf_preprocessing=False)

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            # measure accuracy and record loss
            infer_out = exec_net.infer({input_blob: input})
            output = torch.tensor(infer_out[out_blob], dtype=torch.float32)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                print(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        top1=top1, top5=top5))

        print(
            'Final::: '
            'E2E-Time: {batch_time.sum:.3f}s  '
            'Acc@1: {top1.avg:>7.3f}  '
            'Acc@5: {top5.avg:>7.3f}'.format(
                batch_time=batch_time, top1=top1, top5=top5)
            )



    else:
        input_data = []
        _, _, h, w = net.input_info[input_blob].input_data.shape

        for i in range(num_of_input):
            image = cv2.imread(args.input[i])

            if image.shape[:-1] != (h, w):
                log.warning(f'Image {args.input[i]} is resized from {image.shape[:-1]} to {(h, w)}')
                image = cv2.resize(image, (w, h))

            # Change data layout from HWC to CHW
            image = image.transpose((2, 0, 1))
            # Add N dimension to transform to NCHW
            image = np.expand_dims(image, axis=0)

            input_data.append(image)

        # ---------------------------Step 7. Do inference----------------------------------------------------------------------
        log.info('Starting inference in asynchronous mode')
        for i in range(num_of_input):
            exec_net.requests[i].async_infer({input_blob: input_data[i]})

        # ---------------------------Step 8. Process output--------------------------------------------------------------------
        # Generate a label list
        if args.labels:
            with open(args.labels, 'r') as f:
                labels = [line.split(',')[0].strip() for line in f]

        # Create a list to control a order of output
        output_queue = list(range(num_of_input))

        while True:
            for i in output_queue:
                # Immediately returns a inference status without blocking or interrupting
                infer_status = exec_net.requests[i].wait(0)

                if infer_status == StatusCode.RESULT_NOT_READY:
                    continue

                log.info(f'Infer request {i} returned {infer_status}')

                if infer_status != StatusCode.OK:
                    return -2

                # Read infer request results from buffer
                res = exec_net.requests[i].output_blobs[out_blob].buffer
                # Change a shape of a numpy.ndarray with results to get another one with one dimension
                probs = res.reshape(num_of_classes)
                # Get an array of args.number_top class IDs in descending order of probability
                top_n_idexes = np.argsort(probs)[-args.number_top :][::-1]

                header = 'classid probability'
                header = header + ' label' if args.labels else header

                log.info(f'Image path: {args.input[i]}')
                log.info(f'Top {args.number_top} results: ')
                log.info(header)
                log.info('-' * len(header))

                for class_id in top_n_idexes:
                    probability_indent = ' ' * (len('classid') - len(str(class_id)) + 1)
                    label_indent = ' ' * (len('probability') - 8) if args.labels else ''
                    label = labels[class_id] if args.labels else ''
                    log.info(f'{class_id}{probability_indent}{probs[class_id]:.7f}{label_indent}{label}')
                log.info('')

                output_queue.remove(i)

            if len(output_queue) == 0:
                break

    # ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0




def per_item_sparsity(state_dict):
    def calc_sparsity(tensor):
        if isinstance(tensor, torch.Tensor):
            rate = 1-(tensor.count_nonzero()/tensor.numel())
            return rate.item()
        else:
            rate = 1-(np.count_nonzero(tensor)/tensor.size)
            return rate

    dlist=[]
    for key, param in state_dict.items():
        l = OrderedDict()
        l['layer_id'] = key
        l['shape'] = list(param.shape)
        l['nparam'] = np.prod(l['shape'])
        if isinstance(param, torch.Tensor):
            l['nnz'] = param.count_nonzero().item()
        else:
            l['nnz'] = np.count_nonzero(param)
        l['sparsity'] = calc_sparsity(param)
        dlist.append(l)
    df = pd.DataFrame.from_dict(dlist)
    return df




if __name__ == '__main__':
    sys.exit(main())
