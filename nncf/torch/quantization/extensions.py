"""
 Copyright (c) 2022 Intel Corporation
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

import os.path
import torch
from torch.utils.cpp_extension import load

from nncf.torch.extensions import CudaNotAvailableStub, ExtensionsType, ExtensionLoader, EXTENSIONS
from nncf.definitions import NNCF_PACKAGE_ROOT_DIR

BASE_EXT_DIR = os.path.join(NNCF_PACKAGE_ROOT_DIR, "torch/extensions/src/quantization")

EXT_INCLUDE_DIRS = [
    os.path.join(NNCF_PACKAGE_ROOT_DIR, "torch/extensions/include"),
]

CPU_EXT_SRC_LIST = [
    os.path.join(BASE_EXT_DIR, "cpu/functions_cpu.cpp"),
    os.path.join(NNCF_PACKAGE_ROOT_DIR, "torch/extensions/src/common/cpu/tensor_funcs.cpp")
]

CUDA_EXT_SRC_LIST = [
    os.path.join(BASE_EXT_DIR, "cuda/functions_cuda.cpp"),
    os.path.join(BASE_EXT_DIR, "cuda/functions_cuda_impl.cu")
]

#TODO: revise path
HPU_EXT_SRC_LIST = [
    os.path.join(BASE_EXT_DIR, "hpu/*"),
    os.path.join(BASE_EXT_DIR, "hpu/*")
]

@EXTENSIONS.register()
class QuantizedFunctionsCPULoader(ExtensionLoader):
    @classmethod
    def extension_type(cls):
        return ExtensionsType.CPU

    @classmethod
    def name(cls) -> str:
        return 'quantized_functions_cpu'

    @classmethod
    def load(cls):
        return load(cls.name(),
                    CPU_EXT_SRC_LIST,
                    extra_include_paths=EXT_INCLUDE_DIRS,
                    build_directory=cls.get_build_dir(),
                    verbose=False)


@EXTENSIONS.register()
class QuantizedFunctionsCUDALoader(ExtensionLoader):
    @classmethod
    def extension_type(cls):
        return ExtensionsType.CUDA

    @classmethod
    def load(cls):
        return load(cls.name(),
                    CUDA_EXT_SRC_LIST,
                    extra_include_paths=EXT_INCLUDE_DIRS,
                    build_directory=cls.get_build_dir(),
                    verbose=False)

    @classmethod
    def name(cls) -> str:
        return 'quantized_functions_cuda'

IS_HPU=False

@EXTENSIONS.register()
class QuantizedFunctionsHPULoader(ExtensionLoader):
    @classmethod
    def extension_type(cls):
        return ExtensionsType.HPU

    @classmethod
    def load(cls):
        # TODO: for CPU can CUDA versions, the load function is the torch c++ extension build process
        # For Habana, we need a different process
        # * TPC kernel
        # * TPC Glue code
        # * Torch Custom Op registration 
        #   https://github.com/HabanaAI/Model-References/blob/master/PyTorch/examples/custom_op/custom_relu/hpu_custom_relu.cpp
        # * Torch function and module wrapping 
        #   reuse nncf/torch/quantization/quantize_functions.py
        #   otherwise: https://github.com/HabanaAI/Model-References/blob/master/PyTorch/examples/custom_op/custom_relu/custom_relu.py
        return load(cls.name(),
                    CUDA_EXT_SRC_LIST,
                    extra_include_paths=EXT_INCLUDE_DIRS,
                    build_directory=cls.get_build_dir(),
                    verbose=False)

    @classmethod
    def name(cls) -> str:
        return 'quantized_functions_cpu'


QuantizedFunctionsCPU = QuantizedFunctionsCPULoader.load()

if torch.cuda.is_available():
    QuantizedFunctionsCUDA = QuantizedFunctionsCUDALoader.load()
else:
    QuantizedFunctionsCUDA = CudaNotAvailableStub

if IS_HPU:
    QuantizedFunctionsHPU = QuantizedFunctionsHPULoader.load()
else:
    # Just to avoid error for existing build
    QuantizedFunctionsHPU = None