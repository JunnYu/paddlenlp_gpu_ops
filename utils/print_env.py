#!/usr/bin/env python3

# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# this script dumps information about the environment

import platform
import sys

import paddle.device
import paddle.version


print("Python version:", sys.version)

print("OS platform:", platform.platform())
print("OS architecture:", platform.machine())

try:
    import paddle

    print("Paddle version:", paddle.__version__)
    print("Paddle commit:", paddle.version.commit)
    print("Cuda available:", paddle.device.is_compiled_with_cuda())
    print("Cuda version:", paddle.version.cuda())
    print("CuDNN version:", paddle.version.cudnn())
    print("Number of GPUs available:", paddle.device.get_available_device())
except ImportError:
    print("Paddle version:", None)

try:
    import paddlenlp

    print("paddlenlp version:", paddlenlp.version.show())
except ImportError:
    print("paddlenlp version:", None)
