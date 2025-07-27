# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import torch

# Print PyTorch version
print(torch.__version__)

# Print CUDA version
print(torch.version.cuda)

# Print cuDNN version
print(torch.backends.cudnn.version())