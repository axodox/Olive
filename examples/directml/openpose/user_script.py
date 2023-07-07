# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from openpose.model import ResNetBackbone, CmapPafHeadAttention

# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batchsize, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batchsize = batchsize
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batchsize, self.torch_dtype), label
    
# -----------------------------------------------------------------------------
# OPEN POSE
# -----------------------------------------------------------------------------
    
class OpenPoseResnetModel(torch.nn.Module):
    def __init__(self, path):
        super(OpenPoseResnetModel, self).__init__()
        self.model = torch.nn.Sequential(
            ResNetBackbone(),
            CmapPafHeadAttention(512, 18, 42)
        )
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def forward(self, input):
        # In a ResNet-based neural network used for pose estimation, 
        # the cmap and paf outputs store the confidence maps and part affinity fields respectively. 
        # The confidence maps are used to identify the location of body parts in an image 
        # while the part affinity fields are used to identify the connections between these body parts.

        cmap, paf = self.model.forward(input)
        return cmap, paf
    
def openpose_inputs(batchsize, torch_dtype):
    return {
        "input": torch.rand((batchsize, 3, 224, 224), dtype=torch_dtype)
    }

def openpose_load(model_path):
    model = OpenPoseResnetModel(model_path)
    return model

def openpose_conversion_inputs(model):
    return tuple(openpose_inputs(1, torch.float32).values())


def openpose_data_loader(data_dir, batchsize):
    return RandomDataLoader(openpose_inputs, batchsize, torch.float16)