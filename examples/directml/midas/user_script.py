# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from typing import Union, Optional, Tuple
from midas.dpt_depth import DPTDepthModel

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
# MIDAS
# -----------------------------------------------------------------------------

def midas_inputs(batchsize, torch_dtype):
    return {
        "x": torch.rand((batchsize, 3, 512, 512), dtype=torch_dtype)
    }


def midas_load(model_path):
    model = DPTDepthModel(
        path=model_path,
        backbone="beitl16_512",
        non_negative=True,
    )
    return model


def midas_conversion_inputs(model):
    return tuple(midas_inputs(1, torch.float32).values())


def midas_data_loader(data_dir, batchsize):
    return RandomDataLoader(midas_inputs, batchsize, torch.float16)