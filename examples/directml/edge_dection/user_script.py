# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from detectors.canny import CannyFilter
from detectors.hed import HedFilter

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
# CANNY
# -----------------------------------------------------------------------------

def canny_inputs(batchsize, torch_dtype):
    return {
        "input": torch.rand((batchsize, 3, 512, 512), dtype=torch_dtype)
    }


def canny_load(model_path):
    model = CannyFilter()
    return model


def canny_conversion_inputs(model):
    return tuple(canny_inputs(1, torch.float32).values())


def canny_data_loader(data_dir, batchsize):
    return RandomDataLoader(canny_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# HED
# -----------------------------------------------------------------------------

def hed_inputs(batchsize, torch_dtype):
    return {
        "input": torch.rand((batchsize, 3, 512, 512), dtype=torch_dtype)
    }


def hed_load(model_path):
    model = HedFilter()
    return model


def hed_conversion_inputs(model):
    return tuple(hed_inputs(1, torch.float32).values())


def hed_data_loader(data_dir, batchsize):
    return RandomDataLoader(hed_inputs, batchsize, torch.float16)