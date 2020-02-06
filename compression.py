# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
import time
import random


class NoneCompressor():
    @staticmethod
    def compress(tensor, name=None):
        return tensor, tensor.dtype

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 


class TopKCompressor():
    """
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
    residuals = {}
    c = 0
    sparsities = []
    t = 0.
    zero_conditions = {}
    delay_counters = {}
    zc = None
    values = {} 
    indexes = {} 
    name = 'topk'
    @staticmethod
    def compress(tensor, name=None, sigma_scale=2.5, ratio=0.05):
        start = time.time()
        with torch.no_grad():
            if name not in TopKCompressor.residuals:
                TopKCompressor.residuals[name] = torch.zeros_like(tensor.data)
                TopKCompressor.delay_counters[name] = torch.ones_like(tensor.data)
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            #if name in TopKCompressor.indexes:
            #    tensor.data[TopKCompressor.indexes[name]] *= 1.1 # some improvement!
            #tensor.data.mul_(TopKCompressor.delay_counters[name].data)
            tensor.data.add_(TopKCompressor.residuals[name].data)
            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]
            if name not in TopKCompressor.zero_conditions:
                TopKCompressor.zero_conditions[name] = torch.ones(numel, dtype=torch.float32, device=tensor.device) 
            zero_condition = TopKCompressor.zero_conditions[name]
            zero_condition.fill_(1.0)
            zero_condition[indexes] = 0.0
            TopKCompressor.zc = zero_condition

            TopKCompressor.residuals[name].data = tensor.data * zero_condition
            tensor.data.sub_(TopKCompressor.residuals[name].data)

            TopKCompressor.values[name] = values
            TopKCompressor.indexes[name] = indexes
            return tensor, indexes 

    @staticmethod
    def get_residuals(name, like_tensor):
        if name not in TopKCompressor.residuals:
            TopKCompressor.residuals[name] = torch.zeros_like(like_tensor.data)
        return TopKCompressor.residuals[name]

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            zero_condition = TopKCompressor.zero_conditions[name]
            #TODO: Should be revised. It could have bugs for layer-wise communication
            TopKCompressor.zc = zero_condition
            if included_indexes is None:
                return
            residuals = TopKCompressor.residuals[name]
            if type(included_indexes) is np.ndarray:
                indexes_t = torch.from_numpy(included_indexes).cuda(residuals.device).long()
            else:
                indexes_t = included_indexes
            values = TopKCompressor.values[name]
            values.data[indexes_t] = 0.0
            residuals.data[TopKCompressor.indexes[name]] += values.data
            TopKCompressor.indexes[name] = TopKCompressor.indexes[name][indexes_t]
            delay_counter = TopKCompressor.delay_counters[name]
            delay_counter.add_(1)
            delay_counter[TopKCompressor.indexes[name]] = 1
            #delay_counter[delay_counter>5] = 5 # something good
            delay_counter[delay_counter>6] = 6
            zero_condition.fill_(1.0)
            zero_condition[TopKCompressor.indexes[name]] = 0.0

    @staticmethod
    def decompress(tensor, ctc, name=None):
        z = tensor 
        return z 

    @staticmethod
    def clear_residuals():
        for k in TopKCompressor.residuals:
            v = TopKCompressor.residuals[k]
            v.fill_(0.0)


class TopKCompressor2(TopKCompressor):
    name = 'topk2'


class gTopKCompressor(TopKCompressor):
    name = 'gtopk'


class gTopKRecursiveCompressor(TopKCompressor):
    name = 'gtopkr'


class TopKDenseCompressor(TopKCompressor):
    """
    Sparse + Dense for Distributed Gradient Descent
    """
    name = 'topkdense'


compressors = {
        'topk': TopKCompressor,
        'topk2': TopKCompressor2,
        'gtopk': gTopKCompressor,
        'gtopkr': gTopKRecursiveCompressor,
        'topkdense': TopKDenseCompressor,
        'none': NoneCompressor
        }
