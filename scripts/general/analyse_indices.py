# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os

#compressor='topk'
#compressor='gtopkr'
#bs=16

def plot_update_index(compressor, bs, version='v2'):
    LOGHOME='./logs/allreduce-comp-%s-baseline-gwarmup-dc1-gtopkjournal-%s/resnet20-n8-bs%d-lr0.1000-ns1-sg2.50-ds0.001' % (compressor, version, bs)
    for e in range(120):
        if e > 0 and e % 110 == 0:
            fn = 'index-rank0-epoch%d.npy' % e
            fullfn = os.path.join(LOGHOME, fn)
            index_counter = np.load(fullfn).astype(np.int)
            non_zeros = np.count_nonzero(index_counter)
            size = index_counter.size
            num_zeros = size-non_zeros
            print('[%s-%s,bs=%d], Epoch: %d, min: %d, max: %d, # of zeros: %d, # of parameters: %d, zero ratio: %f' % \
                (compressor, version, bs, e, np.min(index_counter), np.max(index_counter), num_zeros, size, float(num_zeros)/size))
            plt.scatter(range(len(index_counter)), index_counter, label='%s-%s, bs=%d, epoch %d'%(compressor, version, bs, e))
#plt.title('%s, bs=%d'%(compressor, bs))
plot_update_index('gtopkr', 16)
plot_update_index('gtopkr', 32)
plot_update_index('gtopkr', 32, 'r4')
plot_update_index('topk', 32)
plt.legend()
plt.show()
