import logging
import sys

from . import utils
from .networks import *


def get_networks(state, N=None, arch=None):
    N = N or state.local_n_nets
    arch = arch or state.arch
    mod = sys.modules[__name__]
    cls = getattr(mod, arch)
    # if state.input_size not in cls.supported_dims:
    #     raise RuntimeError("{} doesn't support input size {}".format(cls.__name__, state.input_size))
    
    logging.info('Build {} {} network(s) with [{}({})] init'.format(N, arch, state.init, state.init_param))
    nets = []

    state.embed_fn, state.input_ch = get_embedder()
    state.embeddirs_fn, state.input_ch_views = get_embedder(4)

    for n in range(N):
        # TO-DO Initializer embedder for NeRF
        net = NeRF(state)
        net.reset(state)  # verbose only last one
        nets.append(net)
    return nets