import os
import random

import torch
import numpy as np

from poyo.utils import logging

log = logging(header='RANDOM SEED', header_color='red')


def seed_everything(seed: int) -> None:
    """Sets random seed for reproducibility.
    Args:
        seed (int): Random seed.
    """
    if seed is not None:
        log.info("Global seed set to {}.".format(seed))

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
