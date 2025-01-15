import random
import numpy as np
import torch

from utils.constants import SEED


def ensure_deterministic():
    """
    Ensures reproducible results by fixing seeds and forcing deterministic algorithms in PyTorch.
    """
    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
