import torch
from torch.utils._pytree import tree_map
import functools
from threading import Lock
from collections import defaultdict
import traceback
import numpy as np
from functools import partial
from torch.return_types import max as MaxReturnType

"""
    Notes:
        - We don't care about aten.tanh_backward.default (i.e. use standard semi-ring)
        - aten.native_layer_norm_backward.default ?
        - aten.gelu_backward.default ?
        - aten._softmax_backward_data.default ? (I think not touched by BP no?)
        - aten.div.Tensor not affected by MaxProd semi-ring
        - aten.embedding_dense_backward.default ?
"""


def format_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        return np.array2string(
            tensor.detach().cpu().numpy(),
            separator=", ",
            formatter={"float_kind": lambda x: f"{x:.4f}"},  # Adjust precision
        ).replace("\n", "")

    if isinstance(tensor, "HelpTensor"):
        return np.array2string(
            tensor.elem.detach().cpu().numpy(),
            separator=", ",
            formatter={"float_kind": lambda x: f"{x:.4f}"},  # Adjust precision
        ).replace("\n", "")
    return str(tensor)


def print_yellow(text):
    yellow_code = "\033[33m"
    reset_code = "\033[0m"
    print(f"{yellow_code}{text}{reset_code}")


def print_green(text):
    light_green_code = "\033[92m"
    reset_code = "\033[0m"
    print(f"{light_green_code}{text}{reset_code}")


def print_bgreen(text):
    bold_light_green_code = "\033[1m\033[92m"
    reset_code = "\033[0m"
    print(f"{bold_light_green_code}{text}{reset_code}")


def print_blue(text):
    blue_code = "\033[34m"
    reset_code = "\033[0m"
    print(f"{blue_code}{text}{reset_code}")


def print_light_blue(text):
    light_blue_code = "\033[36m"
    reset_code = "\033[0m"
    print(f"{light_blue_code}{text}{reset_code}")


class HelpTensor(torch.Tensor):
    def __new__(cls, t, verbose_level=0):
        # We would need to define autograd on this ring, inception!
        assert t.requires_grad is False

        res = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,  # New class: HelpTensor
            size=t.size(),
            strides=t.stride(),
            storage_offset=0,
            dtype=t.dtype,
            layout=t.layout,
            device=t.device,
            requires_grad=False,
        )

        cls.verbose_level = verbose_level
        cls.log_dict = {}

        return res

    def __init__(self, t):
        self.elem = t

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(t):
            if isinstance(t, cls):
                return t.elem
            else:
                return t

        def wrap(t):
            if isinstance(t, torch.Tensor) and not isinstance(t, cls):
                return cls(t)
            else:
                return t

        if func == torch.ops.aten._log_softmax_backward_data.default:
            breakpoint()

        if func == torch.ops.aten.mm.default:
            breakpoint()

        def run_with_usual_semantic():
            return tree_map(
                wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            )

        # Log the function being called
        print(f"Function dispat: {func.__name__}")

        return run_with_usual_semantic()

    def __repr__(self):
        return f"HelpTensor({self.elem})"
