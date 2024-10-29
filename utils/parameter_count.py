import typing
from collections import defaultdict
import tabulate
from torch import nn


def parameter_count(model: nn.Module, trainable_only: bool = False) -> typing.DefaultDict[str, int]:
    """
    Count parameters of a model and its submodules.

    Args:
        model: a torch module

    Returns:
        dict (str-> int): the key is either a parameter name or a module name.
        The value is the number of elements in the parameter, or in all
        parameters of the module. The key "" corresponds to the total
        number of parameters of the model.
    """
    r = defaultdict(int)
    for name, prm in model.named_parameters():
        if trainable_only:
            if not prm.requires_grad:
                continue
        size = prm.numel()
        name = name.split(".")
        for k in range(0, len(name) + 1):
            prefix = ".".join(name[:k])
            r[prefix] += size
    return r


def parameter_count_table(
    model: nn.Module, max_depth: int = 3, trainable_only: bool = False
) -> str:
    """
    Format the parameter count of the model (and its submodules or parameters)
    in a nice table. It looks like this:

    ::

        | name                            | #elements or shape   |
        |:--------------------------------|:---------------------|
        | model                           | 37.9M                |
        |  backbone                       |  31.5M               |
        |   backbone.fpn_lateral3         |   0.1M               |
        |    backbone.fpn_lateral3.weight |    (256, 512, 1, 1)  |
        |    backbone.fpn_lateral3.bias   |    (256,)            |
        |   backbone.fpn_output3          |   0.6M               |
        |    backbone.fpn_output3.weight  |    (256, 256, 3, 3)  |
        |    backbone.fpn_output3.bias    |    (256,)            |
        |   backbone.fpn_lateral4         |   0.3M               |
        |    backbone.fpn_lateral4.weight |    (256, 1024, 1, 1) |
        |    backbone.fpn_lateral4.bias   |    (256,)            |
        |   backbone.fpn_output4          |   0.6M               |
        |    backbone.fpn_output4.weight  |    (256, 256, 3, 3)  |
        |    backbone.fpn_output4.bias    |    (256,)            |
        |   backbone.fpn_lateral5         |   0.5M               |
        |    backbone.fpn_lateral5.weight |    (256, 2048, 1, 1) |
        |    backbone.fpn_lateral5.bias   |    (256,)            |
        |   backbone.fpn_output5          |   0.6M               |
        |    backbone.fpn_output5.weight  |    (256, 256, 3, 3)  |
        |    backbone.fpn_output5.bias    |    (256,)            |
        |   backbone.top_block            |   5.3M               |
        |    backbone.top_block.p6        |    4.7M              |
        |    backbone.top_block.p7        |    0.6M              |
        |   backbone.bottom_up            |   23.5M              |
        |    backbone.bottom_up.stem      |    9.4K              |
        |    backbone.bottom_up.res2      |    0.2M              |
        |    backbone.bottom_up.res3      |    1.2M              |
        |    backbone.bottom_up.res4      |    7.1M              |
        |    backbone.bottom_up.res5      |    14.9M             |
        |    ......                       |    .....             |

    Args:
        model: a torch module
        max_depth (int): maximum depth to recursively print submodules or
            parameters

    Returns:
        str: the table to be printed
    """
    count: typing.DefaultDict[str, int] = parameter_count(model, trainable_only)
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    param_shape: typing.Dict[str, typing.Tuple] = {
        k: tuple(v.shape) for k, v in model.named_parameters()
    }

    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    table: typing.List[typing.Tuple] = []

    def format_size(x: int) -> str:
        if x > 1e8:
            return "{:.1f}G".format(x / 1e9)
        if x > 1e5:
            return "{:.1f}M".format(x / 1e6)
        if x > 1e2:
            return "{:.1f}K".format(x / 1e3)
        return str(x)

    def fill(lvl: int, prefix: str) -> None:
        if lvl >= max_depth:
            return
        for name, v in count.items():
            if name.count(".") == lvl and name.startswith(prefix):
                indent = " " * (lvl + 1)
                if name in param_shape:
                    table.append((indent + name, indent + str(param_shape[name])))
                else:
                    table.append((indent + name, indent + format_size(v)))
                    fill(lvl + 1, name + ".")

    table.append(("model", format_size(count.pop(""))))
    fill(0, "")

    old_ws = tabulate.PRESERVE_WHITESPACE
    tabulate.PRESERVE_WHITESPACE = True
    tab = tabulate.tabulate(table, headers=["name", "#elements or shape"], tablefmt="pipe")
    tabulate.PRESERVE_WHITESPACE = old_ws
    return tab


import torch
from typing import Optional, List, Dict, Any, Set
import copy
from detectron2.solver.build import reduce_param_groups


def get_default_optimizer_params_unet(
    model: torch.nn.Module,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    bias_lr_factor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
    unet_lr: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Get default param list for optimizer, with support for a few types of
    overrides. If no overrides needed, this is equivalent to `model.parameters()`.

    Args:
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        weight_decay_norm: override weight decay for params in normalization layers
        bias_lr_factor: multiplier of lr for bias parameters.
        weight_decay_bias: override weight decay for bias parameters
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.

    For common detection models, ``weight_decay_norm`` is the only option
    needed to be set. ``bias_lr_factor,weight_decay_bias`` are legacy settings
    from Detectron1 that are not found useful.

    Example:
    ::
        torch.optim.SGD(get_default_optimizer_params(model, weight_decay_norm=0),
                       lr=0.01, weight_decay=1e-4, momentum=0.9)
    """
    if overrides is None:
        overrides = {}
    defaults = {}
    if base_lr is not None:
        defaults["lr"] = base_lr
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay
    bias_overrides = {}
    if bias_lr_factor is not None and bias_lr_factor != 1.0:
        # NOTE: unlike Detectron v1, we now by default make bias hyperparameters
        # exactly the same as regular weights.
        if base_lr is None:
            raise ValueError("bias_lr_factor requires base_lr")
        bias_overrides["lr"] = base_lr * bias_lr_factor
    if weight_decay_bias is not None:
        bias_overrides["weight_decay"] = weight_decay_bias
    if len(bias_overrides):
        if "bias" in overrides:
            raise ValueError("Conflicting overrides for 'bias'")
        overrides["bias"] = bias_overrides

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm

            # Set different lr for pretrained denoising UNet
            if 'unet' in module_name and unet_lr is not None:
                hyperparams["lr"] = unet_lr

            hyperparams.update(overrides.get(module_param_name, {}))
            params.append({"params": [value], **hyperparams})
    return reduce_param_groups(params)


