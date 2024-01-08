from typing import Optional, Mapping, Tuple, Sequence, NoReturn, Union

import torch
import torch.nn as nn
from torch import Tensor

import diffcloth_py as diffcloth

from .functional import SimFunction, SimFunctionForce


class pySim(nn.Module):

    def __init__(self,
    cppSim: diffcloth.Simulation,
    optimizeHelper: diffcloth.OptimizeHelper,
    useFixedPoint: bool
    ) -> NoReturn:
        super().__init__()
        self.cppSim = cppSim
        self.optimizeHelper = optimizeHelper

        self.cppSim.useCustomRLFixedPoint = useFixedPoint

    def forward(
            self,
            x: Tensor,
            v: Tensor,
            a: Tensor
    ) -> Tuple[Tensor, Tensor]:

        return SimFunction.apply(
            x, v, a, self.cppSim, self.optimizeHelper)
    

class pySim_force(nn.Module):

    def __init__(self,
    cppSim: diffcloth.Simulation,
    optimizeHelper: diffcloth.OptimizeHelper,
    useFixedPoint: bool
    ) -> NoReturn:
        super().__init__()
        self.cppSim = cppSim
        self.optimizeHelper = optimizeHelper

        self.cppSim.useCustomRLFixedPoint = useFixedPoint

    def forward(
            self,
            x: Tensor,
            v: Tensor,
            a: Tensor, 
            f: Tensor
    ) -> Tuple[Tensor, Tensor]:

        return SimFunctionForce.apply(
            x, v, a, f, self.cppSim, self.optimizeHelper)