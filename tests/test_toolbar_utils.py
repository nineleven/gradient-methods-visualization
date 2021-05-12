import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import sympy

import pytest

import math

from src.toolbar_utils import build_function, build_gradient


class BuildFunctionCase:

    def __init__(self, name: str, input_string: str,
                 func_sp: sympy.core.function.Function):
        self.name = name
        self.input_string = input_string
        self.func_sp = func_sp

    def __str__(self) -> str:
        return f'test_{self.name}'
    

x, y = sympy.symbols('x y')
BUILD_FUNCTION_CASES = [
    BuildFunctionCase('sum', 'x+y',
                      x+y),
    BuildFunctionCase('sqrt', 'x**2/3-24*sqrt(y)/x',
                      x**2/3-24*y**0.5/x),
    BuildFunctionCase('trigonometry', 'x*y-6.5*(Abs(x)-sin(cos(y)))',
                      x*y-6.5*(sympy.Abs(x)-sympy.sin(sympy.cos(y))))
]


@pytest.mark.parametrize('case', BUILD_FUNCTION_CASES, ids=str)
def test_build_function(case: BuildFunctionCase) -> None:
    func_sp, _ = build_function(case.input_string)
    
    assert sympy.nsimplify(func_sp-case.func_sp) == 0
