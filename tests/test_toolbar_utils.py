import sympy

import pytest

from typing import Optional

from src.toolbar_utils import build_function
from src.errors import Error


class BaseCase:

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return f'test_{self.name}'


class BuildFunctionCase(BaseCase):

    def __init__(self, name: str, error: Error, input_string: str,
                 func_sp: Optional[sympy.core.function.Function]):
        super().__init__(name)

        self.error = error
        self.input_string = input_string
        self.func_sp = func_sp
    

x, y = sympy.symbols('x y')

BUILD_FUNCTION_CASES = [
    BuildFunctionCase('sum',
                      Error.OK,
                      'x+y',
                      x + y),
    BuildFunctionCase('sqrt',
                      Error.OK,
                      'x**2/3-24*sqrt(y)/x',
                      x ** 2 / 3 - 24 * y ** 0.5 / x),
    BuildFunctionCase('trigonometry',
                      Error.OK,
                      'x*y-6.5*(Abs(x)-sin(cos(y)))',
                      x * y - 6.5 * (sympy.Abs(x) - sympy.sin(sympy.cos(y)))),
    BuildFunctionCase('unknown symbol',
                      Error.GRAMMATICAL,
                      'a+b',
                      None),
    BuildFunctionCase('unknown function',
                      Error.GRAMMATICAL,
                      'x+y-f(x)',
                      None),
    BuildFunctionCase('bad syntax',
                      Error.SYNTAX,
                      'x+y/(2',
                      None),
    BuildFunctionCase('comma',
                      Error.SYNTAX,
                      'x+y, x*y',
                      None)
]


@pytest.mark.parametrize('case', BUILD_FUNCTION_CASES, ids=str)
def test_build_function(case: BuildFunctionCase) -> None:
    err, func_sp, _ = build_function(case.input_string)
    assert err == case.error
    if err == Error.OK:
        assert sympy.nsimplify(func_sp - case.func_sp) == 0 # type: ignore
