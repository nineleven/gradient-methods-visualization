import sympy

from typing import Tuple, Callable, Sequence, Optional, List

from pathlib import Path

from utils import get_logger
from errors import Error


logger = get_logger(Path(__file__).name)


def build_function(input_str: str) -> Tuple[Error,
                                            Optional[sympy.core.function.Function],
                                            Optional[Callable]]:
    '''
    Parses objective function from a given string

    Parameters
    ----------
    input_str : str
        Input string

    Returns
    -------
    Tuple[Error, Optional[sympy.core.function.Function], Optional[Callable]]
        Tuple of the error code, parsed sympy function and a callable version
        of this function
    '''
    
    logger.debug('Building function')

    if ',' in input_str:
        logger.warning('The expression contains comma')
        return Error.SYNTAX, None, None
    
    try:
        func_sp = sympy.sympify(input_str)

        unknown_functions = func_sp.atoms(sympy.core.function.AppliedUndef)

        if unknown_functions:
            logger.warning('Unknown functions used: ' + str(unknown_functions))
            return Error.GRAMMATICAL, None, None

        unknown_symbols = func_sp.atoms(sympy.core.Symbol).difference(sympy.symbols('x y'))
        
        if unknown_symbols:
            logger.warning('Unknown symbols used: ' + str(unknown_symbols))
            return Error.GRAMMATICAL, None, None
        
        def func(x: Sequence[float]) -> float:
            nonlocal func_sp
            func_eval = func_sp.subs('x', x[0]).subs('y', x[1]).evalf()
            return float(func_eval)

        return Error.OK, func_sp, func
        
    except (SyntaxError, ValueError):
        logger.warning('Syntax error in function definition')
        return Error.SYNTAX, None, None


def build_gradient(func: sympy.core.function.Function) -> Tuple[Error, Optional[Callable]]:
    '''
    Builds the gradient of the objective function

    Parameters
    ----------
    func : sympy.core.function.Function
        Function to differentiate

    Returns
    -------
    Tuple[Error, Optional[Callable]]
        Tuple of the error code and the gradient function
    '''
    
    logger.debug('Building gradient')
    
    try:
        grad_sp = sympy.Matrix([sympy.diff(func, 'x'),
                                sympy.diff(func, 'y')])

        def grad(x: Sequence[float]) -> List[float]:
            nonlocal grad_sp
            grad_eval = grad_sp.subs('x', x[0]).subs('y', x[1]).evalf()
            return list(map(float, grad_eval))

        return Error.OK, grad
        
    except ValueError:
        logger.warning('Unable to differentiate the function')
        return Error.UNABLE_TO_DIFFERENTIALE, None
