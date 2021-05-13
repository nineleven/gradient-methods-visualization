import numpy as np

from typing import Dict, Callable, Sequence, Any, List, Tuple, Union

from utils import CountCalls


def __make_result_dict(*, x: np.ndarray,
                       n_iter: int,
                       n_grad_calls: int,
                       success: bool) -> Dict['str', Any]:
    '''
    Utility function, used to ensure that all
    of the expected results are included into the
    result dictionary

    Parameters
    ----------
    x : np.ndarray
        Solution
    n_iter : int
        Number of iterations
    n_grad_calls : int
        Number of gradient calls
    success : bool
        Whether the method converged successfully
        
    Returns
    -------
    Dict['str', Any]
        Result dictionary
    '''
    return dict(x=x, n_iter=n_iter,
                n_grad_calls=n_grad_calls,
                success=success)


def recalc_hess_inv(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    Computes new approximation of the inverse of hessian.
    The notation the same as here https://ru.wikipedia.org/wiki/Алгоритм_Бройдена_—_Флетчера_—_Гольдфарба_—_Шанно

    Parameters
    ----------
    H : numpy.ndarray
        Current approximation
    s: numpy.ndarray
        x_k+1 - x_k
    y: numpy.ndarray
        grad(x_k+1) - grad(x_k)
        
    '''
    
    rho = 1 / y.T.dot(s)
    eye = np.eye(H.shape[0])
    return (eye - rho * s.dot(y.T)).dot(H).dot(eye - rho * y.dot(s.T)) + rho * s.dot(s.T)


def __optimize(grad_f: Callable[[np.ndarray], np.ndarray],
               x_0: np.ndarray, epsilon: float, alpha: float) -> List[np.ndarray]:
    '''
    Computes new approximation of the inverse of hessian.
    The notation the same as here https://ru.wikipedia.org/wiki/Алгоритм_Бройдена_—_Флетчера_—_Гольдфарба_—_Шанно

    Parameters
    ----------
    grad_f: Callable[[numpy.ndarray], numpy.ndarray]
        Objective function gradient
    x_0 : np.ndarray
        Initial approximation
    epsilon : float
        Desired precision
    alpha : float
        Step of the alogirithm

    Returns
    -------
    List[numpy.ndarray]
        History
        
    '''
    
    history = []

    H_inv = np.eye(len(x_0))

    grad_value = np.array(grad_f(x_0))
    
    x = x_0
    
    n_iter = 0
    
    while True:
        history.append(x)

        p = H_inv.dot(grad_value)
        
        x_prev = x
        x = x - alpha * p
        
        if np.linalg.norm(grad_value) < epsilon:
            return history

        grad_prev = grad_value
        grad_value = np.array(grad_f(x))
        
        s = (x - x_prev).reshape(-1, 1)
        y = (grad_value - grad_prev).reshape(-1, 1)
        H_inv = recalc_hess_inv(H_inv, s, y)
        
        n_iter += 1


def bfgs(grad_f: Callable[[np.ndarray], np.ndarray],
         x_0: np.ndarray, epsilon: float, alpha: float=1,
         return_history: bool=False) -> Union[Dict['str', Any], Tuple[Dict['str', Any], np.ndarray]]:
    '''
    Computes new approximation of the inverse of hessian.
    The notation the same as here https://ru.wikipedia.org/wiki/Алгоритм_Бройдена_—_Флетчера_—_Гольдфарба_—_Шанно

    Parameters
    ----------
    grad_f: Callable[[numpy.ndarray], numpy.ndarray]
        Objective function gradient
    x_0 : np.ndarray
        Initial approximation
    epsilon : float
        Desired precision
    alpha : float
        Step of the algirithm
    return_history : bool
        Whether to return the history

    Returns
    -------
    Union[Dict['str', Any], Tuple[Dict['str', Any], numpyp.ndarray]]
        Result dictionary or a tuple of the result dictionary and the history
        
    '''
    
    @CountCalls
    def grad_f_wrapper(x: Any) -> Any:
        return grad_f(x)
    
    history_list = __optimize(grad_f_wrapper, np.array(x_0), epsilon, alpha)
    history = np.array(history_list)
        
    result_dict = __make_result_dict(
        x=history[-1],
        n_iter=len(history) - 1,
        n_grad_calls=grad_f_wrapper.n_calls,
        success=True
    )

    if return_history:
        return result_dict, history

    return result_dict
