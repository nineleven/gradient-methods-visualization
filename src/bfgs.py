import numpy as np

from typing import Dict, Callable, Sequence, Any, List, Tuple, Union

from utils import CountCalls


def __make_result_dict(*, x: np.ndarray,
                       n_iter: int,
                       n_grad_calls: int,
                       success: bool) -> Dict['str', Any]:
    return dict(x=x, n_iter=n_iter,
                n_grad_calls=n_grad_calls,
                success=success)


def recalc_hess_inv(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    rho = 1 / y.T.dot(s)
    eye = np.eye(H.shape[0])
    return (eye - rho * s.dot(y.T)).dot(H).dot(eye - rho * y.dot(s.T)) + rho * s.dot(s.T)


def __optimize(grad_f: Callable[[np.ndarray], np.ndarray],
               x_0: np.ndarray, epsilon: float, alpha: float) -> List[np.ndarray]:
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
