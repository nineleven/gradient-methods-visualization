import numpy as np

from .utils import CountCalls


def __make_result_dict(*, x, n_iter, n_grad_calls, success):
    return dict(x=x, n_iter=n_iter, 
                n_grad_calls=n_grad_calls,
                success=success)

def recalc_hess_inv(H, s, y):
    ro = 1 / y.T.dot(s)
    I = np.eye(H.shape[0])
    return (I - ro*s.dot(y.T)).dot(H).dot(I - ro*y.dot(s.T)) + ro*s.dot(s.T)

def __optimize(grad_f, x_0, epsilon, alpha):
    history = []

    x_0 = np.array(x_0)
    
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
            history = np.array(history)
            return history

        grad_prev = grad_value
        grad_value = np.array(grad_f(x))
        
        s = (x - x_prev).reshape(-1, 1)
        y = (grad_value - grad_prev).reshape(-1, 1)
        H_inv = recalc_hess_inv(H_inv, s, y)
        
        n_iter += 1

def bfgs(grad_f, x_0, epsilon, alpha=1, return_history=False):
    
    @CountCalls
    def grad_f_wrapper(x):
        return grad_f(x)
    
    history = __optimize(grad_f_wrapper, x_0, epsilon, alpha)
        
    result_dict = __make_result_dict(
        x=history[-1], 
        n_iter=len(history)-1, 
        n_grad_calls=grad_f_wrapper.n_calls, 
        success=True
    )

    if return_history:
        return result_dict, history

    return result_dict
        
