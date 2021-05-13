from enum import Enum


class Error(Enum):
    OK = 1,
    SYNTAX = 2,
    GRAMMATICAL = 3,
    UNABLE_TO_DIFFERENTIALE = 4


def get_error_message(err: Error) -> str:
    '''
    Returns error message for a given error code

    Parameters
    ----------
    err : Error
        Error code

    Returns
    -------
    str
        Error message
    '''
    
    if err == Error.OK:
        assert False, 'There is no error!'
        return 'OK'
    if err == Error.SYNTAX:
        return 'Syntax error'
    if err == Error.GRAMMATICAL:
        return 'Grammatic error'
    if err == Error.UNABLE_TO_DIFFERENTIALE:
        return 'Unable to differentiate the function'
    
    return 'Unknown error'
