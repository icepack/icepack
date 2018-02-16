
import inspect

def add_kwarg_wrapper(func):
    signature = inspect.signature(func)
    if any(str(signature.parameters[name].kind) == 'VAR_KEYWORD'
           for name in signature.parameters):
        return func

    params = signature.parameters
    def wrapper(*args, **kwargs):
        kwargs_ = dict((name, kwargs[name]) for name in kwargs if name in params)
        return func(*args, **kwargs_)
    return wrapper
