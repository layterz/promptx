from typing import Union, List
import inspect


def get_function_info(func):
    signature = inspect.signature(func)
    parameters = signature.parameters

    func_info = {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }

    for param_name, param in parameters.items():
        param_info = {
            "type": str(param.annotation),
            "description": param.default.__doc__ if param.default else None
        }
        func_info["parameters"][param_name] = param_info

    return func_info


class Tool:
    name: str
    description: str = None
    params = []
    function: callable = None

    def __init__(self, name=None, description=None, params=None, function=None):
        self.name = name or self.name
        self.description = description or self.description
        self.params = params or self.params
        self.function = function or self.function
    
    @classmethod
    def parse(cls, tool):
        if isinstance(tool, Tool):
            return tool
        elif callable(tool):
            tool_info = get_function_info(tool)
            tool = Tool(**{'function': tool, **tool_info})
            return tool
    
    @property
    def info(self):
        func_info = {
            "name": self.name,
            "description": self.description,
            "params": self.params,
        }
        return func_info
    
    def exec(self, *args, **kwargs):
        return self.function(**kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.exec(*args, **kwargs)


ToolList = Union[List[Tool], List[callable]]