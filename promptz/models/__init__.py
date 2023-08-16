import base64
from typing import Any, Dict
from abc import abstractmethod
from pydantic import BaseModel
from IPython.display import display, Image


class ChatLog(BaseModel):
    template: str = None
    input: str = None
    output: str = None


class Callback(BaseModel):
    name: str
    params: Dict[str, Any] = None


class Metrics(BaseModel):
    model: str = None
    input_tokens: int = None
    output_tokens: int = None

    @property
    def total_tokens(self):
        return self.input_length + self.output_length


class Response(BaseModel):
    raw: str = None
    content: Any = None
    metrics: Metrics = None
    callback: Callback = None
    cached: bool = False


class ImageResponse(Response):
    
    def __repr__(self) -> str:
        image_bytes = base64.b64decode(self.raw)
        display(Image(data=image_bytes))


class LLM:

    @abstractmethod
    def generate(self, x) -> Response:
        """Returns the generated output from the model"""


class MockLLM(LLM):
    response_length: int 
    output = None

    def __init__(self, response_length=1000, output=None):
        self.response_length = response_length
        self.output = output

    def generate(self, x, tools=None, **kwargs):
        if self.output is None:
            response = 'This is a mock response.'
        elif isinstance(self.output, BaseModel):
            response = ''
        return Response(
            raw=response,
            metrics=Metrics(
                model='mock',
                input_tokens=len(x),
                output_tokens=self.response_length,
            ),
        )
