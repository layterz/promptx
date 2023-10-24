import base64
import uuid
from typing import Any, Dict
from abc import abstractmethod
from pydantic import BaseModel
from IPython.display import display, Image


from promptx.utils import Entity


class PromptLog(BaseModel):
    id: str
    type: str = 'prompt'
    template: str = None
    input: str = None
    raw_input: str = None
    output: str = None
    raw_output: str = None
    error: str = None

    def __init__(self, **data):
        super().__init__(
            id=str(uuid.uuid4()), **data)


class QueryLog(BaseModel):
    id: str
    type: str = 'query'
    query: list[str] = None
    where: dict = None
    collection: str = None
    result: str = None

    def __init__(self, **data):
        super().__init__(
            id=str(uuid.uuid4()), **data)


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


class LLM(Entity):

    @abstractmethod
    def generate(self, x) -> Response:
        """Returns the generated output from the model"""


class MockLLM(LLM):
    output: str = None

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
            ),
        )


__MODELS__ = {}
def register_model(model: LLM):
    """Registers a model to the global session"""
    __MODELS__[model.name] = model