from .models import LLM
from .utils import Entity
from .collection import Collection


class ChatBot(Entity):

    name: str
    model: LLM
    history: Collection

    def __init__(self, name, model, history=None, **kwargs):
        if history is None:
            history = Collection()
        super().__init__(name=name, model=model, history=history, **kwargs)

    def __call__(self, input, **kwargs):
        print(f'ChatBot: {input}')
        r = self.model.generate(input, **kwargs)
        print(f'ChatBot: {r}')
        return None



