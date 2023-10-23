from .models import LLM
from .utils import Entity
from .collection import Collection
from .template import Template


class DefaultChatTemplate(Template):

    template = """
    {{input}}
    Message: {{message}}
    """

    input_template = """
    Context: {{input}}
    """

    output_template = """
    {{output}}
    """


class ChatBot(Entity):
    name: str
    template: Template
    history: Collection

    def __init__(self, name, template=None, history=None, **kwargs):
        if history is None:
            history = Collection()
        
        if template is None:
            template = DefaultChatTemplate()

        super().__init__(name=name, template=template, history=history, **kwargs)