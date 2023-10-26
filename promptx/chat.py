from .collection import Collection, Entity
from .template import Template


class DefaultChatTemplate(Template):

    template: str = """
    {{input}}
    Message: {{message}}
    """

    input_templat: str  = """
    Context: {{input}}
    """

    output_templat: str  = """
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