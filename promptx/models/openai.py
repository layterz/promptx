from typing import List
import openai

from . import LLM, Response, Metrics, Callback, PromptLog
from ..collection import REGISTERED_ENTITIES


class ChatGPT(LLM):
    version: str = 'gpt-3.5-turbo'
    context: str = '''
    You are a helpful chat assistant.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, x, context=None, history: List[PromptLog]=None, tools=None, **kwargs):
        context = { 'role': 'system', 'content': context or self.context}
        history = history or []
        messages = [context]
        for log in history:
            messages.append({ 'role': 'user', 'content': log.input })
            messages.append({ 'role': 'system', 'content': log.output })
        messages.append({ 'role': 'user', 'content': x })
        
        # OpenAI API has an annoying "feature" where it will throw 
        # an error if you set functions to an empty list or None.
        if tools is None or len(tools) == 0:
            output = openai.ChatCompletion.create(
                model=self.version,
                messages=messages,
            )
        else:
            output = openai.ChatCompletion.create(
                model=self.version,
                messages=messages,
                functions=tools,
            )
        
        message = output.choices[0].message
        function_call = message.get('function_call')
        if function_call is not None:
            callback = Callback(
                name=function_call.get('name'),
                params=message,
            )
        return Response(
            raw=message.get('content'),
            metrics=Metrics(
                model=f'{self.__class__.__name__}.{self.version}',
                input_tokens=output.usage.get('prompt_tokens'),
                output_tokens=output.usage.get('completion_tokens')
            ),
        )

REGISTERED_ENTITIES['chatgpt'] = ChatGPT