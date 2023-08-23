import os
from typing import List
import openai

from . import LLM, ImageResponse, Response, Metrics, Callback, ChatLog


class InstructGPT(LLM):
    version = 'text-davinci-003'

    def __init__(self, version=None):
        self.version = version or self.version

    def generate(self, x, **kwargs) -> Response:
        output = openai.Completion.create(
            model=self.version,
            prompt=x
        )
        text = output.choices[0].text
        return Response(
            raw=text,
            metrics=Metrics(
                model=f'{self.__class__.__name__}.{self.version}',
                input_tokens=len(x),
                output_tokens=len(text),
            )
        )


class ChatGPT(LLM):
    version = 'gpt-3.5-turbo'
    context = '''
    You are a helpful chat assistant.
    '''

    def __init__(self, version=None, context=None, api_key=None, org_id=None):
        self.version = version or self.version
        openai.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        openai.organization = org_id or os.environ.get('OPENAI_ORG_ID')

    def generate(self, x, context=None, history: List[ChatLog]=None, tools=None, **kwargs):
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
        callback = None
        if function_call is not None:
            callback = Callback(
                name=function_call.get('name'),
                params=message,
            )
        return Response(
            raw=message.get('content'),
            callback=callback,
            metrics=Metrics(
                model=f'{self.__class__.__name__}.{self.version}',
                input_tokens=output.usage.get('prompt_tokens'),
                output_tokens=output.usage.get('completion_tokens')
            ),
        )


class DALL_E(LLM):
    
    def __init__(self, api_key=None, org_id=None):
        openai.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        openai.organization = org_id or os.environ.get('OPENAI_ORG_ID')

    def generate(self, x, **kwargs) -> Response:
        output = openai.Image.create(
            prompt=x,
            n=1,
            size='512x512',
            response_format='b64_json',
        )
        return ImageResponse(
            raw=output['data'][0]['b64_json'],
            metrics=Metrics(
                input_tokens=len(x),
            )
        )