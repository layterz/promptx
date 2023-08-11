import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer, LlamaForCausalLM, LlamaTokenizer

from . import LLM, Response, Metrics

class HuggingfaceTransformer(LLM):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    max_length: int

    def __init__(self, model, tokenizer, max_length=50, **kwargs):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate(self, x, context=None, **kwargs):
        input_tokens = self.tokenizer.encode(x, return_tensors='pt')
        generated_tokens = input_tokens
        with torch.no_grad():
            for _ in range(self.max_length):
                predictions = self.model(generated_tokens)[0]
                predicted_token = torch.argmax(predictions[0, -1, :]).unsqueeze(0)
                generated_tokens = torch.cat(
                    (generated_tokens, predicted_token.unsqueeze(0)), dim=1)
        
        generated_text = self.tokenizer.decode(
            generated_tokens[0][input_tokens.shape[1]:])

        return Response(
            raw=generated_text,
            metrics=Metrics(
                model=f'{self.__class__.__name__}',
                input_tokens=len(x),
                output_tokens=len(generated_text),
            ),
        )


class Llama(HuggingfaceTransformer):
    
    def __init__(self, model_path):
        model = LlamaForCausalLM.from_pretrained(model_path)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        super().__init__(model, tokenizer)