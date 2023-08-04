# Config

## Models

## Vector store

## Logging

## Error handling

Output from LLMs can be unpredictable. **promptz** provides validation through **Pydantic** models, but what happens when a model is unable to generate a valid response? 

By default, a prompt will retry 3 times before throwing an error, but you can change this by setting the `reties=` parameter:

```python
from promptz import prompt

prompt("What is the capital of France?", retries=5)
```