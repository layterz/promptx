# Evals

**promptz** provides simple validation via **Pydantic** models, but this doesn't help to evaluate the quality of the generated output. This is where **evals** comes in.

**evals** let you assess the quality of a prompt output by comparing it to some ideal reference and returning a score from 0 to 100. The score is calculated using the same logic used to query the embeddings space. Here's a simple example:

```python
from promptz import prompt, evaluate

expected = "The capital of France is Paris."
actual = prompt("What is the capital of France?")
score = evaluate(actual, expected)
>>> 100.0
```

In this example the model returns the exact expected output, so the score is 100.0. Let's try a more realistic example:

```python
expected = "Batman is a superhero who protects Gotham City."
actual = prompt("Write a description of Batman.")
score = evaluate(actual, expected)
>>> 87.5
```

In this case the model returns a similar but not identical response, so the score is 87.5. The score is calculated by comparing the embeddings of the expected and actual outputs. The closer the embeddings are, the higher the score.

You can do the same for structured data:

```python
expected = Character(name="Batman", description="Batman is a superhero who protects Gotham City.", age=32)

actual = prompt("Generate a character profile for Batman.", output=Character)
score = evaluate(actual, expected)
>>> 92.3
```

This begs the question: what is a "good" score? There's not a single answer to this question, and will depend on the type of output you are evaluating. But, as a rough rule of thumb, a score of 80 or above is usually a good indicator that the output is of roughly the same format and anything over 90 usually means the content is similar.

To evaluate prompts in production, a better way to think about this is as a form of anomaly detection. I.e. if a prompt typically produces output that scores ~85.0 compared to the expected output, then you could set a threshold of 80.0 and monitor the number of outputs that fall below that value.