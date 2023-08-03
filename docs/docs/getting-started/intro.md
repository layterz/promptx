# Introduction

**promptz** is a framework for building complex prompt-based applications. It provides a simple, flexible API for interacting with generative models, which uses **Pydantic** models to define the expected prompt output. Once you have generated some data you can store the output as **embeddings** in collections, which use **Pandas** to provide a simple, familiar interface.

## Installation

First, install with pip:

```bash
pip install promptz
```

If this hangs or fails, try running with `--no-cache-dir`.

## Setup

Next, we need to initialize the library with the LLM you want to use. Here's an example using OpenAI's ChatGPT:

```python
import os
import promptz

llm = promptz.ChatGPT(
    api_key=os.environ['OPENAI_API_KEY'],
    org_id=os.environ['OPENAI_ORGANIZATION_ID'],
)

promptz.init(llm=llm)
```

Now, let's test it's working with a simple prompt:

```python
promptz.prompt('What is the capital of France')
>>> "The capital of France is Paris."
```

## Structured output

The real power of **promptz** comes from the ability to define structured output using **Pydantic** models. Let's define a simple model for a character:

```python
from typing import List
from pydantic import BaseModel, Field
from promptz import prompt

class Character(BaseModel):
    name: str = Field(..., unique=True, embed=False),
    description: str = Field(
        ..., description='Describe the character in a few sentences')
    age: int = Field(..., min=1, max=120)

characters = prompt(
    'Generate some characters from the Batman universe',
    output=List[Character],
)
```

This will return a list of `Character` model instances wrapped in a **Pandas** DataFrame:

```
	type	    name	    description	                                        age
0	character	Batman	    'Batman' is a brooding figure, his cape flowin...	35
1	character	Joker	    'Joker' is a chaotic force, his face twisted i...	40
2	character	Catwoman	'Catwoman' is a seductive thief, her lithe for...	30
```

To access the models you can use the `objects` helper:

```python
characters.objects[0]
>>> Character(name='Batman', description="Batman is a wealthy businessman and philanthropist who becomes a vigilante to fight crime in Gotham City. He uses his intelligence, strength, and technology to take down criminals and protect the innocent.", age=35) 
```

## Storing output

Now we have some data let's store it in a collection.

```python
from promptz import store

store(characters)
```

This returns a `Collection`, which wraps a **Pandas** DataFrame:

```
	id	                                    name	    description	                                        type	    age
0	ef3061a8-f03e-4425-9411-2873457d72f0	Batman	    'Batman' is a brooding figure, his cape flowin...	character	35
1	a8a3e93b-d186-4900-a065-4cbbad7d0e6b	Joker	    'Joker' is a chaotic force, his face twisted i...	character	40
2	94cccb31-4ac7-4794-936e-dc8057ca6306	Catwoman	'Catwoman' is a seductive thief, her lithe for...	character	30
```


This stores the data in the default collection, but you can also create new collections:

```python
store(characters, name='characters')
```

## Querying collections

Now we have stored some data we can query it by directly calling the collection:

```python
from promptz import query

villains = query('They are a villain')
```

```
	id	                                    name	    description	                                        type	    age
0	a8a3e93b-d186-4900-a065-4cbbad7d0e6b	joker	    'joker' is a chaotic force, his face twisted i...	character	40
1	94cccb31-4ac7-4794-936e-dc8057ca6306	catwoman	'catwoman' is a seductive thief, her lithe for...	character	30
2	ef3061a8-f03e-4425-9411-2873457d72f0	batman	    'batman' is a brooding figure, his cape flowin...	character	35
```

When data is stored in a collection, embeddings are generated for each field on the model - in this case embeddings would be created for name, description and age. When collections are queried the input text is compared to the embeddings for each field and the results are sorted by the closest match across all fields.

If you want to query a specific field you can pass `field=name` to the query:

```python
villains = characters('They are a villain', field='description')
```

The response from querying a collection is also a `Collection`, so you can chain queries and use standard **Pandas** to further filter the results:

```python
old_masked_villains = villains('They wear a mask')[villains['age'] > 40]
```
```
	id	                                    name	    description	                                        type	    age
0	a8a3e93b-d186-4900-a065-4cbbad7d0e6b	joker	    'joker' is a chaotic force, his face twisted i...	character	40
```

To get the first row as a model instance you can use the `first` helper:

```python
old_masked_villains.first
>>> Character(name='Joker', description="Joker is a chaotic force, his face twisted into a permanent grin. He wears a purple suit and a green bow tie, and his hair is dyed green. He is a master of chaos and an agent of anarchy.", age=40) 
```

## Few shots

Providing a small number of examples that show the model how you want it to generate data is a powerful way to control the output. You can pass few shots using the `examples=` parameter with a list of tuples containing the input and output.

Let's define few shots for our character model using the data we've stored in the collection:

```python
from promptz import Prompt, query

characters = query(where={'type': 'character'})
p = Prompt(
    '''
    Generate a list of new characters from the Batman universe.
    Don't use any of the existing characters.
    ''',
    examples=[
        (
            { 'existing_characters': characters['name'][:2].to_list() },
            characters[2:].objects,
        ),
    ],
    output=List[Character],
)

for _ in range(5):
    existing_characters = query(where={'type': 'character'})
    try:
        cs = prompt(prompt=p, input={'existing_characters': existing_characters})
        store(cs)
        characters = pd.concat([characters, cs]).reset_index(drop=True)
    except Exception as e:
        print(e)

characters
```

This example is a bit more complicated so lets breakdown what's going on.

First, we fetch the existing characters from the collection and define a prompt instance with `examples=` using that data. In this case we are using the examples to both demonstrate the style of output we want and also to tell the model not to use any of the existing characters.

*Note: we're using `Prompt` to create an instance instead of directly executing it using `prompt()` as we want to reuse the same prompt instance for each iteration of the loop.*

Next, we loop over the prompt 5 times, each time passing the existing characters to the prompt as input. This allows the model to learn from the existing characters and generate new ones.

Finally, we store the new characters in the collection. Here's the result:

```
	id	                                    name            description	                                        type	    age
0	ef3061a8-f03e-4425-9411-2873457d72f0	Batman	        'Batman' is a brooding figure, his cape flowin...	character	35
1	a8a3e93b-d186-4900-a065-4cbbad7d0e6b	Joker	        'Joker' is a chaotic force, his face twisted i...	character	40
2	94cccb31-4ac7-4794-936e-dc8057ca6306	Catwoman	    'Catwoman' is a seductive thief, her lithe for...	character	30
3	693aa746-15af-4e51-8687-2e0aa052e0ee	Two-Face	    Once a respected district attorney, Two-Face's...	character	42
4	c6e9b988-54cc-498a-b0a2-3a03eeae62c5	Harley Quinn	Harley Quinn, formerly known as Dr. Harleen Qu...	character	29
5	47f436c6-396c-42f0-8283-b72280e3bf7a	Riddler	        The Riddler, also known as Edward Nygma, is ob...	character	35
6	0bee3f9f-627c-4d8f-9235-75c5277c008d	Scarecrow	    Dr. Jonathan Crane, better known as the Scarec...	character	37
7	3ce572cc-e970-4ad0-ac50-50fc07a34114	Poison Ivy	    Pamela Isley, also known as Poison Ivy, posses...	character	28
8	c11d5797-df30-4bfc-8657-102d29ef2d0e	Firefly	        Firefly is a pyromaniac and an expert in fire ...	character	35
9	2cf55198-0608-46de-b99e-62816a3cc22b	Man-Bat	        Man-Bat is a scientist who accidentally turns ...	character	45
10	368fcef1-024d-4240-8e56-2fea45ecd015	Mad Hatter	    Mad Hatter is a deranged inventor who uses min...	character	50
11	a517bfb5-348f-4805-b193-e6e38a82dff8	Black Mask	    Black Mask is a ruthless crime lord who wears ...	character	40
12	7cd54180-9bb4-4ddb-b086-2663cbb3f99f	Mr. Freeze	    Mr. Freeze is a scientist who, after a lab acc...	character	55
13	9a30169c-6c5a-48e8-b342-446bdd22649d	Red Hood	    Formerly the second Robin, Jason Todd, the Red...	character	25
14	c25ad735-3b07-44a6-b049-81c3718e823c	Nightwing	    Dick Grayson, the original Robin, now patrols ...	character	27
15	5b491de9-55ad-49af-9199-27f9c0e08c9c	Batwoman	    Kate Kane, the cousin of Bruce Wayne, took up ...	character	28
16	571c753e-1c5a-4748-b9b3-4bced34f6f9b	Hush	        Dr. Thomas Elliot, a childhood friend of Bruce...	character	35
17	da1d4926-5e06-4b2f-b781-74b8f31947a6	Oracle	        Formerly known as Batgirl, Barbara Gordon now ...	character	31
18	25bccd8e-9237-4f1c-89a6-bd2338eaff21	Azrael	        Jean-Paul Valley, an assassin trained by a sec...	character	30
19	762ec2cb-1f82-4756-9573-a8073800b388	Ace	            Ace is a highly intelligent and agile dog that...	character	5
20	0f5b46af-1051-4baf-8afd-6df546edf48a	Echo	        Echo is a skilled thief with the ability to di...	character	25
21	4a07195a-2cdf-42d5-987f-daf39a57e4f2	Siren	        Siren is a seductive and dangerous femme fatal...	character	28
22	62a09be3-dac3-4a99-956e-67418e9230e1	Talon	        Talon is a highly trained assassin who serves ...	character	35
23	d9e96327-753e-4b99-867a-99b610b218e3	Shadow	        Shadow is a mysterious vigilante who operates ...	character	30
24	590d1c0f-dd59-479d-a675-53a1ba55e91b	Batgirl	        Batgirl is the female counterpart to Batman. S...	character	25
25	74d802e0-30dd-46c3-a57c-b8594b7747db	Penguin	        The Penguin is a cunning and manipulative crim...	character	45
26	e694ca90-d7f5-40a1-9377-86d9a6b00b41	Black Canary	Black Canary is a highly trained martial artis...	character	28
27	f8e7247b-541b-4ee6-bbd8-65bb300cfa5d	Killer Croc	    Killer Croc is a mutated crocodile-human hybri...	character	35
28	e037e9e7-7965-4508-9134-6cdf2f886b03	Professor Pyg	Professor Pyg is a deranged surgeon and sadist...	character	40
```

## Updating data

You can update existing data using standard **Pandas** transformations and storing the updated dataframe:

```python
villains['evil'] = True
store(villains)
```

This adds an `evil` column to the dataframe and assigns it a value of `True` for all rows in the villain collection.

## Wrap up

Finally, lets use the techniques we've learned to generate some story ideas:

```python
class StoryIdea(BaseModel):
    title: str
    description: str = None
    characters: List[str] = []

batman = query('Bruce Wayne').first
villains = query('they are a villain').sample(3)

ideas = prompt(
    'Generate some story ideas',
    input={
        'characters': [batman] + villains.objects,
    },
    output=List[StoryIdea],
)
```

Now lets store them in a new collection:

```python
from promptz import collection

store(ideas, collection='story_ideas')
collection('story_ideas')
```

```
	id	                                    title	                description	                                        type	    characters
0	4a234823-b560-4aae-a152-be970d49ff22	The Dark Knight Rises	As Gotham City faces its darkest hour, Batman ...	storyidea	[Batman, Shadow, Talon, Black Mask]
1	a89c556e-952a-4075-a447-429dc2c7fd89	Shadows of Revenge	    When a series of mysterious murders shakes Got...	storyidea	[Shadow, Talon, Black Mask]
2	47ee7f37-07d3-4b79-862a-387f52ef7779	Deadly Pursuit	        Talon embarks on a relentless pursuit to elimi...	storyidea	[Talon, Batman, Shadow]
3	d62d296e-5821-4455-8d48-335749d01e59	The Reign of Black Mask	Black Mask's grip on Gotham City tightens as h...	storyidea	[Black Mask, Batman]
```

## Next steps