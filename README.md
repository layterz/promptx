# promptx

A framework for building AI systems.

```bash
pip install pxx
```

First, we need to create a project, which defines the embedding database that's being used.

```bash
px init .
```

*This will create a project in the current directory. Replace `.` with a path to create a project in a different directory.*

A project is defined by a hidden `.px/` directory that `promptx` uses to store data and discover the project.

```python
from promptx import load

load()
```

Now that `promptx` is loaded, we can call `prompt` to generate some data with the default model.

```python
from promptx import prompt

character = 'Batman'
prompt(f'Write a character profile for {character}')
```

```
'Name: Batman\nOccupation: Superhero/Vigilante\nAlias: The Dark Knight, The Caped Crusader\nReal Name: Bruce Wayne\nAge: Late 30s to early 40s\nHeight: 6\'2"\nWeight: 210 lbs\nEthnicity: Caucasian\nNationality: American\nPlace of Birth: Gotham City\nParents: Thomas and Martha Wayne (deceased)\nAffiliations: Justice League, Bat-Family, Gotham City Police Department (unofficially)\nSkills/Abilities: Exceptionally skilled detective, master martial artist and hand-to-hand combatant, peak human physical condition, proficient in various forms of weaponry and gadgets, expert in stealth and espionage, genius-level intellect and deductive reasoning\nCostume: Black bat-themed bodysuit, utility belt with various gadgets, protective body armor, cape with bat-like wingspan\nVehicle: Batmobile...
```

By default, this returns a plain string response, but to generate complex data you can pass in the expected schema along with the prompt input.

*Note: `Entity` is a thin layer on top of `pydantic.BaseModel` that allows the object to be stored as an embedding. You can use `pydantic.BaseModel` directly if you don't need to store the object as an embedding and just want to use it as the prompt output schema.*

```python
from pydantic import Field
from promptx.collection import Entity

class Character(Entity):
    name: str = Field(..., embed=False),
    description: str = Field(..., description='Describe the character in a few sentences')
    age: int = Field(..., ge=0, le=120)

batman = prompt('Generate a character profile for Batman', output=Character)
batman
```

```
Character(
    id='dd740305-737a-4bde-8596-3525490d2f6f',
    type='character',
    name='Bruce Wayne',
    description='Batman is a superhero who operates in Gotham City. He is known for his detective skills, martial arts training, and use of high-tech gadgets to fight crime. Bruce Wayne, his alter ego, is a billionaire philanthropist and the owner of Wayne Enterprises.',
    age=35
)
```

This returns an instance of the specified schema using the generated response as the input data. Let's create a list of instead.

```python
characters = prompt(
    'Generate some characters from the Batman universe',
    output=[Character],
)

characters
```

```
id	type	name	description	age
0	7ec67bd1-0626-4293-ab0a-509cee7f5b66	character	Batman	Bruce Wayne is a billionaire philanthropist by...	35
1	7d7e53e6-f1dd-4f0a-bda7-6533ba385960	character	Joker	The Joker is a deranged and unpredictable crim...	40
2	45157783-7006-476e-a492-2f6881d2410d	character	Catwoman	Selina Kyle is a skilled thief with a complica...	30
```

If the output is a list, `prompt` returns a `Collection`, which extends `pd.DataFrame`. To extract the `Entity` representations, use the `objects` property.

We can now store these generated objects as embeddings in a collection.

```python
from promptx import store

store(*characters.objects)
```

This stores the object as an embedding, along with some metadata, in a vector database (ChromaDB by default). The process is quite simple, it embeds the whole object as a JSON string and each field individually. This allows us to query the database using any field in the object.

```python
from promptx import query

query()
```

```
id	type	name	description	age
0	7ec67bd1-0626-4293-ab0a-509cee7f5b66	character	Batman	Bruce Wayne is a billionaire philanthropist by...	35
1	7d7e53e6-f1dd-4f0a-bda7-6533ba385960	character	Joker	The Joker is a deranged and unpredictable crim...	40
2	45157783-7006-476e-a492-2f6881d2410d	character	Catwoman	Selina Kyle is a skilled thief with a complica...	30
```

Now let's generate some more characters and add them to the collection. We'll first get any existing characters and extract their names, which we can pass to the prompt to avoid generating duplicates. Any characters generated will be added the list during iteration. Finally, we'll store all the generated characters in the collection.

```python
n = 3
characters = query().objects

for _ in range(n):
    characters += prompt(
        '''
        Generate a list of new characters from the Batman universe.
        Don't use any of the existing characters.
        ''',
        input = {
            'existing_characters': [c.name for c in characters],
        },
        output=[Character],
    ).objects

store(*characters)
query()
```

```
id	type	name	description	age
0	7ec67bd1-0626-4293-ab0a-509cee7f5b66	character	Batman	Bruce Wayne is a billionaire philanthropist by...	35
1	7d7e53e6-f1dd-4f0a-bda7-6533ba385960	character	Joker	The Joker is a deranged and unpredictable crim...	40
2	45157783-7006-476e-a492-2f6881d2410d	character	Catwoman	Selina Kyle is a skilled thief with a complica...	30
3	2ad9eb88-406a-4d55-8e19-a5cb0dc7035f	character	Nightshade	Nightshade is a highly skilled acrobat and thi...	28
4	63472046-2be4-40a0-a4b0-ba918e5a7ad4	character	Shadowstrike	Shadowstrike is a master of stealth and weapon...	33
5	97fd9c56-9531-4444-b7d6-246070b98405	character	Twilight	Twilight is a vigilante who fights for justice...	25
6	b40dfcbf-107e-4867-adad-1de2c1260fef	character	Gotham Girl	Gotham Girl is a young superhuman who gained h...	18
7	97c965d9-2669-49c1-8817-1ab53f1f97ff	character	Midnight	Midnight is a master of stealth and prefers to...	28
8	16c569fe-18ee-4b18-8c86-428c05cfbd2d	character	Raven	Raven is a sorceress who possesses the ability...	25
9	dc555573-05f5-4aae-babe-16ece0c401f8	character	Harbinger	Harbinger is a mysterious and enigmatic vigila...	35
10	c8945846-4cf0-4e6f-ab25-82ec8fbdb8c8	character	Silverwolf	Silverwolf is a skilled martial artist with a ...	28
11	56e0004f-a656-41fb-902a-9b46522c4e23	character	Luna	Luna is a master of illusions and deception. S...	27
```

This compares the query text with the stored objects, returning results that are closest in vector space.

*Note: the effectiveness of embedding queries will depend on what data has been embedded. In this case, ChatGPT will know some details about the generated characters and so does a decent job on this data. For other data, you may find generating synthetic intermediary data to be helpful. E.g. generating `thoughts` and/or `quotes` about a set of documents.*

Because `Collection` extends `pd.DataFrame`, we can use all the usual Pandas methods to filter and sort the results.

```python
villains[villains.age < 30]
```

```
	id	type	name	description	age
0	c8945846-4cf0-4e6f-ab25-82ec8fbdb8c8	character	Silverwolf	Silverwolf is a skilled martial artist with a ...	28
3	97fd9c56-9531-4444-b7d6-246070b98405	character	Twilight	Twilight is a vigilante who fights for justice...	25
4	97c965d9-2669-49c1-8817-1ab53f1f97ff	character	Midnight	Midnight is a master of stealth and prefers to...	28
5	56e0004f-a656-41fb-902a-9b46522c4e23	character	Luna	Luna is a master of illusions and deception. S...	27
```

Relationships can be defined by setting the field to a type which subclasses `Entity` (or a list of that type). Internally, this is stored as a query and then loaded when the field is accessed from the database.

```python
class StoryIdea(Entity):
    title: str
    description: str = None
    characters: list[Character] = None

characters = query('they are a villain').sample(3).objects

ideas = prompt(
    'Generate some story ideas',
    input={
        'characters': characters,
    },
    output=[StoryIdea],
).objects

for idea in ideas:
    idea.characters = characters

store(*ideas, collection='story-ideas')
query(collection='story-ideas')
```

```
id	type	title	description	characters
0	886bc464-7b4a-4e4e-b228-e3791c6d7f77	storyidea	The Revenge of Silverwolf	After years of training and honing his skills,...	[{'id': 'c8945846-4cf0-4e6f-ab25-82ec8fbdb8c8'...
1	63668f1a-8a1f-4d68-9926-7b6cb83147fb	storyidea	The Shadow's Game	Midnight finds herself caught in a deadly game...	[{'id': 'c8945846-4cf0-4e6f-ab25-82ec8fbdb8c8'...
2	cedc978b-463f-43e1-9898-690aada5b832	storyidea	The Illusionist's Gambit	Luna finds herself lured into a dangerous game...	[{'id': 'c8945846-4cf0-4e6f-ab25-82ec8fbdb8c8'...
```

Note that the output is being stored in a collection called `story-ideas`, which is created if it doesn't exist. Previously, all the data we've stored has been in the 'default' collection.

*Collections are widely used internally to represent stored models, templates, prompt history, etc. This provides a consistent interface for accessing and manipulating data.*

So far we've used the default model (GPT-3.5) when generating data, but you can specify a custom model using the `llm=` parameter.

```python
from promptx.models.openai import ChatGPT

gpt4 = ChatGPT(id='gpt4', model='gpt4')

characters = prompt(
    'Generate some characters from the Batman universe',
    output=[Character],
    llm=gpt4,
)
```

You can define any commonly used models, templates, etc, along with defining other settings, by creating a `config.py` file in the root of the project (i.e. adjacent to the `.px/` directory). This file is loaded when the project is initialized and a `setup` function is expected. Here's a simple example that defines a few custom models and a template.

```python
# ./config.py

from promptx.models.openai import ChatGPT

gpt4 = ChatGPT(id='gpt4', model='gpt4')

def setup(session):
    session.store(gpt4, collection='models')
```