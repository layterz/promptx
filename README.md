# promptz

A lightweight library, built on top of pandas and pydantic, that lets you interact with language models and store the output as queryable embeddings.

## Getting starting

To follow along and run the examples interactively you can use [./getting-started.ipynb](https://github.com/layterz/promptz/blob/main/getting-started.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layterz/promptz/blob/main/getting-started.ipynb)

```python
pip install promptz
```

*Note: if install hangs or returns `Killed`, try running `pip install promptz --no-cache-dir`*

First, you need to initialize the library, specifying the language model and embedding function to use.

```python
import os
import promptz
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

llm = promptz.ChatGPT(
    api_key=os.environ['OPENAI_API_KEY'],
    org_id=os.environ['OPENAI_ORGANIZATION_ID'],
)

ef = OpenAIEmbeddingFunction(
    api_key=os.environ['OPENAI_API_KEY'],
    model_name="text-embedding-ada-002",
)

promptz.init(llm=llm, ef=ef)
```

Now we can use the `prompt` helper to make a request to the initialised language model.

```python
from promptz import prompt

character = 'Batman'
prompt(f'Write a character profile for {character}')
```

Output will vary depending on the model, but using ChatGPT we get something like:

```
Name: Bruce Wayne (Batman)                                                      
                                                                                
Age: 35                                                                         
                                                                                
Occupation: Vigilante, CEO of Wayne Enterprises                                 
                                                                                
Physical Appearance: Tall and muscular build, with a dark and brooding presence.
Wears a black batsuit with a cape, utility belt, and a bat symbol on his chest.
His face is covered by a mask, leaving only his piercing blue eyes visible.
                                                                                
Personality: Intense, determined, and highly disciplined. Bruce Wayne is known
for his relentless pursuit of justice and his unwavering commitment to
protecting Gotham City. He is intensely focused, often seen as aloof and...
```

The response is a raw string from the model output. This might be what you want if you're writing an email or asking a question, but other times you'll need to convert the response into structured data.

To do that, you can pass a `pydantic.BaseModel` as `output=` when calling `prompt`. This will add format instructions, based on the pydantic schema and field annotations, and return an instance of the output model using the generated data.

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
```
        type      name                                        description  age
0  character    Batman  Batman, also known as Bruce Wayne, is a billio...   35
1  character     Joker  The Joker, also known as Arthur Fleck, is the ...   40
2  character  Catwoman  Catwoman, also known as Selina Kyle, is a skil...   30
```

This example defines the output as `List[Character]` so a `pd.Dataframe` will be returned with multiple characters. If we used `output=Character` a single `Character` object would be returned instead.

```python
batman = prompt('Generate a character profile for Batman', ouput=Character)
```
```
{"age": 35, "type": "character", "name": "Batman", "id": "ee49c4e5-4c7d-4790-ba1b-80ebff4c43d7", "description": "Batman, also known as Bruce Wayne, is a wealthy philanthropist and vigilant crime-fighter who operates in Gotham City. Driven by the tragic murder of his parents, Batman uses his intelligence, strength, and gadgets to protect his city from chaos and corruption."}
```

You can further guide the model output by providing few shot examples. Here, we first generate some new descriptions for the characters and update the existing objects.

```python
descriptions = prompt(
    'Write some 2-3 sentence character descriptions in the style of Alan Moore',
    input={'characters': [c.name for c in characters.objects]},
    output=List[str],
)

characters['description'] = descriptions['output']
```
```
    type	    name	    description	                                        age
0	character	Batman	    'Batman' is a brooding figure, his cape flowin...	35
1	character	Joker	    'Joker' is a chaotic force, his face twisted i...	40
2	character	Catwoman	'Catwoman' is a seductive thief, her lithe for...	30
```

Run the above until it's generated some output you're happy with and then we can use them as examples to guide future output.

```python
import pandas as pd
from promptz import Prompt

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
    existing_characters = characters['name'].to_list()
    cs = prompt(prompt=p, input={'existing_characters': existing_characters})
    characters = pd.concat([characters, cs]).reset_index(drop=True)
```

Here we iterate 5 times, on each iteration the existing character names are passed to the prompt and are used to prevent duplicates from being generated. The initial characters generated above are used as examples to demonstrate avoiding duplicates as well as guideing the description style.

Running this for 5 iterations on ChatGPT generates something like this.
```
	type	    name	        description	                                        age
0	character	Batman	        'Batman' is a brooding figure, his cape flowin...	35
1	character	Joker	        'Joker' is a chaotic force, his face twisted i...	40
2	character	Catwoman	    'Catwoman' is a seductive thief, her lithe for...	30
3	character	Two-Face	    Once a respected district attorney, Two-Face's...	42
4	character	Harley Quinn	Harley Quinn, formerly known as Dr. Harleen Qu...	29
5	character	Riddler	        The Riddler, also known as Edward Nygma, is ob...	35
6	character	Scarecrow	    Dr. Jonathan Crane, better known as the Scarec...	37
7	character	Poison Ivy	    Pamela Isley, also known as Poison Ivy, posses...	28
8	character	Firefly	        Firefly is a pyromaniac and an expert in fire ...	35
9	character	Man-Bat	        Man-Bat is a scientist who accidentally turns ...	45
10	character	Mad Hatter	    Mad Hatter is a deranged inventor who uses min...	50
11	character	Black Mask	    Black Mask is a ruthless crime lord who wears ...	40
12	character	Mr. Freeze	    Mr. Freeze is a scientist who, after a lab acc...	55
13	character	Red Hood	    Formerly the second Robin, Jason Todd, the Red...	25
14	character	Nightwing	    Dick Grayson, the original Robin, now patrols ...	27
15	character	Batwoman	    Kate Kane, the cousin of Bruce Wayne, took up ...	28
16	character	Hush	        Dr. Thomas Elliot, a childhood friend of Bruce...	35
17	character	Oracle	        Formerly known as Batgirl, Barbara Gordon now ...	31
18	character	Azrael	        Jean-Paul Valley, an assassin trained by a sec...	30
19	character	Ace             Ace is a highly intelligent and agile dog that...	5
20	character	Echo	        Echo is a skilled thief with the ability to di...	25
21	character	Siren	        Siren is a seductive and dangerous femme fatal...	28
22	character	Talon	        Talon is a highly trained assassin who serves ...	35
23	character	Shadow	        Shadow is a mysterious vigilante who operates ...	30
24	character	Batgirl	        Batgirl is the female counterpart to Batman. S...	25
25	character	Penguin	        The Penguin is a cunning and manipulative crim...	45
26	character	Black Canary	Black Canary is a highly trained martial artis...	28
27	character	Killer Croc     Killer Croc is a mutated crocodile-human hybri...	35
28	character	Professor Pyg	Professor Pyg is a deranged surgeon and sadist...	40
```

You can store any generated output in a `Collection` using the `store` helper.  Collections are dataframes that also support querying using embeddings by using vector space modelling to convert each field into an embedding.

```python
from promptz import store

store(characters)
```
```
    id	                                    description	                                        type	    name	            age
0	646c1d45-f5d1-45b6-b9a7-ca029df51d8f	Batman is a brooding vigilante with a traumati...	character	Batman	            32
1	d22ce125-2d4d-4b84-b447-c63f648820c2	Joker is the embodiment of chaos and unpredict...	character	Joker	            36
2	c3501a5a-9d40-40d2-8d68-76aa3f3cd9f3	Catwoman is a skilled thief and a master of di...	character	Catwoman	        29
3	282d4ae3-da39-4203-b71b-a249ef2d8f81	Robin is Batman's steadfast and loyal sidekick...	character	Robin	            18
4	94092fa7-51c1-4120-804c-413a3ab6df9a	Batgirl is a highly skilled martial artist and...	character	Batgirl	            25
5	5cac4b09-7f27-421e-9e94-985eab41b4b6	The Riddler is a cunning and intelligent crimi...	character	The Riddler	        35
...
```

For example, `[{name: 'Batman'}, {description: 'Batman, also known as...'}, {age: 35}]` would be converted into 3 embeddings for the first item. You can then query collections using those field embeddings.

```python
from promptz import query

villains = query('they are a villain')
```
```
    id	                                    description	                                        type	    name	            age
26	7143239c-9a6a-4dc3-84a6-f4c53faf8dda	Nightstrike is a former Gotham City Police off...	character	Nightstrike	        35
27	03f387bd-3264-4add-a296-fa74a7e9fa5d	Crimson Shade is a master of deception and ill...	character	Crimson Shade	    29
17	1bad4da4-a6d1-408e-a785-61649fbe7a8e	The Enforcer is a highly trained martial artis...	character	The Enforcer	    28
6	61896717-f5f5-4d79-9f46-268232b2a1b0	Nightshade is a mysterious vigilante who uses ...	character	Nightshade	        34
1	d22ce125-2d4d-4b84-b447-c63f648820c2	Joker is the embodiment of chaos and unpredict...	character	Joker	            36
11	8a883f0c-323b-4ae9-b920-ea5f9a0a51b2	Two-Face is the alter ego of Harvey Dent, a fo...	character	Two-Face	        41
8	094b9b13-eaf3-43ea-97aa-35eb2251e504	Gotham Guardian is an armored warrior who patr...	character	Gotham Guardian	    40
20	88a6e9e6-900b-4754-8802-0518720b0cdf	Oracle is the superhero identity of Barbara Go...	character	Oracle	            30
18	ffc823ef-43b8-40a9-89c3-7f2879080295	The Nightwatcher is a skilled acrobat and park...	character	The Nightwatcher    25
5	5cac4b09-7f27-421e-9e94-985eab41b4b6	The Riddler is a cunning and intelligent crimi...	character	The Riddler	        35
```

Data can be updated by transforming the collection data using standard pandas methods and passing the result back to `store`.

```python
villains['evil'] = True
store(villains)
```
```
	id	                                    description	                                        type	    name	    age	evil
0	646c1d45-f5d1-45b6-b9a7-ca029df51d8f	Batman is a brooding vigilante with a traumati...	character	Batman	    32	NaN
1	d22ce125-2d4d-4b84-b447-c63f648820c2	Joker is the embodiment of chaos and unpredict...	character	Joker	    36	True
2	c3501a5a-9d40-40d2-8d68-76aa3f3cd9f3	Catwoman is a skilled thief and a master of di...	character	Catwoman	29	NaN
3	282d4ae3-da39-4203-b71b-a249ef2d8f81	Robin is Batman's steadfast and loyal sidekick...	character	Robin	    18	NaN
4	94092fa7-51c1-4120-804c-413a3ab6df9a	Batgirl is a highly skilled martial artist and...	character	Batgirl	    25	NaN
5	5cac4b09-7f27-421e-9e94-985eab41b4b6	The Riddler is a cunning and intelligent crimi...	character	The Riddler	35	True
6	61896717-f5f5-4d79-9f46-268232b2a1b0	Nightshade is a mysterious vigilante who uses ...	character	Nightshade	34	True
...
```

Queries return a collection of results meaning it can be further filtered or queried.

```python
old_and_masked = villains('they wear a mask')[villains['age'] > 40]
```
```
	id	                                    description	                                        type	    name	    age	evil
11	8a883f0c-323b-4ae9-b920-ea5f9a0a51b2	Two-Face is the alter ego of Harvey Dent, a fo...	character	Two-Face	41	True
```

Now lets pull them together to generate some story ideas.

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
```
	type	    title	                description	                                        characters
0	storyidea	The Dark Knight Rises	Batman must face off against the combined forc...	[Batman, Two-Face, Joker, Crimson Shade]
1	storyidea	Gotham's Shadows	    Batman and Crimson Shade form an unlikely alli...	[Batman, Crimson Shade, Two-Face, Joker]
2	storyidea	Unmasked Truth	        As Batman fights against the chaos unleashed b...	[Batman, Joker, Two-Face]
3	storyidea	Illusions of Justice	Crimson Shade's illusions lead Batman to quest...	[Batman, Crimson Shade, Two-Face, Joker]
```

Finally, let's store those ideas, but instead of using the default collection we'll add them to a new `story_ideas` collection.

```python
store(*ideas, collection='story_ideas')
collection('story_ideas')
```
```
    id	                                    description	                                        type	    characters	                                title
0	4fad21db-014d-4b6f-ba69-ab5b83d440a6	Batman must face off against the combined forc...	storyidea	[Batman, Two-Face, Joker, Crimson Shade]	The Dark Knight Rises
1	11578c9f-037e-45eb-8c49-9fa2d4a61245	Batman and Crimson Shade form an unlikely alli...	storyidea	[Batman, Crimson Shade, Two-Face, Joker]	Gotham's Shadows
2	d0b171f4-2f6d-4477-a6a8-47f872af4bd3	As Batman fights against the chaos unleashed b...	storyidea	[Batman, Joker, Two-Face]	                Unmasked Truth
3	9f16b44c-6f50-4ac0-b669-fd2dcd75c5d1	Crimson Shade's illusions lead Batman to quest...	storyidea	[Batman, Crimson Shade, Two-Face, Joker]	Illusions of Justice
```