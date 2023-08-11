import requests
import pandas as pd
from pydantic import BaseModel
from dash import Dash, html, dcc, dash_table, page_container, page_registry, register_page, Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from . import World

API_URL = 'http://localhost:8000'


class AdminPage(BaseModel):
    name: str
    path: str = None
    path_template: str = None

    def layout(self, **kwargs):
        return html.Div(children=[
            html.H1(self.name),
            html.P('Not implemented yet'),
        ])


class Admin:
    world: World

    def __init__(self, world, logger=None):
        self.world = world
        self.logger = logger or world.logger.getChild('admin')
        self.app = Dash(
            world.name, 
            use_pages=True,
            pages_folder='',
            external_stylesheets=[dbc.themes.ZEPHYR],
        )

        def prompts_list_layout():
            response = requests.get(f'{API_URL}/prompts')
            if response.status_code == 200:
                prompts = response.json()
            else:
                raise Exception(f'Error getting prompts: {response.status_code}')
            
            if len(prompts['response']) == 0:
                content = [
                    html.Div([
                        html.P('No prompts found.'),
                    ])
                ]
            else:
                df = pd.DataFrame(prompts['response'])

                def generate_open_link(row):
                    return f'[{row["name"]}](/prompts/{row["id"]})'
                
                df['name'] = df.apply(lambda row: generate_open_link(row), axis=1)
                df = df[
                    ['name', 'instructions']
                ]
                content = [
                    dash_table.DataTable(
                        id='prompts-table',
                        columns=[{"name": i, "id": i, 'presentation': 'markdown'} for i in df.columns],
                        data=df.to_dict('records'),
                    ),
                ]

            return html.Div([
                html.H1('Prompts'),
                *content,
            ])

        register_page(
            'Prompts',
            layout=prompts_list_layout,
            path='/prompts',
        )

        def prompt_layout(id: str = None):
            response = requests.get(f'{API_URL}/prompts/{id}')
            if response.status_code == 200:
                data = response.json()
                prompt = data['prompt']
                results = data['results']
            else:
                raise Exception(f'Error getting prompt: {response.status_code}')
            
            if len(results) > 0:
                df = pd.DataFrame(results)

                def generate_prompt_link(id):
                    return f'[{id}](/prompts/{id})'
                
                df['prompt'] = df['prompt'].apply(lambda id: generate_prompt_link(id))
                df = df[
                    ['prompt', 'input', 'output']
                ]

                results_table = dash_table.DataTable(
                    id='results-table',
                    columns=[{"name": i, "id": i, 'presentation': 'markdown'} for i in df.columns],
                    data=df.to_dict('records'),
                )
            else:
                results_table = html.Div([
                    html.P('No results found.'),
                ])

            return html.Div(children=[
                html.H1(prompt['name']),
                html.P(prompt['instructions']),
                dbc.Button('Run', id='run-prompt', n_clicks=0, name=id, color='primary'),
                dcc.Store(id='api-call-result', storage_type='session'),
                html.Div(id, id='prompt-id', style={'display': 'none'}),
                html.H2('Results'),
                results_table,
            ])

        register_page('Prompt', layout=prompt_layout, path_template='/prompts/<id>')

        @self.app.callback(
            Output('api-call-result', 'data'),
            [Input('run-prompt', 'n_clicks')],
            [State('api-call-result', 'data'),
             State('prompt-id', 'children')],
        )
        def run_prompt(n_clicks, current_data, id):
            if n_clicks is None or n_clicks == 0:
                raise PreventUpdate
            else:
                requests.post(f'{API_URL}/prompts/{id}/run')

        def history_layout():
            response = requests.get(f'{API_URL}/history')
            if response.status_code == 200:
                history = response.json()['response']
            else:
                raise Exception(f'Error getting prompts: {response.status_code}')
            
            if len(history) == 0:
                return html.Div(children=[
                    html.H1(children='History'),
                    html.P('No history yet'),
                ])
            
            df = pd.DataFrame(history)

            def generate_prompt_link(id):
                return f'[{id}](/prompts/{id})'
            
            df['prompt'] = df['prompt'].apply(lambda id: generate_prompt_link(id))
            df = df[
                ['prompt', 'input', 'output']
            ]

            return html.Div(children=[
                html.H1(children='History'),
                dash_table.DataTable(
                    id='history-table',
                    columns=[{"name": i, "id": i, 'presentation': 'markdown'} for i in df.columns],
                    data=df.to_dict('records'),
                ),
            ])

        register_page(
            'History',
            layout=history_layout,
            path='/history',
        )

        def noop_layout(name: str):
            def f():
                return html.Div(children=[
                    html.H1(name),
                    html.P('Not implemented yet'),
                ])
            return f

        register_page(
            'Inbox',
            layout=noop_layout('Inbox'),
            path='/inbox',
        )

        register_page(
            'Chats',
            layout=noop_layout('Chats'),
            path='/chats',
        )

        register_page(
            'Collections',
            layout=noop_layout('Collections'),
            path='/collections',
        )

        def notebook_list_layout():
            response = requests.get(f'{API_URL}/notebooks')
            if response.status_code == 200:
                notebooks = response.json()['response']
            else:
                raise Exception(f'Error getting notebooks: {response.status_code}')
            
            if len(notebooks) == 0:
                return html.Div(children=[
                    html.H1(children='Notebooks'),
                    html.P('No notebooks yet'),
                ])
            
            df = pd.DataFrame(notebooks)

            def generate_notebook_link(id):
                return f'[{id}](/notebooks/{id})'
            
            #df['name'] = df['id'].apply(lambda id: generate_notebook_link(id))
            df = df[
                ['name', 'description']
            ]

            return html.Div(children=[
                html.H1(children='Notebooks'),
                dash_table.DataTable(
                    id='notebooks-table',
                    columns=[{"name": i, "id": i, 'presentation': 'markdown'} for i in df.columns],
                    data=df.to_dict('records'),
                ),
            ])
        
        register_page(
            'Notebooks',
            layout=notebook_list_layout,
            path='/notebooks',
        )

        register_page(
            'Systems',
            layout=noop_layout('Systems'),
            path='/systems',
        )

        menu = [
            'Inbox',
            'Prompts',
            'Chats',
            'Collections',
            'Notebooks',
            'Systems',
            'History',
        ]

        self.app.layout = dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                html.Div(
                                    dcc.Link(
                                        f"{page_registry[name]['name']}", 
                                        href=page_registry[name]["relative_path"],
                                    )
                                ),
                            )
                            for name in menu
                        ], 
                        width=3,
                        style={
                            'border-right': '1px solid black',
                            'background-color': 'lightgray',
                            'height': '100vh',
                            'overflow-y': 'scroll',
                        }
                    ),
                    dbc.Col(
                        [
                            page_container
                        ], 
                        width=9,
                    ),
                ],
            ),
            fluid=True,
        )


class PromptListPage(AdminPage):

    def __init__(self):
        super().__init__(
            name="Prompts",
            path="/prompts",
        )

    def layout(self):
        response = requests.get(f'{API_URL}/prompts')
        if response.status_code == 200:
            prompts = response.json()
        else:
            raise Exception(f'Error getting prompts: {response.status_code}')
        
        if len(prompts['response']) == 0:
            content = [
                html.Div([
                    html.P('No prompts found.'),
                ])
            ]
        else:
            df = pd.DataFrame(prompts['response'])

            def generate_open_link(row):
                return f'[{row["name"]}](/prompts/{row["id"]})'
            
            df['name'] = df.apply(lambda row: generate_open_link(row), axis=1)
            df = df[
                ['name', 'instructions']
            ]
            content = [
                dash_table.DataTable(
                    id='prompts-table',
                    columns=[{"name": i, "id": i, 'presentation': 'markdown'} for i in df.columns],
                    data=df.to_dict('records'),
                ),
            ]

        return html.Div([
            html.H1('Prompts'),
            *content,
        ])