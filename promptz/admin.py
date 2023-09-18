import os
from urllib.parse import urljoin
import requests
import pandas as pd
from pydantic import BaseModel
from textblob import Word
from dash import Dash, html, dcc, dash_table, no_update, page_container, page_registry, register_page, Output, Input, State
from dash.exceptions import PreventUpdate
from dash.dependencies import ALL
import dash_bootstrap_components as dbc

from promptz.world import World


API_URL = 'http://localhost:8000'


class AdminPage(BaseModel):
    app: Dash
    name: str
    path: str = None
    path_template: str = None
    menu: bool = False
    api_path: str = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, app, name, path=None, path_template=None, menu=False):
        super().__init__(
            app=app,
            name=name,
            path=path,
            path_template=path_template,
            menu=menu,
        )
        self.register_callbacks()

    def layout(self, **kwargs):
        content = self.render(**kwargs)
        return html.Div(children=[
            dcc.Interval(
                id='fetch-interval',
                interval=1,  # Set an interval that triggers once
                max_intervals=1  # Only trigger once
            ),
            content,
        ])
    
    def render(self, **kwargs):
        return html.Div(children=[
            html.H1(self.name),
        ])
    
    def register_callbacks(self):
        pass


class AdminIndexPage(AdminPage):
    collection: str = None

    def __init__(self, app, collection, **kwargs):
        super().__init__(
            app,
            **kwargs,
        )
        self.collection = collection

    @property
    def api_url(self):
        return urljoin(API_URL, self.path)

    def generate_link(self, row):
        link = os.path.join(f'/{self.collection}', row['id'])
        return f'[{row.get("name", row["id"])}]({link})'
    
    def get_columns(self, df):
        return df.columns
    
    def register_callbacks(self):
        @self.app.callback(
            Output(f'{self.name}-details', 'children'),
            Output(f'{self.name}-table', 'children'),
            Input('fetch-interval', 'n_intervals'),
            Input('url', 'pathname'),
        )
        def fetch_data(n_intervals, pathname):
            if n_intervals is not None and n_intervals > 0:
                return self.fetch(pathname)
            else:
                return no_update
    
    def fetch(self, pathname):
        api_path = urljoin(API_URL, pathname)
        response = requests.get(api_path)
        if response.status_code == 200:
            data = response.json()
            details = data.get('details', {})
            index = data.get('list', [])
        else:
            raise Exception(f'Error getting index {self.name}: {response.status_code}')
        
        details = html.H1(details.get('name', self.name))
        if len(index) == 0:
            return details, html.P('Nothing to see here.'),

        df = pd.DataFrame(index)
        df['id'] = df.apply(self.generate_link, axis=1)

        def format_fields(x):
            if isinstance(x, str):
                x = x.split('\n')[0]
                return x[:64] + ' ...'
            else:
                return x
        formatted = df.drop(columns=['id']).applymap(format_fields)
        df = pd.concat([df['id'], formatted], axis=1)
        index_table = dash_table.DataTable(
            id='prompts-table',
            columns=[
                {"name": i, "id": i, 'presentation': 'markdown'} 
                for i in self.get_columns(df)
            ],
            data=df.to_dict('records'),
            style_as_list_view=True,
            style_header={
                'textAlign': 'left',
            },
            markdown_options={
                'link_target': '_self',
            },
        )

        return details, index_table

    def layout(self):
        return html.Div([
            html.Div(id=f'{self.name}-details'),
            html.Div(id=f'{self.name}-table'),
            
            dcc.Location(id='url', refresh=False), 
            dcc.Interval(
                id=f'fetch-interval',
                interval=1,  # Set an interval that triggers once
                max_intervals=1  # Only trigger once
            ),
        ])


class AdminEntityPage(AdminPage):

    def fetch(self, pathname):
        api_path = urljoin(API_URL, pathname)
        response = requests.get(api_path)
        if response.status_code == 200:
            data = response.json()
            print('DATA', data)
            details_data = [
                {'field': k, 'value': v}
                for k, v in data.get('details', {}).items()
                if type(v) in [str, int, float, bool]
            ]
            details_data += [
                {'field': k, 'value': v.get('title')}
                for k, v in data['details'].items()
                if type(v) == dict
            ]
            details = dash_table.DataTable(
                id='details-table',
                columns=[{"name": i, "id": i} for i in ['field', 'value']],
                data=details_data,
                style_as_list_view=True,
            )

            # TODO: this isn't handling "null", which should be parsed into None
            if data['details'].get('input') is None or data['details'].get('input') == 'null':
                inputs = [{
                    'id': 'input',
                    'label': 'Input',
                }]
            else:
                input_schema = {
                    name: field
                    for name, field in data['details']['input']['properties'].items()
                }
                inputs = []
                for name, field in input_schema.items():
                    input = {
                        'id': name,
                        'label': field['title'],
                    }
                    if field['type'] == 'string':
                        input['type'] = 'text'
                    elif field['type'] == 'integer':
                        input['type'] = 'number'
                    elif field['type'] == 'boolean':
                        input['type'] = 'checkbox'
                    else:
                        input['type'] = 'text'
                    
                    inputs.append(input)
            
            form = dbc.Form(
                [
                    *[
                        html.Div([
                            dbc.Label(input['label']),
                            dbc.Input(
                                id={'type': 'json_field', 'index': input['id']},
                                name=input['label'],
                                placeholder=f'Enter {input["label"]}',
                            )
                        ])
                        for input in inputs
                    ],
                    *[
                        html.Div(id=f'{self.name}-{i}', style={ 'display': 'none'})
                        for i in range(len(inputs), 100) 
                    ],
                    html.Div(
                        dbc.Button('Submit', id=f'{self.name}-submit', n_clicks=0, color='primary')
                    ),
                    html.Div(id=f'{self.name}-form-output'),
                ],
                style={
                    'padding': '10px',
                    'background-color': 'lightgray',
                }
            )

            if len(data['results']) == 0:
                results = html.P('Nothing to see here.')
            else:
                df = pd.DataFrame(data['results'])

                def format_fields(x):
                    if isinstance(x, str):
                        x = x.split('\n')[0]
                        return x[:64] + ' ...'
                    else:
                        return x
                
                print('DF', df)
                formatted = df.drop(columns=['id']).applymap(format_fields)
                df = pd.concat([df['id'], formatted], axis=1)

                results = html.Div([
                html.H2('Logs'),
                dash_table.DataTable(
                    id='results-table',
                    columns=[
                        {"name": i, "id": i, 'presentation': 'markdown'} 
                        for i in df.columns
                    ],
                    data=df.to_dict('records'),
                    style_as_list_view=True,
                ),
            ])

            return data, details, form, results
        else:
            raise Exception(f'Error getting entity ({api_path}): {response.status_code}')
    
    def actions(self, pathname):
        pass
    
    def register_callbacks(self):
        @self.app.callback(
            Output(f'{self.name}-data-store', 'data'),
            Output(f'{self.name}-details', 'children'),
            Output(f'{self.name}-form', 'children'),
            Output(f'{self.name}-results', 'children'),
            Input('fetch-interval', 'n_intervals'),
            Input('url', 'pathname'),
        )
        def fetch_data(n_intervals, pathname):
            if n_intervals is not None and n_intervals > 0:
                return self.fetch(pathname)
            else:
                return no_update

        @self.app.callback(
            Output(f'{self.name}-actions', 'children'),
            Input('url', 'pathname'),
        )
        def actions(pathname):
            return self.actions(pathname)
        
        @self.app.callback(
            Output(f'{self.name}-form-output', 'children'),
            [
                Input(f'{self.name}-submit', 'n_clicks'),
                Input('url', 'pathname'),
            ],
            State({'type': 'json_field', 'index': ALL}, 'value'),
            State({'type': 'json_field', 'index': ALL}, 'id')
        )
        def submit_form(n_clicks, pathname, values, ids):
            if n_clicks is None or n_clicks == 0:
                raise PreventUpdate
            
            form_data = {id_dict['index']: value for id_dict, value in zip(ids, values)}
            api_path = urljoin(API_URL, pathname + '/run')
            response = requests.post(api_path, json={'input': form_data})
            if response.status_code == 200:
                data = response.json()
                return data['response'] 
            else:
                return f'Error: {response.status_code}'

    def layout(self, **kwargs):
        return html.Div(children=[
            dcc.Location(id='url', refresh=False), 
            dcc.Loading(id='loading', type='default', children=[
                dcc.Store(id=f'{self.name}-data-store'),
                dbc.Row(
                    id='header',
                    children=[
                            dbc.Col(
                                [
                                    html.H1(self.name),
                                ],
                                width=9,
                                style={
                                },
                            ),
                            dbc.Col(
                                id=f'{self.name}-actions',
                                width=3,
                                children=[],
                                style={
                                },
                            ),
                    ],
                ),
                html.Div(id=f'{self.name}-details'),
                html.Div(id=f'{self.name}-form'),
                html.Div(id=f'{self.name}-results'),
            ]),
            
            dcc.Interval(
                id=f'fetch-interval',
                interval=1,  # Set an interval that triggers once
                max_intervals=1  # Only trigger once
            ),
        ])


class TemplateIndex(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            'templates',
            name="Templates",
            path="/templates",
            **kwargs,
        )


class CollectionIndex(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Collections",
            path="/collections",
            collection='collections',
            **kwargs,
        )


class Inbox(AdminPage):
    menu: bool = False

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Inbox",
            path="/inbox",
            **kwargs,
        )


class Logs(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Logs",
            path="/logs",
            collection='logs',
            **kwargs,
        )


class CollectionPage(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Collection",
            path_template="/collections/<id>",
            **kwargs,
        )
    
    def layout(self, id=None):
        return super().layout()


class DetailsPage(AdminEntityPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Details",
            path_template="/<collection>/<id>",
            **kwargs,
        )


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

        pages = [
            DetailsPage(self.app),
            Inbox(self.app, menu=True),
            TemplateIndex(self.app, menu=True),
            CollectionIndex(self.app, menu=True),
            Logs(self.app, menu=True),
        ]

        for page in pages:
            register_page(page.name, layout=page.layout, path=page.path, path_template=page.path_template)
        
        menu = [page.name for page in pages if page.menu]

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
