import os
from urllib.parse import urljoin
import uuid
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


class Index(BaseModel):
    
    class Config:
        arbitrary_types_allowed = True

    id: str
    app: Dash
    collection: str
    columns: list = None

    def __init__(self, app, collection, columns=None, **kwargs):
        super().__init__(
            id=str(uuid.uuid4()),
            app=app,
            collection=collection,
            columns=columns,
            **kwargs,
        )
        
        @self.app.callback(
            Output(
                f'{self.id}-query-results', 'children',
            ),
            [
                Input(f'{self.id}-search-submit', 'n_clicks'),
            ],
            [
                State(f'{self.id}-search', 'value'),
            ],
        )
        def submit_form(n_clicks, value):
            if n_clicks is None or n_clicks == 0:
                return no_update
            
            return self.fetch(query=value)
        
        @self.app.callback(
            Output(
                f'{self.id}-query-results', 'children',
                allow_duplicate=True,
            ),
            [
                Input(f'fetch-interval', 'n_intervals'),
            ],
            prevent_initial_call=True,
        )
        def load(n_intervals):
            print('load', n_intervals)
            if n_intervals is not None and n_intervals > 0:
                return self.fetch()
            else:
                return no_update
    
    def fetch(self, query=None):
        if self.collection == '':
            return html.P('No collection specified.')
        api_path = urljoin(API_URL, self.pathname)
        response = requests.get(api_path, params={'query': query})
        if response.status_code == 200:
            data = response.json()
            l = data.get('list', [])
        else:
            raise Exception(f'Error getting index {self.collection}: {response.status_code}')
        
        if len(l) == 0:
            return html.P('Nothing to see here.'),

        df = pd.DataFrame(l)
        df['id'] = df.apply(self.generate_link, axis=1)

        if self.columns is not None:
            df = df[self.columns]
        
        table = dbc.Table.from_dataframe(df)
        return table
    
    @property
    def pathname(self):
        return f'/{self.collection}'

    def generate_link(self, row):
        link = os.path.join(f'/{self.collection}', row['id'])
        return html.A(row.get('id'), href=link, target='_self')

    def render(self, **kwargs):
        results = html.Div([], id=f'{self.id}-query-results')

        search = dbc.Input(
            name='search',
            placeholder='Search',
            id=f'{self.id}-search',
        )

        search_input = dbc.Form([
            dbc.Row([
                dbc.Col([
                    search,
                ], width=9),
                dbc.Col([
                    dbc.Button('Submit', id=f'{self.id}-search-submit', n_clicks=0, color='secondary')
                ], width=3)
            ])
        ])

        HEADER_STYLE = {
            'padding': '10px',
            'border-bottom': '1px solid lightgray',
        }

        INDEX_STYLE = {
            'padding': '10px',
            'background-color': 'white',
            'border-radius': '5px',
        }

        return html.Div(children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3(self.collection),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                search_input,
                            ),
                        ],
                        width=9,
                    ),
                ],
                style=HEADER_STYLE,
            ),
            results,
            dcc.Interval(
                id=f'fetch-interval',
                n_intervals=1,  # Set an interval that triggers once
                max_intervals=1  # Only trigger once
            ),
            dcc.Store(id=f'{self.id}-data-store'),
        ], style=INDEX_STYLE)


class AdminPage(BaseModel):
    app: Dash
    name: str
    path: str = None
    path_template: str = None
    menu: bool = False
    icon: str = None
    api_path: str = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, app, name, path=None, path_template=None, menu=False, icon=None):
        super().__init__(
            app=app,
            name=name,
            path=path,
            path_template=path_template,
            menu=menu,
            icon=icon,
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
    index: Index = None
    collection: str = None

    def __init__(self, app, collection, columns=None, **kwargs):
        super().__init__(
            app,
            **kwargs,
        )
        self.collection = collection
        self.index = Index(
            app=self.app, collection=self.collection, columns=columns
        )

    def layout(self):
        content = self.index.render()
        return content


class EntityDetails(BaseModel):
    data: dict

    def render(self):
        table_data = [
            {'field': k, 'value': v}
            for k, v in self.data.items()
            if type(v) in [str, int, float, bool]
        ]
        table_data += [
            {'field': k, 'value': v.get('title')}
            for k, v in self.data.items()
            if type(v) == dict
        ]
        
        table = dbc.Table.from_dataframe(pd.DataFrame(table_data))

        details = html.Div([
            html.H2(self.data.get('name', self.data.get('id'))),
            table,
        ])

        return details


class EntityInputForm(BaseModel):
    data: dict

    def render(self):
        # TODO: this isn't handling "null", which should be parsed into None
        if self.data.get('input') is None or self.data.get('input') == 'null':
            inputs = [{
                'id': 'input',
                'label': 'Input',
            }]
        else:
            input_schema = {
                name: field
                for name, field in self.data['input']['properties'].items()
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
        
        _id = self.data.get('id')
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
                    html.Div(id=f'{_id}-{i}', style={ 'display': 'none'})
                    for i in range(len(inputs), 100) 
                ],
                html.Div(
                    dbc.Button('Submit', id=f'{_id}-submit', n_clicks=0, color='primary')
                ),
                html.Div(id=f'{_id}-form-output'),
            ],
            style={
                'padding': '10px',
                'background-color': 'lightgray',
            }
        )
        return form


class AdminEntityPage(AdminPage):
    results_index: Index = None
    
    def register_callbacks(self):
        @self.app.callback(
            Output(f'{self.name}-data-store', 'data'),
            Output(f'{self.name}-details', 'children'),
            Output(f'{self.name}-input-form', 'children'),
            Output(f'{self.name}-results', 'children'),
            Input('fetch-interval', 'n_intervals'),
            Input('url', 'pathname'),
        )
        def fetch_data(n_intervals, pathname):
            if n_intervals is None or n_intervals == 0:
                return no_update
            api_path = urljoin(API_URL, pathname)
            response = requests.get(api_path)
            if response.status_code == 200:
                data = response.json()
                details = self.details(data)
                input_form = self.input_form(data)
                results = self.results(data)
                return data, details, input_form, results
            else:
                raise Exception(f'Error getting entity ({api_path}): {response.status_code}')

    def layout(self, **kwargs):
        return html.Div(children=[
            dcc.Location(id='url', refresh=False), 
            dcc.Loading(id='loading', type='default', children=[
                dcc.Store(id=f'{self.name}-data-store'),
                html.Div(id=f'{self.name}-details'),
                html.Div(id=f'{self.name}-input-form'),
                html.Div(id=f'{self.name}-results'),
            ]),
            
            dcc.Interval(
                id=f'fetch-interval',
                interval=1,  # Set an interval that triggers once
                max_intervals=1  # Only trigger once
            ),
        ])
    
    def details(self, data):
        details = data.get('details')
        if details is None:
            return None
        return EntityDetails(data=details).render()
    
    def input_form(self, data):
        return None
    
    def results(self, data):
        return None
    

class TemplateIndex(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            'templates',
            name="Templates",
            path="/templates",
            columns=['id', 'name', 'instructions'],
            icon='bi bi-code-slash',
            **kwargs,
        )


class QueryIndex(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Queries",
            path="/queries",
            collection='queries',
            icon='bi bi-search',
            **kwargs,
        )


class AgentIndex(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Agents",
            path="/agents",
            collection='agents',
            icon='bi bi-robot',
            **kwargs,
        )


class SubscriptionIndex(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Subscriptions",
            path="/subscriptions",
            collection='subscriptions',
            icon='bi bi-inbox',
            **kwargs,
        )


class ModelIndex(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Models",
            path="/models",
            collection='models',
            icon='bi bi-infinity',
            **kwargs,
        )


class CollectionIndex(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Collections",
            path="/collections",
            collection='collections',
            icon='bi bi-database',
            **kwargs,
        )


class Inbox(AdminIndexPage):
    menu: bool = False

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Inbox",
            path="/inbox",
            collection='inbox',
            icon='bi bi-envelope',
            **kwargs,
        )


class Logs(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Logs",
            path="/logs",
            collection='logs',
            icon='bi bi-list',
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

    def __init__(self, app, collection, **kwargs):
        super().__init__(
            app,
            name="Details",
            path_template=f"/{collection}/<id>",
            **kwargs,
        )


class TemplateDetailsPage(AdminEntityPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Template Details",
            path_template="/templates/<id>",
            **kwargs,
        )
        self.results_index = Index(
            app=self.app, 
            collection='logs', 
            columns=['id', 'name', 'instructions']
        )
    
    def input_form(self, data):
        return EntityInputForm(data=data.get('details')).render()
    
    def results(self, data):
        return self.results_index.render()


class CollectionDetailsPage(AdminEntityPage):
    results_index: Index = None

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Collection Details",
            path_template="/collections/<id>",
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
            external_stylesheets=[dbc.themes.ZEPHYR, dbc.icons.FONT_AWESOME, dbc.icons.BOOTSTRAP],
        )

        pages = [
            TemplateDetailsPage(self.app),
            CollectionDetailsPage(self.app),

            Inbox(self.app, menu=True),
            QueryIndex(self.app, menu=True),
            SubscriptionIndex(self.app, menu=True),
            TemplateIndex(self.app, menu=True),
            AgentIndex(self.app, menu=True),
            CollectionIndex(self.app, menu=True),
            ModelIndex(self.app, menu=True),
            Logs(self.app, menu=True),
        ]

        for page in pages:
            register_page(page.name, layout=page.layout, path=page.path, path_template=page.path_template)
        
        SIDEBAR_STYLE = {
            "position": "fixed",
            "top": "68px",
            "left": 0,
            "bottom": 0,
            "width": "18rem",
            "padding": "1rem 1rem",
            "background-color": "#fff",
        }

        CONTENT_STYLE = {
            "margin-top": "66px",
            "padding": "1rem 2rem 0 20rem",
            "background-color": "#F5F5F5",
            "width": "100vw",
            "min-height": "100vh",
        }

        menu = [page for page in pages if page.menu]
        nav = dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.I(className=page.icon),
                        '   ',
                        page_registry[page.name]['name'], 
                    ],
                    href=page_registry[page.name]['relative_path'], 
                    active='exact',
                )
                for page in menu
            ],
            vertical=True,
            pills=True,
        )

        self.app.layout = html.Div([
            dbc.Navbar(
                dbc.Container(
                    children=[
                        dbc.Col(
                            dbc.NavbarBrand(
                                "Promptz",
                                href="/",
                                className="ml-2",
                            ),
                            width=3,
                        ),
                        dbc.Col(
                            dbc.Input(
                                type="text",
                                placeholder="Run command >",
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.NavItem(
                                dbc.NavLink(html.I(className="bi bi-person-circle"), href="/")
                            ),
                            width=3,
                            style={
                                'text-align': 'right',
                                'font-size': '1.5em',
                            },
                        )
                    ],
                ),
                color="white",
                dark=False,
                fixed='top',
                style={
                    'border-bottom': '1px solid lightgray'
                }
            ),
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    nav,
                                ],
                                width=3,
                                style=SIDEBAR_STYLE,
                            ),
                            dbc.Col(
                                [
                                    page_container
                                ], 
                                style=CONTENT_STYLE,
                                width=9,
                            ),
                        ],
                    ),
                ],
                fluid=True,
                style={
                },
            ),
        ])
