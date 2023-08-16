import os
import json
from urllib.parse import urljoin
import requests
import pandas as pd
from pydantic import BaseModel
from dash import Dash, html, dcc, dash_table, no_update, page_container, page_registry, register_page, Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from . import World, Collection

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
    
    def render(self):
        return html.Div(children=[
            html.H1(self.name),
        ])
    
    def register_callbacks(self):
        pass


class Index:

    def generate_link(self, row):
        return f'[{row["name"]}](/{self.name}/{row["id"]})'

    def layout(self):
        df['name'] = df.apply(self.generate_link, axis=1)
        content = [
            dash_table.DataTable(
                id='prompts-table',
                columns=[{"name": i, "id": i, 'presentation': 'markdown'} for i in df.columns],
                data=df.to_dict('records'),
                style_as_list_view=True,
            ),
        ]


class AdminIndexPage(AdminPage):

    @property
    def api_url(self):
        return urljoin(API_URL, self.path)

    def generate_link(self, row):
        link = os.path.join(self.path, row['id'])
        return f'[{row["id"]}]({link})'

    def layout(self):
        response = requests.get(self.api_url)
        if response.status_code == 200:
            index = response.json()['response']
        else:
            raise Exception(f'Error getting index {self.name}: {response.status_code}')
        
        if len(index) == 0:
            return html.Div(children=[
                html.H1(self.name),
                html.P('Nothing to see here.'),
            ])
        
        df = pd.DataFrame(index)
        
        df['id'] = df.apply(self.generate_link, axis=1)
        content = [
            dash_table.DataTable(
                id='prompts-table',
                columns=[
                    {"name": i, "id": i, 'presentation': 'markdown'} 
                    for i in df.columns
                ],
                data=df.to_dict('records'),
                style_as_list_view=True,
                style_header={
                    'textAlign': 'left',
                },
                markdown_options={
                    'link_target': '_self',
                },
            ),
        ]

        return html.Div([
            html.H1(self.name),
            *content,
        ])


class AdminEntityPage(AdminPage):

    def fetch(self, pathname):
        api_path = urljoin(API_URL, pathname)
        response = requests.get(api_path)
        if response.status_code == 200:
            data = response.json()
            details = data['details']
            results = data['results']
            print('fetch', data, details, results)
            return data, json.dumps(details), json.dumps(results)
        else:
            raise Exception(f'Error getting entity: {response.status_code}')
    
    def register_callbacks(self):
        @self.app.callback(
            Output('data-store', 'data'),
            Output('details', 'children'),
            Output('results', 'children'),
            Input('fetch-interval', 'n_intervals'),
            Input('url', 'pathname'),
        )
        def fetch_data(n_intervals, pathname):
            print('fetch_data', n_intervals, pathname)
            if n_intervals is not None and n_intervals > 0:
                return self.fetch(pathname)
            else:
                return no_update

    def layout(self, **kwargs):
        content = self.render(**kwargs)
        return html.Div(children=[
            dcc.Location(id='url', refresh=False), 
            dcc.Loading(id='loading', type='default', children=[
                dcc.Store(id='data-store'),
                content,
            ]),
            
            dcc.Interval(
                id='fetch-interval',
                interval=1,  # Set an interval that triggers once
                max_intervals=1  # Only trigger once
            ),
        ])
    
    def render(self, id=None):
        return html.Div([
            html.Div(id='details'),
            html.Div(id='results'),
        ])
        #return html.Div(children=[
        #    html.H1(details['name']),
        #    dbc.Button('Run', id='run-prompt', n_clicks=0, name=id, color='primary'),
        #    dcc.Store(id='api-call-result', storage_type='session'),
        #    html.Div(id, id='result-id', style={'display': 'none'}),
        #    html.H2('Results'),
        #    index,
        #])


class TemplateIndex(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
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
            **kwargs,
        )


class SystemIndex(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Systems",
            path="/systems",
            **kwargs,
        )


class NotebookIndex(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Notebooks",
            path="/notebooks",
            **kwargs,
        )


class ConversationIndex(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Conversations",
            path="/conversations",
            **kwargs,
        )


class Inbox(AdminIndexPage):
    menu: bool = False

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Inbox",
            path="/inbox",
            **kwargs,
        )


class History(AdminIndexPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="History",
            path="/history",
            **kwargs,
        )


class TemplatePage(AdminEntityPage):

    def __init__(self, app, **kwargs):
        super().__init__(
            app,
            name="Template",
            path_template="/templates/<id>",
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
            Inbox(self.app, menu=True),
            ConversationIndex(self.app, menu=True),
            TemplateIndex(self.app, menu=True),
            CollectionIndex(self.app, menu=True),
            NotebookIndex(self.app, menu=True),
            SystemIndex(self.app, menu=True),
            History(self.app, menu=True),

            TemplatePage(self.app),
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
