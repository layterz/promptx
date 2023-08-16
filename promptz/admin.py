from urllib.parse import urljoin
import requests
import pandas as pd
from pydantic import BaseModel
from dash import Dash, html, dcc, dash_table, page_container, page_registry, register_page, Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from . import World, Collection

API_URL = 'http://localhost:8000'


class AdminPage(BaseModel):
    name: str
    path: str = None
    path_template: str = None
    menu: bool = False
    api_path: str = None

    def __init__(self, name, path=None, path_template=None, menu=False):
        super().__init__(
            name=name,
            path=path,
            path_template=path_template,
            menu=menu,
        )

    def layout(self, **kwargs):
        return html.Div(children=[
            html.H1(self.name),
            html.P('Not implemented yet'),
        ])


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
        return f'[{row["id"]}](/{self.name}/{row["id"]})'

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
            ),
        ]

        return html.Div([
            html.H1(self.name),
            *content,
        ])


class AdminEntityPage(AdminPage):
    
    def layout(self):
        api_path = urljoin(API_URL, self.path_template.format(id=123))
        response = requests.get(api_path)
        if response.status_code == 200:
            data = response.json()
            print('data', data)
            details = data['details']
            results = data['results']
        else:
            raise Exception(f'Error getting entity: {response.status_code}')
        
        if len(results) > 0:
            index = Index(results)
        else:
            index = html.Div([
                html.P('No results found.'),
            ])

        return html.Div(children=[
            html.H1(details['name']),
            dbc.Button('Run', id='run-prompt', n_clicks=0, name=id, color='primary'),
            dcc.Store(id='api-call-result', storage_type='session'),
            html.Div(id, id='result-id', style={'display': 'none'}),
            html.H2('Results'),
            index,
        ])


class TemplateIndex(AdminIndexPage):

    def __init__(self, **kwargs):
        super().__init__(
            name="Templates",
            path="/templates",
            **kwargs,
        )


class CollectionIndex(AdminIndexPage):

    def __init__(self, **kwargs):
        super().__init__(
            name="Collections",
            path="/collections",
            **kwargs,
        )


class SystemIndex(AdminIndexPage):

    def __init__(self, **kwargs):
        super().__init__(
            name="Systems",
            path="/systems",
            **kwargs,
        )


class NotebookIndex(AdminIndexPage):

    def __init__(self, **kwargs):
        super().__init__(
            name="Notebooks",
            path="/notebooks",
            **kwargs,
        )


class ConversationIndex(AdminIndexPage):

    def __init__(self, **kwargs):
        super().__init__(
            name="Conversations",
            path="/conversations",
            **kwargs,
        )


class Inbox(AdminIndexPage):
    menu: bool = False

    def __init__(self, **kwargs):
        super().__init__(
            name="Inbox",
            path="/inbox",
            **kwargs,
        )


class History(AdminIndexPage):

    def __init__(self, **kwargs):
        super().__init__(
            name="History",
            path="/history",
            **kwargs,
        )


class TemplatePage(AdminEntityPage):

    def __init__(self, **kwargs):
        super().__init__(
            name="Template",
            path_template="/templates/{id}",
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
            Inbox(menu=True),
            ConversationIndex(menu=True),
            TemplateIndex(menu=True),
            CollectionIndex(menu=True),
            NotebookIndex(menu=True),
            SystemIndex(menu=True),
            History(menu=True),
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
