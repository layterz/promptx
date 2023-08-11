import requests
import pandas as pd
from dash import Dash, html, dcc, dash_table, page_container, page_registry, register_page, callback_context

from . import AdminPage

API_URL = 'http://localhost:8000'


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