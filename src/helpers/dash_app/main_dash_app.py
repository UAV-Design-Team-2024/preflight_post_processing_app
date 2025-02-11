from dash import Dash, html, dcc, Input, Output
from flask import g, current_app
import logging
from enum import StrEnum

logger = logging.getLogger()
logger.setLevel(0)


class dash_aliases(StrEnum):
    main_app = 'main_app'

class dash_data_app():
    def __init__(self, base_url):
        self.dash_app = Dash(server=current_app, url_base_pathname=base_url)

    def create_layout(self):
        test_layout =  html.Div([
            html.Div(children='My First App with Dash & SqlAlchemy'),
            html.Hr(),
            dcc.RadioItems(options=['pop', 'lifeExp', 'gdpPercap'], value='lifeExp', id='controls'),
            dcc.Graph(id='first-graph')
        ])

        return test_layout

    def main_dash_app_init(self):
        # App layout

        self.dash_app.layout = self.create_layout()

        return self.dash_app.server