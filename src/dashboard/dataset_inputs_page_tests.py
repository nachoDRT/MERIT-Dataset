import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dashboard_helper as dhelp
from os.path import dirname, abspath
from pathlib import Path
from typing import List, Tuple

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ])

langs_dict = {
    "english": ["eng_school_1", "eng_school_2"],
    "spanish": ["esp_school_1", "esp_school_2"],
    "portuguese": ["por_1", "por_2", "por_3", "por_4", "por_5"],
}


def create_lang_selector(lang: str):
    lang_card = dbc.Card(
        [
            dbc.Checklist(
                options=[{"label": lang.capitalize(), "value": lang}],
                value=[],
                id=f"checkbox-{lang}",
                switch=True,
            ),
            dbc.Collapse(
                dbc.Checklist(
                    options=[
                        {"label": elemento, "value": elemento}
                        for elemento in langs_dict[lang]
                    ],
                    value=[],
                    id=f"checklist-{lang}",
                    inline=True,
                ),
                id=f"collapse-lang-{lang}",
            ),
        ]
    )
    return lang_card


@app.callback(
    [Output(f"collapse-lang-{lang}", "is_open") for lang in langs_dict.keys()],
    [Input(f"checkbox-{lang}", "value") for lang in langs_dict.keys()],
)
def update_collapse(*language_selections):
    updates = [bool(value) for value in language_selections]

    return updates


@app.callback(
    [Output("continue-button", "style"), Output("continue-button", "className")],
    [Input(f"checklist-{lang}", "value") for lang in langs_dict.keys()],
)
def update_collapse(*school_selections):
    selected_schools = [school for language in school_selections for school in language]

    dhelp.update_schools_requirements(selected_schools)

    if len(selected_schools) != 0:
        return {
            "display": "block",
            "justifyContent": "center",
            "marginTop": "20px",
        }, "btn btn-secondary"
    else:
        return {
            "display": "none",
        }, "btn btn-outline-dark"


def generate_lang_card(langs: list, langs_selectors: list):
    row = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(f"Option {i + 1}"),
                        dbc.CardBody(card_content),
                    ]
                )
            )
            for i, (_, card_content) in enumerate(zip(langs, langs_selectors))
        ],
        className="g-10",
    )
    return [row]


def create_continue_button():
    continue_button = dbc.Row(
        [
            dbc.Col(
                html.Button(
                    "Continue",
                    id="continue-button",
                ),
                width=1,
                style={"margin": "0 auto"},
            )
        ]
    )
    return [continue_button]


in_layout = [html.H1("Select Languages and School Templates")]
langs = [lang for lang in langs_dict.keys()]
langs_selectors = [create_lang_selector(lang) for lang in langs_dict.keys()]
row = generate_lang_card(langs, langs_selectors)
in_layout.extend(row)
in_layout.extend(create_continue_button())


layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(width=2),
                dbc.Col(in_layout, width=8, style={"margin": "0 auto"}),
                dbc.Col(width=2),
            ]
        )
    ],
    fluid=True,
)


app.layout = html.Div(layout)

if __name__ == "__main__":
    app.run_server(debug=True)
