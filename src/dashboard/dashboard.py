from dash import html, dcc, Input, Output, ALL, State
import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from os.path import dirname, abspath
from dash.exceptions import PreventUpdate
import dashboard_helper as dhelp
import dash_bootstrap_components as dbc
import json
from typing import List, Dict
import copy
import numpy as np
import scipy.stats as stats
from dashboard_helper import JsonManagement

START_COLOR = "#ccffff"
END_COLOR = "#009933"
BACK_COLOR = "#000000"
TEXT_COLOR = "#FFFFFF"
NUM_MAX_ORIGINS = 3

MIN_GRADE = 0
MAX_GRADE = 10

json_management = JsonManagement()
blueprint_name = "dataset_blueprint.csv"
blueprint_path = os.path.join(dirname(abspath(__file__)), blueprint_name)
df = pd.read_csv(blueprint_path)
css_path = os.path.join(dirname(abspath(__file__)), "assets", "styles.css")
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css",
        dbc.themes.QUARTZ,
        css_path,
    ],
    suppress_callback_exceptions=True,
)

langs_dict = dhelp.get_available_schools_per_language()
origin_langs = dhelp.get_available_origin_langs()


def create_prop_slider(
    team_a: str, team_b: str, step: int = 10, min: int = 0, max: int = 100
):
    title = f"{team_a}/{team_b} Proportion"
    mean = (max - min) / 2
    slider_id = f"slider-{team_a}-{team_b}"
    output_id = f"output-{team_a}-{team_b}"

    return html.Div(
        [
            html.H3(title, style={"marginTop": "1.5em"}),
            html.Div(
                [
                    dcc.Slider(
                        min,
                        max,
                        step,
                        value=mean,
                        id={"type": "slider", "index": slider_id},
                        className="custom-slider",
                    ),
                    html.Div(
                        id={"type": "output", "index": output_id},
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "fontSize": 12,
                            "marginTop": "0.5em",
                        },
                    ),
                ],
                style={"marginTop": "25px"},
            ),
        ],
        style={"marginTop": "20px"},
    )


def create_gender_bias_card(gender: str):
    card = (
        dbc.Card(
            [
                dbc.CardHeader(f"{gender.capitalize()} Input"),
                dbc.CardBody(
                    [
                        dcc.Slider(
                            id=f"{gender[0]}-average-slider",
                            min=0,
                            max=10,
                            step=1,
                            value=5,
                        ),
                        dcc.Slider(
                            id=f"{gender[0]}-deviation-slider",
                            min=0.01,
                            max=5,
                            step=None,
                            value=2.5,
                        ),
                    ]
                ),
            ],
        ),
    )
    return card


def create_gender_bias_col_left():
    female_card = create_gender_bias_card("female")
    male_card = create_gender_bias_card("male")
    gender_bias_first_col_layout = (
        dbc.Col(
            [
                dbc.Row(female_card),
                dbc.Row(
                    male_card,
                    style={"marginTop": "25px"},
                ),
            ],
            width=4,
        ),
    )

    return gender_bias_first_col_layout


def create_gender_bias_col_right():
    gender_bias_second_col_layout = (
        dbc.Col(
            [
                dcc.Graph(id="distribution-plot"),
                html.Div(
                    [
                        dcc.Markdown(id="clipping-warning"),
                    ]
                ),
            ],
            width=8,
        ),
    )
    return gender_bias_second_col_layout


def create_gender_bias_layout():
    title = "Bias Selector"
    gender_bias_col_left = create_gender_bias_col_left()
    gender_bias_col_right = create_gender_bias_col_right()
    bias_selector = html.Div(
        [
            html.H3(title, style={"marginTop": "1.5em"}),
            dbc.Row([gender_bias_col_left[0], gender_bias_col_right[0]]),
        ]
    )

    return [bias_selector]


def check_ranges(data, min_value, max_value):
    exceeded_lower_limit = data < min_value
    exceeded_upper_limit = data > max_value

    data[exceeded_lower_limit] = min_value
    data[exceeded_upper_limit] = max_value

    return data, exceeded_lower_limit, exceeded_upper_limit


@app.callback(
    Output("distribution-plot", "figure"),
    Output("clipping-warning", "children"),
    Input("f-average-slider", "value"),
    Input("f-deviation-slider", "value"),
    Input("m-average-slider", "value"),
    Input("m-deviation-slider", "value"),
)
def update_distribution_plot(f_average, f_deviation, m_average, m_deviation):
    fig = go.Figure()

    dhelp.update_fe_male_bias_distributions(
        json_management, f_average, f_deviation, m_average, m_deviation
    )

    f_data = np.random.normal(f_average, f_deviation, 2000)
    m_data = np.random.normal(m_average, m_deviation, 2000)

    f_data, f_lower_limit, f_upper_limit = check_ranges(f_data, MIN_GRADE, MAX_GRADE)
    m_data, m_lower_limit, m_upper_limit = check_ranges(m_data, MIN_GRADE, MAX_GRADE)

    f_hist, f_bin_edges = np.histogram(f_data, bins=np.arange(MIN_GRADE, MAX_GRADE + 1))
    m_hist, m_bin_edges = np.histogram(m_data, bins=np.arange(MIN_GRADE, MAX_GRADE + 1))

    fig.add_trace(
        go.Bar(
            x=f_bin_edges[:-1],
            y=f_hist,
            width=0.35,
            name="Female",
            marker=dict(
                color="rgba(255, 255, 255, 0.25)",
                line=dict(color="rgb(102, 16, 242)", width=1.15),
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=m_bin_edges[:-1],
            y=m_hist,
            width=0.35,
            name="Male",
            marker=dict(
                color="rgba(255, 255, 255, 0.25)",
                line=dict(color="rgb(65, 215, 167)", width=1.15),
            ),
        )
    )

    # Compute distribution
    x_values = np.linspace(MIN_GRADE, MAX_GRADE, 200)
    f_pdf_values = stats.norm.pdf(x_values, f_average, f_deviation)
    m_pdf_values = stats.norm.pdf(x_values, m_average, m_deviation)

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=f_pdf_values,
            mode="lines",
            name="Female Prob. Dist.",
            yaxis="y2",
            line=dict(color="rgb(102, 16, 242)"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=m_pdf_values,
            mode="lines",
            name="Male Prob. Dist.",
            yaxis="y2",
            line=dict(color="rgb(65, 215, 167)"),
        )
    )

    pdf_values = np.concatenate((f_pdf_values, m_pdf_values))
    # Configure axes
    fig.update_layout(
        xaxis_title="Grades",
        xaxis=dict(
            range=[MIN_GRADE - 0.5, MAX_GRADE + 0.5],
            showgrid=True,
            titlefont=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="rgba(255, 255, 255, 0.25)",
        ),
        yaxis=dict(
            title="Number of Students",
            side="left",
            titlefont=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="rgba(255, 255, 255, 0.25)",
        ),
        yaxis2=dict(
            title="Probability",
            overlaying="y",
            side="right",
            range=[0, max(pdf_values) + 0.5],
            showgrid=False,
            titlefont=dict(color="white"),
            tickfont=dict(color="white"),
        ),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        legend=dict(
            font=dict(color="white"),
            orientation="h",
            x=0.5,
            y=1.1,
            xanchor="center",
            yanchor="bottom",
        ),
    )

    # Create annotations for the bins
    bin_labels = [
        f"[{int(f_bin_edges[i])} - {int(f_bin_edges[i+1])})"
        if i != len(f_bin_edges) - 2
        else f"[{int(f_bin_edges[i])} - {int(f_bin_edges[i+1])}]"
        for i in range(len(f_bin_edges) - 1)
    ]

    for i, (f_value, m_value) in enumerate(zip(f_hist, m_hist)):
        if f_value >= m_value:
            value = f_value
        else:
            value = m_value

        fig.add_annotation(
            x=f_bin_edges[i],
            y=value,
            text=bin_labels[i],
            showarrow=False,
            yshift=10,
            font=dict(color="white"),
        )

    if (
        (any(f_lower_limit) and any(f_upper_limit))
        or (any(m_lower_limit) and any(m_upper_limit))
        or (any(m_lower_limit) and any(f_upper_limit))
        or (any(f_lower_limit) and any(m_upper_limit))
    ):
        return (
            fig,
            (
                f"Notice that grades lower than {MIN_GRADE} are computed as {MIN_GRADE}"
                + f" and grades greater than {MAX_GRADE} are computed as {MAX_GRADE}"
            ),
        )
    elif any(f_lower_limit) or any(m_lower_limit):
        return (
            fig,
            f"Notice that grades lower than {MIN_GRADE} are computed as {MIN_GRADE}",
        )
    elif any(f_upper_limit) or any(m_upper_limit):
        return (
            fig,
            f"Notice that grades greater than {MAX_GRADE} are computed as {MAX_GRADE}",
        )
    else:
        return fig, ""


@app.callback(
    Output({"type": "output", "index": ALL}, "children"),
    [Input({"type": "slider", "index": ALL}, "value")],
    [State({"type": "slider", "index": ALL}, "id")],
)
def update_slider_output(values, ids):
    if not dash.callback_context.triggered:
        raise PreventUpdate

    value = values[0]
    id = ids[0]

    slider_id = id["index"]
    _, team_a, team_b = slider_id.split("-")
    response = f"Selection: {value}% {team_a}, {100 - value}% {team_b}"

    dhelp.update_fe_male_proportion_requirements(json_management, value)

    return [response]


def generate_carousel_item(language: str):
    carousel_dict = {}

    if language == "no_lang":
        header = "No Language Options"
        caption = "Select a school"
    else:
        header = language.capitalize()
        caption = f"Explore options for {language}"

    carousel_dict["key"] = language
    carousel_dict["src"] = "".join(["/assets/", language, ".svg"])
    carousel_dict["header"] = header
    carousel_dict["caption"] = caption

    return carousel_dict


def create_lang_options_carousel():
    carousel_items = [generate_carousel_item("no_lang")]
    carousel = dbc.Row(
        dbc.Col(
            [
                dbc.Carousel(
                    items=carousel_items,
                    style={
                        "marginTop": "1em",
                        "maxWidth": "300px",
                        "height": "150px",
                        "margin": "auto",
                        "marginBottom": "10em",
                    },
                    id="lang-options-carousel",
                ),
                html.Div(
                    id="hidden-div-carousel",
                    style={"display": "none"},
                    children=json.dumps(carousel_items),
                ),
            ]
        )
    )
    return [carousel]


def create_student_origin_x():
    origin_layout = [html.H3("Select Students' Features", style={"marginTop": "1.5em"})]
    row = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(html.Div(id="origin-selection-card-title")),
                        dbc.CardBody(
                            html.Div(id="origin-selection-card-content"),
                        ),
                    ],
                    id="origin-selection-card",
                )
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader(html.Div(id="origin-proportion-card-title")),
                            dbc.CardBody(html.Div(id="origin-proportion-card-content")),
                        ],
                        id="origin-proportion-card",
                    ),
                    dbc.Card(
                        [
                            dbc.CardHeader(f"Option C"),
                            dbc.CardBody("HELLO"),
                        ],
                        id="origin-marks-bias-card",
                        style={"marginTop": "25px"},
                    ),
                ]
            ),
            dcc.Store(id="checklist-selections-store", storage_type="memory"),
        ]
    )
    origin_layout.extend([row])
    return origin_layout


@app.callback(
    [
        Output("origin-proportion-card-title", "children"),
        Output("origin-proportion-card-content", "children"),
    ],
    [
        Input("lang-options-carousel", "active_index"),
        Input("hidden-div-carousel", "children"),
        Input({"type": "checklist-origin", "index": ALL}, "value"),
    ],
    [State("checklist-selections-store", "data")],
)
def create_origin_proportion_card(*args):
    ctx = dash.callback_context
    triggered_input_id = ctx.triggered[0]["prop_id"].split(".")[0]

    active_index, json_data, _selections, stored_selections = args

    if isinstance(json_data, str):
        json_data = eval(json_data)
    active_index = remap_active_index(active_index, json_data)

    if json_data[0]["key"] == "no_lang":
        card_title = "No language Option"
        card_body = create_origin_prop_slider(None, "", True)

        return card_title, card_body

    # Get the card title (based on the carousel)
    language = json_data[active_index]["header"]
    card_title = f"Select Origin Proportions in {language}"

    if stored_selections == None:
        if (
            triggered_input_id == "lang-options-carousel"
            or triggered_input_id == "hidden-div-carousel"
        ):
            card_body = create_origin_prop_slider(
                stored_selections, language, True, change=True
            )
        else:
            card_body = create_origin_prop_slider(stored_selections, language, True)
    else:
        if (
            triggered_input_id == "lang-options-carousel"
            or triggered_input_id == "hidden-div-carousel"
        ):
            card_body = create_origin_prop_slider(
                stored_selections, language, change=True
            )
        else:
            card_body = create_origin_prop_slider(stored_selections, language)

    return card_title, card_body


def create_origin_prop_slider(
    stored: None or Dict,
    lang: str,
    disabled: bool = False,
    values: List = None,
    change: bool = False,
) -> html.Div:
    if stored == None and not disabled:
        return html.Div()

    else:
        if not disabled and not values:
            values = []
            intermediate_values = [
                round((100 * (i + 1) / (len(stored[lang]) + 1)), 0)
                for i, _ in enumerate(stored[lang])
            ]
            values.extend(intermediate_values)
        elif not values:
            values = []

        prov_values = dhelp.get_ethnic_origins_proportions(json_management, lang)

        if change:
            values = []
            accumulated = 0
            for i, value in enumerate(prov_values):
                if i + 1 == len(prov_values):
                    break

                accumulated += value
                values.append(accumulated)

        return html.Div(
            [
                dcc.RangeSlider(
                    id="origin-proportion-slider",
                    marks={0: "0%", 100: "100%"},
                    min=0,
                    max=100,
                    step=0.01,
                    value=values,
                    pushable=10,
                    disabled=disabled,
                    className="custom-slider",
                ),
                html.Div(
                    [
                        dcc.Markdown(id="slider-output"),
                    ],
                    style={"marginTop": "2em"},
                ),
            ]
        )


@app.callback(
    Output("slider-output", "children"),
    Output("origin-proportion-slider", "value"),
    [
        Input("lang-options-carousel", "active_index"),
        Input("hidden-div-carousel", "children"),
        Input("origin-proportion-slider", "value"),
    ],
    [State("checklist-selections-store", "data")],
)
def update_proportion_slider(active_index, json_data, values: List, stored_selections):
    ctx = dash.callback_context
    triggered_input_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if isinstance(json_data, str):
        json_data = eval(json_data)
        active_index = remap_active_index(active_index, json_data)

    language = json_data[active_index]["header"]

    if triggered_input_id == "lang-options-carousel":
        proportions = dhelp.get_ethnic_origins_proportions(
            json_management, lang=language
        )
        values = dhelp.props_2_values(proportions)

    try:
        if stored_selections == None:
            selection_message = "Select any Language and School to unlock"
            values = [0]

            return selection_message, values

        elif len(stored_selections[language]) == 0:
            selection_message = f"100% of {language} names"
            values = [100]
            dhelp.update_ethnic_origins_in_requirements(
                json_management, values, stored_selections, language
            )
            return selection_message, values

        else:
            main_lang_prop = values[0]
            main_lang_prop_corrected = check_slider_limits(main_lang_prop)
            if main_lang_prop != main_lang_prop_corrected:
                values[0] = main_lang_prop_corrected
            message_fragment = f"{language}: {main_lang_prop_corrected:.0f}%   \n"

            props_fragments = [main_lang_prop_corrected]
            for i, lang_prop in enumerate(values):
                lang_prop_corrected = check_slider_limits(lang_prop)
                if lang_prop != lang_prop_corrected:
                    values[i] = lang_prop_corrected

                try:
                    next_value = values[i + 1]
                    next_value_corrected = check_slider_limits(next_value)
                except IndexError:
                    next_value_corrected = 100

                proportion = next_value_corrected - lang_prop_corrected
                props_fragments.append(proportion)
                try:
                    message_fragment += (
                        f"{stored_selections[language][i].capitalize()}: "
                        f"{proportion:.0f}%   \n"
                    )
                except IndexError:
                    message_fragment = "Loading"
            dhelp.update_ethnic_origins_in_requirements(
                json_management, props_fragments, stored_selections, language
            )
            return message_fragment, values

    except KeyError:
        return "Loading", [0]


def check_slider_limits(value: float):
    min_limit = 10
    max_limit = 90

    if value < min_limit:
        value = min_limit
    if value > max_limit:
        value = max_limit

    return value


def remap_active_index(active_index: int or None, json_data: list):
    if active_index == None:
        active_index = 0
    elif active_index >= len(json_data):
        active_index -= 1

    return active_index


def update_stored_selections(stored_selections: Dict, json_data: List):
    # Include every language active in the carousel in the stored_selections
    stored_selections_languages = [
        language.capitalize() for language in stored_selections.keys()
    ]

    for i, _ in enumerate(json_data):
        if json_data[i]["key"].capitalize() not in stored_selections_languages:
            stored_selections[json_data[i]["key"].capitalize()] = []

    # Also remove those that are not in the carousel
    stored_selections_helper = {}
    carousel_languages = [data["key"].capitalize() for data in json_data]

    for i, language in enumerate(stored_selections):
        if language in carousel_languages:
            stored_selections_helper[language] = stored_selections[language]
    stored_selections = stored_selections_helper

    return stored_selections


@app.callback(
    [
        Output("origin-selection-card-title", "children"),
        Output("origin-selection-card-content", "children"),
        Output("checklist-selections-store", "data"),
        Output({"type": "checklist-origin", "index": ALL}, "value"),
    ],
    [
        Input("lang-options-carousel", "active_index"),
        Input("hidden-div-carousel", "children"),
        Input({"type": "checklist-origin", "index": ALL}, "value"),
    ],
    [State("checklist-selections-store", "data")],
)
def update_language_origin_selector(*args):
    # Check what input triggers the callback
    ctx = dash.callback_context
    triggered_input_id = ctx.triggered[0]["prop_id"].split(".")[0]

    active_index, json_data, selections, stored_selections = args

    # Inputs due to (upper) Language Selector and Language Carousel
    if (
        triggered_input_id == "lang-options-carousel"
        or triggered_input_id == "hidden-div-carousel"
    ):
        if len(selections) == 1:
            selections = [[]]
        else:
            selections = []
        checklist_origin_flag = False

    # Inputs due to the checklist
    else:
        checklist_origin_flag = True

    # Logic beginning
    if isinstance(json_data, str):
        json_data = eval(json_data)

    if json_data[0]["key"] == "no_lang":
        card_title = "No language Option"
        card_body = "Select any Language and School to get a list of names origins"

        return card_title, card_body, stored_selections, selections

    # Get the index of the element in the carousel
    active_index = remap_active_index(active_index, json_data)

    # Make sure every language active in the carousel is in the stored_selections
    if stored_selections != None:
        stored_selections = update_stored_selections(stored_selections, json_data)

    # Get the card title (based on the carousel)
    language = json_data[active_index]["header"]
    card_title = f"Names Ethnic-Origins in {language}"

    if stored_selections is None:
        stored_selections = {}

    if language in stored_selections:
        try:
            stored_selections[language] += selections[0]

            # Limit the max number of origins
            if len(stored_selections[language]) > NUM_MAX_ORIGINS:
                stored_selections[language] = stored_selections[language][
                    -NUM_MAX_ORIGINS:
                ]
                selections[0] = stored_selections[language]

            if checklist_origin_flag:
                stored_selections[language] = selections[0]
            else:
                selections[0] = stored_selections[language]

        except:
            stored_selections[language] = selections

        card_body = create_lang_origin_checklist(
            lang=language, selected_values=stored_selections[language]
        )

    else:
        card_body = create_lang_origin_checklist(lang=language, selected_values=[])
        if len(selections) == 1:
            stored_selections[language] = selections[0]
        else:
            stored_selections[language] = selections

    # Remove duplicates
    stored_selections[language] = list(dict.fromkeys(stored_selections[language]))

    return card_title, card_body, stored_selections, selections


def create_lang_origin_checklist(lang: str, selected_values: List):
    lang = lang.lower()
    adapted_origin_langs = copy.deepcopy(origin_langs)
    adapted_origin_langs.remove(lang)
    origin_checklist = (
        html.Div(
            [
                html.H6(f"Choose {NUM_MAX_ORIGINS} max."),
                dbc.Checklist(
                    options=[
                        {"label": element.capitalize(), "value": element}
                        for element in adapted_origin_langs
                    ],
                    value=selected_values,
                    id={"type": "checklist-origin", "index": lang},
                    inline=False,
                ),
            ]
        ),
    )

    return origin_checklist


def create_contact_info():
    url_github = "https://github.com/nachoDRT"
    github_link = html.Div(
        [
            html.A(
                [
                    html.Div([html.I(className="bi bi-github"), " nachoDRT"]),
                ],
                href=url_github,
                target="_blank",
            )
        ]
    )

    contact = dbc.Row(
        [
            dbc.Col(
                github_link,
                width=1,
                style={"margin": "auto", "marginTop": "10em", "marginBottom": "4em"},
            )
        ]
    )

    return contact


def create_num_students_selector_card():
    selector = dbc.Card(
        [
            dbc.CardHeader(
                html.Div("No Schools Selected"),
                id="num-students-selector-card-title",
            ),
            dbc.CardBody(
                html.Div("Select any School"),
                id="num-students-selector-card-content",
            ),
        ]
    )
    return selector


def create_school_num_students_selector(school: str, prev_num: int = 0):
    return dbc.Row(
        [
            dbc.Col(html.Div(school.capitalize()), width=6, align="center"),
            dbc.Col(
                dbc.Input(
                    type="number",
                    min=0,
                    max=10000,
                    step=100,
                    value=prev_num,
                    style={"width": "120px"},
                    id={"type": "num-students-selection", "index": school},
                ),
                width=5,
                align="center",
            ),
        ],
        style={"marginTop": "0.5em"},
    )


@app.callback(
    [
        Output("num-students-selector-card-title", "children"),
        Output("num-students-selector-card-content", "children"),
    ],
    [Input(f"checklist-{lang}", "value") for lang in langs_dict.keys()]
    + [Input(f"collapse-lang-{lang}", "is_open") for lang in langs_dict.keys()],
    [State({"type": "num-students-selection", "index": ALL}, "value")],
)
def update_num_students_selector(*args):
    args = list(args)
    state = args.pop(-1)

    """ Select only the relevant argument since the "collapse-lang-{lang}" input is just 
    to trigger the callbak"""
    args = args[: -len(langs_dict.keys())]

    # Recover info from previous selection in the Language/School selector
    previous_selections_flag = any(state)

    mapped_data = {}

    if previous_selections_flag:
        index = 0
        for lang_schools in args:
            for school in lang_schools:
                try:
                    mapped_data[school] = state[index]
                    index += 1
                except IndexError:
                    pass

    if any(args):
        numerical_input = []

        for lang_schools in args:
            for school in lang_schools:
                if school in mapped_data:
                    prev_num = mapped_data[school]
                    numerical_input.append(
                        create_school_num_students_selector(school, prev_num)
                    )
                else:
                    numerical_input.append(create_school_num_students_selector(school))

        title = "Number of Students"
        content = html.Div(numerical_input)

    else:
        title = "No Schools Selected"
        content = "Select any School"

    return title, content


def create_num_students_plot(hist_values: List = None, hist_classes: List = None):
    fig = go.Figure()

    if not hist_classes:
        hist_classes = [
            school.capitalize()
            for lang_schools in langs_dict.values()
            for school in lang_schools
        ]

    if not hist_values:
        hist_values = [0 for _ in hist_classes]

    fig.add_trace(
        go.Bar(
            x=hist_classes,
            y=hist_values,
            width=0.35,
            marker=dict(
                color="rgba(255, 255, 255, 0.25)",
                line=dict(color="rgb(255, 255, 255)", width=1.15),
            ),
        )
    )

    # Configure axes
    fig.update_layout(
        xaxis_title="Schools",
        xaxis=dict(
            showgrid=True,
            titlefont=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="rgba(255, 255, 255, 0.25)",
            tickangle=-45,
        ),
        yaxis=dict(
            title="Number of Students",
            side="left",
            titlefont=dict(color="white"),
            tickfont=dict(color="white"),
            range=[0, 10000],
            gridcolor="rgba(255, 255, 255, 0.25)",
        ),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        legend=dict(
            font=dict(color="white"),
            orientation="h",
            x=0.5,
            y=1.1,
            xanchor="center",
            yanchor="bottom",
        ),
    )

    return dcc.Graph(figure=fig)


@app.callback(
    [Output("num-students-plot-col", "children")],
    [Input(f"checklist-{lang}", "value") for lang in langs_dict.keys()]
    + [Input({"type": "num-students-selection", "index": ALL}, "value")],
)
def update_num_students_plot(*args):
    num_students_school = args[-1]
    language_schools = args[: len(args) - 1]

    hist_classes = [
        school.capitalize() for language in language_schools for school in language
    ]

    hist_values = [
        num_students if num_students != None else 0
        for num_students in num_students_school
    ]

    total_schools = [
        school for lang_schools in langs_dict.values() for school in lang_schools
    ]

    dhelp.update_num_students_per_school(json_management, hist_values, hist_classes)

    return [create_num_students_plot(hist_values, hist_classes)]


def generate_num_students_layout():
    num_students_selector = html.Div(
        [
            html.H3("Number of students selector", style={"marginTop": "1.5em"}),
            dbc.Row(
                [
                    dbc.Col(create_num_students_selector_card(), width=4),
                    dbc.Col(
                        create_num_students_plot(), width=8, id="num-students-plot-col"
                    ),
                ],
            ),
        ]
    )
    return [num_students_selector]


# Define the layout for the dataset input page
def layout_input_dataset() -> dash.Dash.layout:
    in_layout = [
        html.Div(
            [
                html.H1(
                    "Dataset Inputs",
                    style={
                        "marginTop": "0.5em",
                        "marginBottom": "0.05em",
                        "fontWeight": "bold",
                    },
                ),
                html.P("Design your own dasaset", style={"marginTop": "0.05em"}),
            ]
        )
    ]
    in_layout.extend(
        [html.H3("Select Languages and School Templates", style={"marginTop": "1.5em"})]
    )
    langs = [lang for lang in langs_dict.keys()]
    langs_selectors = [create_lang_selector(lang) for lang in langs_dict.keys()]
    in_layout.extend(generate_lang_card(langs, langs_selectors))
    in_layout.extend(generate_num_students_layout())
    in_layout.extend([create_prop_slider(team_a="Female", team_b="Male")])
    in_layout.extend(create_gender_bias_layout())
    in_layout.extend(create_lang_options_carousel())
    in_layout.extend(create_student_origin_x())

    layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(width=2),
                    dbc.Col(in_layout, width=8, style={"margin": "0 auto"}),
                    dbc.Col(width=2),
                ]
            ),
            create_continue_button(),
            create_contact_info(),
        ],
        fluid=True,
    )
    return html.Div(layout)


def create_lang_selector(lang: str):
    lang_card = html.Div(
        [
            dbc.Checklist(
                options=[{"label": lang.capitalize(), "value": lang}],
                value=[],
                id=f"checkbox-{lang}",
                switch=True,
                className="custom-switch",
            ),
            dbc.Collapse(
                dbc.Checklist(
                    options=[
                        {"label": element.capitalize(), "value": element}
                        for element in langs_dict[lang]
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
    [Output(f"checklist-{lang}", "value") for lang in langs_dict.keys()],
    [Input(f"collapse-lang-{lang}", "is_open") for lang in langs_dict.keys()]
    + [Input(f"checklist-{lang}", "value") for lang in langs_dict.keys()],
)
def update_checklist(*args):
    num_langs = len(langs_dict.keys())
    callapse_open_values = args[:num_langs]
    checklist_values = args[num_langs:]
    checklist_values_list = list(checklist_values)

    for i, collapse_open_value in enumerate(callapse_open_values):
        if collapse_open_value == False:
            checklist_values_list[i] = []

    return tuple(checklist_values_list)


@app.callback(
    [
        Output("continue-button", "style"),
        Output("continue-button", "className"),
        Output("lang-options-carousel", "items"),
        Output("hidden-div-carousel", "children"),
    ],
    [Input(f"checklist-{lang}", "value") for lang in langs_dict.keys()],
)
def update_collapse(*school_selections):
    selected_schools = [school for language in school_selections for school in language]

    dhelp.update_schools_requirements(json_management, selected_schools)
    selected_languages = dhelp.get_langs_with_replicas(json_management)
    if len(selected_languages) == 0:
        selected_languages.append("no_lang")
    carousel_items = [generate_carousel_item(str(i)) for i in selected_languages]

    if len(selected_schools) != 0:
        return (
            {
                "display": "block",
                "justifyContent": "center",
                "marginTop": "20px",
            },
            "btn btn-secondary",
            carousel_items,
            json.dumps(carousel_items),
        )
    else:
        return (
            {
                "display": "none",
            },
            "btn btn-outline-dark",
            carousel_items,
            json.dumps(carousel_items),
        )


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
                    "Run Generator",
                    id="continue-button",
                ),
                width=1,
                style={"margin": "auto", "marginTop": "10em", "marginBottom": "4em"},
            )
        ]
    )
    return continue_button


@app.callback(
    Output("url", "pathname"),
    Input("continue-button", "n_clicks"),
    prevent_initial_call=True,
)
def navigate_to_statistics(n_clicks):
    if n_clicks:
        return "/statistics"
    return dash.no_update


def layout_statistics_dataset() -> dash.Dash.layout:
    """
    Configures the layout of the Dash app and sets up callback functions for interactive
    elements.

    This method takes a Dash app instance and a DataFrame as arguments. It sets up the
    layout of the app, defining the structure and contents of the dashboard. This inclu-
    des creating a bar plot for language distribution, a dropdown menu for language se-
    lection, placeholders for a school bar plot, and two gauge plots for replication and
    modification statuses. The method also defines three callback functions within the
    app to update the school bar plot, and the replication and modification gauges in
    response to changes in the selected language from the dropdown menu.

    Args:
        app (dash.Dash): The Dash app instance.
        df (pandas.DataFrame): The DataFrame with the data to display on the dashboard.

    Returns:
        dash.Dash.layout: The app page layout.
    """

    languages_bar_plot_title = "Samples Distribution per Language"
    capitalized_languages = [lang.capitalize() for lang in df["language"].unique()]

    object = html.Div(
        [
            html.Div(
                [
                    html.H1(
                        "Data Visualization Page",
                    ),
                    dcc.Link(
                        "Go Back",
                        href="/",
                        style={
                            "backgroundColor": BACK_COLOR,
                            "color": TEXT_COLOR,
                            "textDecoration": "none",
                            "fontFamily": "Arial",
                        },
                    ),
                ],
                style={
                    "backgroundColor": BACK_COLOR,
                    "color": TEXT_COLOR,
                    "fontFamily": "Arial",
                },
            ),
            # First Bar Plot
            dcc.Graph(
                id="language-bar-plot",
                figure=config_fig(
                    df,
                    "language",
                    languages_bar_plot_title,
                    capitalized_languages,
                    "Language",
                ),
            ),
            # Language Selector Dropdown
            dcc.Dropdown(
                id="language-selector",
                options=[
                    {"label": i.title(), "value": i} for i in df["language"].unique()
                ],
                value=df["language"].unique()[0],
                style={
                    "fontFamily": "Arial",
                    "fontSize": "16px",
                    "backgroundColor": BACK_COLOR,
                    "fontColor": START_COLOR,
                },
            ),
            html.Div(
                [
                    # School Bar Plot Placeholder
                    dcc.Graph(
                        id="school-bar-plot",
                        style={
                            "display": "inline-block",
                            "width": "33.33%",
                        },
                    ),
                    # Replication Gauge Placeholder
                    dcc.Graph(
                        id="replication-gauge",
                        style={
                            "display": "inline-block",
                            "width": "33.33%",
                        },
                    ),
                    # Modification Gauge Placeholder
                    dcc.Graph(
                        id="modification-gauge",
                        style={
                            "display": "inline-block",
                            "width": "33.33%",
                        },
                    ),
                ],
            ),
        ],
    )

    return object


@app.callback(
    dash.dependencies.Output("school-bar-plot", "figure"),
    [dash.dependencies.Input("language-selector", "value")],
    [dash.dependencies.Input("interval-component", "n_intervals")],
)
def update_school_plot(selected_language: str, _: int = None) -> go.Figure:
    """
    Updates the school bar plot based on the selected language.

    This method is decorated with `@app.callback`, making it a callback function
    within the Dash app. It gets triggered whenever the dropdown changes.

    The method first filters the dataset to include only the samples corresponding
    to the selected language. It retrieves the unique school names from the filtered
    dataset and capitalizes each school name. A title string for the bar plot is
    constructed to include the selected language. The 'config_fig' method is then
    called with the filtered dataset, the key for the school name column, the title
    string, the list of capitalized school names, and the x-axis title "School" to
    generate a new bar plot figure, which is then returned.

    Args:
        selected_language (str): The currently selected language in dropdown.

    Returns:
        go.Figure: The updated bar plot figure showing the number of samples per
                    school for the selected language.

    """
    blueprint_name = "dataset_blueprint.csv"
    blueprint_path = os.path.join(dirname(abspath(__file__)), blueprint_name)
    df = pd.read_csv(blueprint_path)
    filtered_df = df[df["language"] == selected_language]
    school_names = [
        school.capitalize() for school in filtered_df["school_name"].unique()
    ]
    schools_bar_plot_title = f"Samples per School in {selected_language.capitalize()}"
    figure = config_fig(
        filtered_df,
        "school_name",
        schools_bar_plot_title,
        school_names,
        "School",
    )
    return figure


@app.callback(
    dash.dependencies.Output("replication-gauge", "figure"),
    [dash.dependencies.Input("language-selector", "value")],
    [dash.dependencies.Input("interval-component", "n_intervals")],
)
def update_replicas_gauge(selected_language: str, _: int = None) -> go.Figure:
    """
    Updates the replication gauge figure based on the selected language.

    This method is decorated with '@app.callback', making it a callback function
    within the Dash app. It gets triggered whenever the value of the language
    selector dropdown changes.

    The method first constructs a title string for the gauge that includes the se-
    lected language. It then filters the dataset to include only the samples corres-
    ponding to the selected language, and computes the fraction of these samples for
    which the replication has been completed (i.e., the "replication_done" column is
    True). This fraction is passed to 'update_gauge' along with the title string to
    generate a new gauge figure, which is then returned.

    Args:
        selected_language (str): The currently selected language in the dropdown.

    Returns:
        go.Figure: The updated gauge figure showing the percentage of replications
                    completed for the selected language.
    """
    blueprint_name = "dataset_blueprint.csv"
    blueprint_path = os.path.join(dirname(abspath(__file__)), blueprint_name)
    df = pd.read_csv(blueprint_path)
    gauge_title = f"Replicas in {selected_language.capitalize()} Completion"
    filtered_df = df[df["language"] == selected_language]
    true_count = filtered_df["replication_done"].sum()
    total_count = len(filtered_df)
    fraction = true_count / total_count
    gauge_figure = update_gauge(gauge_title, fraction)

    return gauge_figure


@app.callback(
    dash.dependencies.Output("modification-gauge", "figure"),
    [dash.dependencies.Input("language-selector", "value")],
    [dash.dependencies.Input("interval-component", "n_intervals")],
)
def update_mods_gauge(selected_language: str, _: int = None) -> go.Figure:
    """
    Updates the modifications gauge figure based on the selected language.

    This method is decorated with '@app.callback', making it a callback function
    within the Dash app. It gets triggered whenever the value of the language selec-
    tor dropdown changes.

    The method first constructs a title string for the gauge that includes the
    selected language. It then filters the dataset to include only the samples co-
    rresponding to the selected language, and computes the fraction of these samples
    for which the modification has been completed (i.e., the "modification_done" co-
    lumn is True). This fraction is passed to 'update_gauge' along with the title
    string to generate a new gauge figure, which is then returned.

    Args:
        selected_language (str): The currently selected language in the dropdown.

    Returns:
        go.Figure: The updated gauge figure showing the percentage of modifications
                    completed for the selected language.
    """

    blueprint_name = "dataset_blueprint.csv"
    blueprint_path = os.path.join(dirname(abspath(__file__)), blueprint_name)
    df = pd.read_csv(blueprint_path)
    gauge_title = f"Modifications in {selected_language.capitalize()} Completion"
    filtered_df = df[df["language"] == selected_language]
    true_count = filtered_df["modification_done"].sum()
    total_count = (~filtered_df["modification_done"].astype(bool)).sum()
    fraction = true_count / total_count
    gauge_figure = update_gauge(gauge_title, fraction)

    return gauge_figure


def update_gauge(gauge_title: str, fraction: float) -> go.Figure:
    """
    Generate a gauge figure with a colored bar that represents a progress bar

    This function first calls `interpolate_color` to compute an RGB color value that
    represents the specified fraction along a color gradient. It then creates a gauge
    figure using the 'go.Indicator' class, with the gauge's value set to the specified
    fraction (expressed as a percentage), and the gauge's bar color set to the interpo-
    lated RGB color. The gauge is labelled with the specified title, and also displays
    the percentage value as anumber. The figure's background color and text color are
    set according to the global 'BACK_COLOR' and 'TEXT_COLOR' variables.

    Args:
        gauge_title (str): The title to display above the gauge.
        fraction (float): The progress bar fraction, where 0.0 is 0% and 1.0 is 100%.

    Returns:
        go.Figure: A Figure object representing the gauge.
    """

    interpolated_rgb = interpolate_color(START_COLOR, END_COLOR, fraction)
    r = interpolated_rgb[0]
    g = interpolated_rgb[1]
    b = interpolated_rgb[2]

    gauge_figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fraction * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": gauge_title,
                "font": {"size": 18},
            },
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": f"rgb({r},{g},{b})"},
            },
            number={"suffix": "%", "font": {"size": 36}},
        ),
        layout=go.Layout(
            paper_bgcolor=BACK_COLOR,
            plot_bgcolor=BACK_COLOR,
            font=dict(color=TEXT_COLOR),
        ),
    )
    return gauge_figure


def hex_to_rgb(hex_color: str) -> tuple:
    """
    Convert a hexadecimal color string to an RGB tuple.

    The function first strips any leading '#' character from the hexadecimal string.
    It  iterates over the characters of the string, taking two characters at a time
    (representing the red, green, and blue color channels in order), converts each pair
    of characters to an integer (interpreting them as a hexadecimal number), and
    collects the results into a tuple which is then returned.

    Args:
        hex_color (str): The color to convert (e.g., '#FF0000' for red).

    Returns:
        uple: A tuple of three integers representing the color in RGB format, where each
              integer is in the range 0-255.
    """

    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return rgb


def interpolate_color(start_color: str, end_color: str, fraction: float) -> tuple:
    """
    Interpolate between two colors based on a specified fraction.

    The function first converts the start and end colors from hexadecimal to RGB format.
    It then calculates the change in each color channel (red, green, blue) based on the
    specified fraction, and computes the interpolated color by adding these changes to
    the starting color. Finally, it returns the interpolated color as an RGB tuple.

    Args:
        start_color (str): The starting color in hexadecimal format.
        end_color (str): The ending color in hexadecimal format.
        fraction (float): The fraction step between the starting and ending color.


    Returns:
        tuple: A tuple representing the interpolated color in RGB format.
    """

    start_rgb = hex_to_rgb(start_color)
    end_rgb = hex_to_rgb(end_color)
    delta_rgb = [(end - start) * fraction for start, end in zip(start_rgb, end_rgb)]
    interpolated_rgb = tuple(
        int(start + delta) for start, delta in zip(start_rgb, delta_rgb)
    )
    return interpolated_rgb


def config_fig(
    df: pd.DataFrame, key: str, title_text: str, x_ticks: list, x_ticks_title: str
) -> go.Figure:
    """
    Configure a Plotly Express bar plot based on the specified parameters.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to be plotted.
        key (str): The column key in the DataFrame to be used for the bar plot's x-axis.
        title_text (str): The title text to be displayed above the bar plot.
        x_ticks (list): A list of tick labels for the x-axis.
        x_ticks_title (str): The title text for the x-axis.

    Returns:
        go.Figure: The configured bar plot figure.
    """

    figure = px.bar(df[key].value_counts())
    figure.update_layout(
        # Colors
        paper_bgcolor="black",
        plot_bgcolor="black",
        legend=dict(font=dict(color="white")),
        title={
            "text": title_text,
            "font": {"color": "white", "size": 24},
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis={
            "title": {
                "text": x_ticks_title,
                "font": {
                    "color": "white",
                },
            },
            "tickfont": {
                "color": "white",
            },
        },
        yaxis={
            "title": {
                "text": "Number of Samples",
                "font": {
                    "color": "white",
                },
            },
            "tickfont": {
                "color": "white",
            },
        },
    )

    figure.update_xaxes(tickvals=list(range(len(x_ticks))), ticktext=x_ticks)
    figure.update_traces(showlegend=False)

    return figure


# Main layout of the app
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
        dcc.Interval(
            id="interval-component",
            interval=1 * 1000,  # in milliseconds
            n_intervals=0,
        ),
    ]
)


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")],
)
def display_page(pathname: str):
    if pathname == "/statistics":
        return layout_statistics_dataset()
    else:
        return layout_input_dataset()


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
