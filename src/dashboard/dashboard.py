from dash import Dash, html, dcc, Input, Output, ALL, State
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

START_COLOR = "#ccffff"
END_COLOR = "#009933"
BACK_COLOR = "#000000"
TEXT_COLOR = "#FFFFFF"


blueprint_name = "dataset_blueprint.csv"
blueprint_path = os.path.join(dirname(abspath(__file__)), blueprint_name)
df = pd.read_csv(blueprint_path)
css_path = os.path.join(dirname(abspath(__file__)), "assets", "styles.css")
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.QUARTZ, css_path],
    suppress_callback_exceptions=True,
)

langs_dict = dhelp.get_available_schools_per_language()


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

    dhelp.update_fe_male_proportion_requirements(value)

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
    lang = "portuguese"
    iterable_dict = ["hola", "hola_1"]
    origin_layout = [html.H3("Select Students' Features", style={"marginTop": "1.5em"})]
    row = dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader(html.Div(id="card-title")),
                            dbc.CardBody(
                                html.Div(id="card-content"),
                            ),
                        ],
                        id="origin-proportion-card",
                    )
                    # dbc.Checklist(
                    #     options=[{"label": lang.capitalize(), "value": lang}],
                    #     value=[],
                    #     id=f"origin-dropdown-{lang}",
                    #     switch=True,
                    #     className="custom-switch",
                    # ),
                    #     dbc.Collapse(
                    #         dbc.Checklist(
                    #             options=[
                    #                 {"label": element.capitalize(), "value": element}
                    #                 for element in iterable_dict
                    #             ],
                    #             value=["hola"],
                    #             id=f"checklist-{lang}",
                    #             inline=True,
                    #         ),
                    #         id=f"collapse-lang-{lang}",
                    #     ),
                ]
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(f"Option B"),
                        dbc.CardBody("HELLO"),
                    ],
                    id="origin-proportion-card",
                )
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(f"Option C"),
                        dbc.CardBody("HELLO"),
                    ],
                    id="origin-marks-bias-card",
                )
            ),
        ]
    )
    origin_layout.extend([row])
    return origin_layout


@app.callback(
    Output("card-title", "children"),
    Output("card-content", "children"),
    Input("lang-options-carousel", "active_index"),
    Input("hidden-div-carousel", "children"),
)
def update_language_origin_selector(active_index, json_data):
    print(json_data)
    print(active_index)

    if isinstance(json_data, str):
        json_data = eval(json_data)

    if json_data[0]["key"] == "no_lang":
        title = "No language Option"
        body = "Select a school"

    else:
        if active_index == None:
            active_index = 0
        elif active_index >= len(json_data):
            active_index -= 1
        title = json_data[active_index]["header"]
        body = title
    return title, body


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
    in_layout.extend([create_prop_slider(team_a="Female", team_b="Male")])
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

    dhelp.update_schools_requirements(selected_schools)
    selected_languages = dhelp.get_langs_with_replicas()
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
                style={"margin": "auto"},
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
