import os
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as collects
import seaborn as sns
import numpy as np
import ast
import json
from matplotlib.colors import to_hex, rgb_to_hsv, to_rgb, hsv_to_rgb

DARK_VIOLET = "#7030A0"
DARK_GREEN = "#006337"
LIGHT_VIOLET = "#bc71f5"
LIGHT_GREEN = "#02F186"
GREY = "#b3b3b3"

violin_color_palette = {
    "female": DARK_VIOLET,
    "male": DARK_GREEN,
    "female_pale": LIGHT_VIOLET,
    "male_pale": LIGHT_GREEN,
    "grey": GREY,
}


def read_json(name: str):
    """
    Read a JSON file from the current working directory and return its content.

    Args:
        name (str): The name of the JSON file (including the .json extension).

    Returns:
        dict: A dictionary containing the data from the JSON file.

    """
    file_path = os.path.join(os.getcwd(), name)

    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def plot_grade_violins(dataframe: pd.DataFrame, reqs: json, save_path: str):
    languages = dataframe["language"].unique()

    for language in languages:
        # Load data for vertical axis configuration
        max_grade = reqs["samples"][language]["grades_system"]["max_grade"]
        min_grade = reqs["samples"][language]["grades_system"]["min_grade"]
        grades_segments = reqs["samples"][language]["grades_system"]["alfa_grades"]

        # Filter dataframe and capitalize strings
        df_filtered = dataframe[dataframe["language"] == language].copy()
        df_filtered.loc[:, "student_name_origin"] = df_filtered[
            "student_name_origin"
        ].str.capitalize()
        df_filtered.loc[:, "student_gender"] = df_filtered[
            "student_gender"
        ].str.capitalize()
        df_filtered["average_grade"] = df_filtered["average_grade"].apply(
            ast.literal_eval
        )

        # Expand the Df to include as many rows as elements in "average_grade" cells'
        # list, convert the elements to floats and sort
        df_expanded = df_filtered.explode("average_grade")
        df_expanded["average_grade"] = df_expanded["average_grade"].astype(float)
        df_expanded = df_expanded.sort_values(by="average_grade", ascending=False)

        # Generate Plot
        plt.figure(figsize=(12, 8))

        # Draw grades categories
        for grade, info in grades_segments.items():
            plt.axhline(
                y=info["max"],
                color=violin_color_palette["male_pale"],
                linestyle="dotted",
                zorder=2,
                alpha=0.5,
            )

            midpoint = info["min"] + (info["max"] - info["min"]) / 2
            plt.text(
                1.01,
                midpoint,
                grade,
                ha="left",
                va="center",
                transform=plt.gca().get_yaxis_transform(),
                fontsize=9,
            )

        # Draw vertical grid
        positions = np.arange(len(df_expanded["student_name_origin"].unique()))
        for position in positions:
            plt.vlines(
                position,
                ymin=min_grade,
                ymax=max_grade,
                colors=violin_color_palette["grey"],
                linestyles="solid",
                alpha=0.5,
            )

        ax = sns.violinplot(
            x="student_name_origin",
            y="average_grade",
            hue="student_gender",
            data=df_expanded,
            palette={
                "Male": violin_color_palette["male_pale"],
                "Female": violin_color_palette["female_pale"],
            },
            split=True,
            inner=None,
            hue_order=["Female", "Male"],
            zorder=3,
        )

        # Borders color adjustment
        for i, artist in enumerate(ax.findobj(collects.PolyCollection)):
            color = (
                violin_color_palette["female"]
                if i % 2 == 0
                else violin_color_palette["male"]
            )
            artist.set_edgecolor(color)
            artist.set_linewidth(1.5)

        # Draw vertical left axis
        ax.set_yticks(
            np.arange(min_grade, max_grade + 1, int((max_grade - min_grade) / 10))
        )
        ax.set_yticklabels(
            np.arange(min_grade, max_grade + 1, int((max_grade - min_grade) / 10))
        )
        plt.ylim(min_grade, max_grade)

        # Legend
        ax.legend(
            title="Gender",
            labels=["Female", "Male"],
            handles=[
                plt.Line2D([0], [0], color=violin_color_palette["female"], lw=4),
                plt.Line2D([0], [0], color=violin_color_palette["male"], lw=4),
            ],
        )

        plt.title(
            f"Samples' Grade Distribution by Student Origin for {language.capitalize()}"
        )
        plt.xlabel("Student Name Origin in Sample")
        plt.ylabel("Average Grade")

        plot_save_path = os.path.join(
            save_path, "".join(["grades_distribution_", language, ".pdf"])
        )
        plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
        plt.show()


def get_blueprint():
    """
    Retrieves the dataset blueprint from a CSV file.

    This method builds the path to the CSV blueprint ('dataset_blueprint.csv'),
    located in the 'dashboard' directory. It then reads the CSV file into a pandas
    DataFrame.

    Returns:
        - pandas.DataFrame: The DataFrame created from the CSV file.
    """

    blueprint_path = os.path.join(
        Path(__file__).resolve().parents[1], "src", "dashboard", "dataset_blueprint.csv"
    )
    blueprint_df = pd.read_csv(blueprint_path)

    return blueprint_df


def config_histogram_columns(
    color_palette: dict, labels: dict, bars: matplotlib.axes._axes.Axes
):
    columns_dict = {}
    for label in labels:
        columns_dict[label] = {}
        columns_dict[label]["color"] = color_palette[label]
        columns_dict[label]["x_pos"] = None

    for bar in bars.patches:
        x_pos = bar.get_x()

        x_positions = [column["x_pos"] for column in columns_dict.values()]

        if x_pos not in x_positions:
            for column in columns_dict.values():
                if column["x_pos"] == None:
                    column["x_pos"] = x_pos
                    break

    for column in columns_dict.values():
        column["total_height"] = sum(
            bar.get_height() for bar in bars.patches if bar.get_x() == column["x_pos"]
        )

    return columns_dict


def plot_visual_categories_histogram(df: pd.DataFrame, save_path: str):
    # Preprocess data
    modifications = df["modification_done"]
    languages = df["language"].str.capitalize()

    modifications_mapped = modifications.map(
        {True: "Photorealistic", False: "Digital Document", np.nan: "N/A"}
    )
    adhoc_df = pd.DataFrame(
        {"Language": languages, "Modification": modifications_mapped}
    )
    language_modification_count = (
        adhoc_df.groupby(["Modification", "Language"]).size().unstack(fill_value=0)
    )
    normalized_counts = language_modification_count.div(
        language_modification_count.sum(axis=1), axis=0
    )

    # Color palette definition
    color_palette = {
        "Photorealistic": DARK_VIOLET,
        "Digital Document": DARK_GREEN,
        "N/A": GREY,
    }

    # Define plot
    _, ax = plt.subplots(figsize=(10, 6))
    bars = language_modification_count.plot(
        kind="bar", stacked=True, ax=ax, legend=None
    )

    labels = [item.get_text() for item in ax.get_xticklabels()]

    # Config the columns based on their x_pos
    columns_dict = config_histogram_columns(color_palette, labels, bars)

    # Draw the blocks that compose every column
    for bar, lang in zip(
        bars.patches,
        pd.Series(normalized_counts.columns).repeat(len(normalized_counts)),
    ):
        bar_x = bar.get_x()
        base_color = get_base_color(columns=columns_dict, x=bar_x)
        intensity = bar.get_height() / get_column_height(columns=columns_dict, x=bar_x)
        adjusted_color = adjust_color_intensity(base_color, intensity)
        bar.set_facecolor(adjusted_color)

        # Add text
        if bar.get_height() != 0:
            text_x = bar.get_x() + bar.get_width() / 2
            bar_height = bar.get_y() + bar.get_height() / 2
            print(f"{lang} {bar.get_x()}, {bar.get_height()}")
            plt.text(
                text_x, bar_height, f"{lang}", ha="center", va="center", color="white"
            )

    # Tune the plot
    plt.title("Sample Visual Categories: Photorealistic vs. Digital Document")
    plt.xlabel("Modification Label")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=0)
    # plt.legend(title="Language", loc="upper left", bbox_to_anchor=(1, 1))

    # Save the plot
    plot_save_path = os.path.join(save_path, "visual_categories.pdf")
    plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
    plt.show()


def get_base_color(columns: dict, x: float):

    for column in columns.values():
        if column["x_pos"] == x:
            base_color = column["color"]
            break

    return base_color


def get_column_height(columns: dict, x: float):
    for column in columns.values():
        if column["x_pos"] == x:
            height = column["total_height"]
            break

    return height


def adjust_color_intensity(color, intensity, min_intensity=0.25):
    """Adjuts color intensity"""

    rgb = to_rgb(color)
    hsv = rgb_to_hsv(rgb)
    hsv[2] = np.clip(1 - intensity, min_intensity, 1)
    rgb = hsv_to_rgb(hsv)
    hex = to_hex(rgb)

    return hex


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    requirements_path = os.path.join(
        Path(__file__).resolve().parents[1],
        "src",
        "replication_pipeline",
        "assets",
        "requirements.json",
    )
    save_path = os.path.join(
        Path(__file__).resolve().parents[0], "dataset_figs", "metrics"
    )

    # Load data
    reqs = read_json(requirements_path)
    blueprint_df = get_blueprint()

    """PLOTS"""

    # Original vs. Blender Mod replicas
    plot_visual_categories_histogram(blueprint_df, save_path)

    # Grade distributions per language, origin and gender
    plot_grade_violins(blueprint_df, reqs, save_path)
