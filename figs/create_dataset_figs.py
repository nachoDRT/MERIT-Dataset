import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as collects
import seaborn as sns
import numpy as np
import ast
import json

color_palette = {
    "female": "#7030A0",
    "male": "#006337",
    "female_pale": "#bc71f5",
    "male_pale": "#02F186",
    "grey": "#b3b3b3",
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
                color=color_palette["male_pale"],
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
                colors=color_palette["grey"],
                linestyles="solid",
                alpha=0.5,
            )

        ax = sns.violinplot(
            x="student_name_origin",
            y="average_grade",
            hue="student_gender",
            data=df_expanded,
            palette={
                "Male": color_palette["male_pale"],
                "Female": color_palette["female_pale"],
            },
            split=True,
            inner=None,
            hue_order=["Female", "Male"],
            zorder=3,
        )

        # Borders color adjustment
        for i, artist in enumerate(ax.findobj(collects.PolyCollection)):
            color = color_palette["female"] if i % 2 == 0 else color_palette["male"]
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
                plt.Line2D([0], [0], color=color_palette["female"], lw=4),
                plt.Line2D([0], [0], color=color_palette["male"], lw=4),
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

    reqs = read_json(requirements_path)
    blueprint_df = get_blueprint()

    plot_grade_violins(blueprint_df, reqs, save_path)
