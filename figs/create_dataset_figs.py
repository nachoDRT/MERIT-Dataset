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
BRIGHT_GREEN = "#68E807"
LIGHT_VIOLET = "#bc71f5"
LIGHT_GREEN = "#02F256"
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
        column["total_height_plotted"] = False

    return columns_dict


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


def get_flag_column_total_num_samples_plotted(columns: dict, x: float):
    for column in columns.values():
        if column["x_pos"] == x:
            flag = column["total_height_plotted"]
            break

    return flag


def set_flag_column_total_num_samples_plotted(columns: dict, x: float):
    for column in columns.values():
        if column["x_pos"] == x:
            column["total_height_plotted"] = True
            break


def adjust_color_intensity(color, intensity, min_intensity=0.35):
    """Adjuts color intensity"""

    rgb = to_rgb(color)
    hsv = rgb_to_hsv(rgb)
    hsv[2] = np.clip(1 - intensity, min_intensity, 1)
    rgb = hsv_to_rgb(hsv)
    hex = to_hex(rgb)

    return hex


def get_mapping_dict(features: pd.core.series.Series, maps_to: pd.core.series.Series):
    mapping_dict = {}

    for feature, map_to in zip(features, maps_to):
        if feature not in mapping_dict:
            mapping_dict[feature] = map_to

    return mapping_dict


def get_color_generic_palette(labels: list):
    color_palette = {}

    for label in labels:
        if len(color_palette) % 2 == 0:
            color_palette[label] = DARK_GREEN
        else:
            color_palette[label] = DARK_VIOLET

    return color_palette


def plot_students_per_name_origin_lang_gender(df: pd.DataFrame, save_path: str):
    languages = df["language"].unique()

    color_palette = {
        "Female": DARK_VIOLET,
        "Male": DARK_GREEN,
    }

    for language in languages:

        filtered_categories = df.drop_duplicates(
            subset=["student_gender", "student_index"]
        )
        categories = filtered_categories["student_gender"].str.capitalize()
        filtered_categories = categories[df["language"] == language]

        filtered_category_items = df.drop_duplicates(
            subset=["student_name_origin", "student_index"]
        )
        categories_items = filtered_category_items[
            "student_name_origin"
        ].str.capitalize()
        filtered_category_items = categories_items[df["language"] == language]

        plot_histogram(
            cat_items=filtered_category_items,
            cat_mapped=filtered_categories,
            save_path=save_path,
            title=f"Students per name origin in {language.capitalize()}",
            x_label="Gender",
            y_label="Number of Samples",
            plot_name=f"students_per_name_origin_{language}_recount",
            color_palette=color_palette,
        )


def plot_samples_per_name_origin_lang_gender(df: pd.DataFrame, save_path: str):
    languages = df["language"].unique()

    color_palette = {
        "Female": DARK_VIOLET,
        "Male": DARK_GREEN,
    }

    for language in languages:

        categories = df["student_gender"].str.capitalize()
        filtered_categories = categories[df["language"] == language]

        categories_items = df["student_name_origin"].str.capitalize()
        filtered_category_items = categories_items[df["language"] == language]

        plot_histogram(
            cat_items=filtered_category_items,
            cat_mapped=filtered_categories,
            save_path=save_path,
            title=f"Samples per name origin in {language.capitalize()}",
            x_label="Gender",
            y_label="Number of Samples",
            plot_name=f"samples_per_name_origin_{language}_recount",
            color_palette=color_palette,
        )


def plot_samples_per_school(df: pd.DataFrame, save_path: str):
    categories = df["language"].str.capitalize()
    categories_items = df["school_name"].str.capitalize()
    mapping = get_mapping_dict(categories_items, categories)
    cat_mapped = categories_items.map(mapping)

    plot_histogram(
        cat_items=categories_items,
        cat_mapped=cat_mapped,
        save_path=save_path,
        title="Samples per School",
        x_label="Languages",
        y_label="Number of Samples",
        plot_name="samples_per_school_recount",
    )


def plot_students_per_school(df: pd.DataFrame, save_path: str):

    filtered_categories = df.drop_duplicates(subset=["language", "student_index"])
    categories = filtered_categories["language"].str.capitalize()

    categories_items = df["school_name"].str.capitalize()
    filtered_categories_items = df.drop_duplicates(
        subset=["school_name", "student_index"]
    )
    categories_items = filtered_categories_items["school_name"].str.capitalize()

    mapping = get_mapping_dict(categories_items, categories)
    cat_mapped = categories_items.map(mapping)

    plot_histogram(
        cat_items=categories_items,
        cat_mapped=cat_mapped,
        save_path=save_path,
        title="Students per School",
        x_label="Languages",
        y_label="Number of Students",
        plot_name="students_per_school_recount",
    )


def plot_visual_categories_histogram(df: pd.DataFrame, save_path: str):
    categories = df["modification_done"]
    categories_items = df["language"].str.capitalize()
    color_palette = {
        "Photorealistic": DARK_VIOLET,
        "Digital Document": DARK_GREEN,
        "Non Processed": GREY,
    }

    mapping = {
        True: "Photorealistic",
        False: "Non Processed",
        np.nan: "Digital Document",
    }

    cat_mapped = categories.map(mapping)

    plot_histogram(
        cat_items=categories_items,
        cat_mapped=cat_mapped,
        save_path=save_path,
        title="Visual Styles",
        x_label="Visual Styles",
        y_label="Number of Samples",
        plot_name="documents_visual_styles_recount",
        color_palette=color_palette,
    )


def plot_samples_per_layout_model_histogram(df: pd.DataFrame, save_path: str):
    categories = df["layout"]
    categories_items = df["language"].str.capitalize()

    mapping = {
        "model_a_single_column": "Model A.1",
        "model_a_double_column": "Model A.2",
        "model_b_two_tables": "Model B.1",
        "model_b_three_tables": "Model B.2",
        "model_c": "Model C",
    }
    cat_mapped = categories.map(mapping)

    plot_histogram(
        cat_items=categories_items,
        cat_mapped=cat_mapped,
        save_path=save_path,
        title="Layout Styles",
        x_label="Languages",
        y_label="Number of Samples",
        plot_name="layout_styles_recount",
    )


def plot_rendering_styles_histogram(df: pd.DataFrame, save_path: str, b_props: dict):
    categories = df["rendering_style"]
    categories_items = df["language"].str.capitalize()

    # TODO: read styles from b_prop
    color_palette = {
        "Scanner": DARK_VIOLET,
        "Natural": DARK_GREEN,
        "Office": BRIGHT_GREEN,
        "Non Processed": GREY,
    }

    mapping = {
        "scanner": "Scanner",
        "natural": "Natural",
        "office": "Office",
        np.nan: "Non Processed",
    }

    cat_mapped = categories.map(mapping)

    plot_histogram(
        cat_items=categories_items,
        cat_mapped=cat_mapped,
        save_path=save_path,
        title="Rendering Styles",
        x_label="Rendering Styles",
        y_label="Number of Samples",
        plot_name="documents_render_styles_recount",
        color_palette=color_palette,
    )


def plot_modified_meshes_histogram(df: pd.DataFrame, save_path: str):
    categories = df["modify_mesh"]
    categories_items = df["language"].str.capitalize()
    color_palette = {
        "Cloth Simulation": DARK_VIOLET,
        "Simple Plane": DARK_GREEN,
        "Non Processed": GREY,
    }

    mapping = {
        True: "Cloth Simulation",
        False: "Simple Plane",
        np.nan: "Non Processed",
    }

    cat_mapped = categories.map(mapping)

    plot_histogram(
        cat_items=categories_items,
        cat_mapped=cat_mapped,
        save_path=save_path,
        title="Meshes",
        x_label="Mesh Modifications",
        y_label="Number of Samples",
        plot_name="documents_meshes_recount",
        color_palette=color_palette,
    )


def plot_shadow_samples_histogram(df: pd.DataFrame, save_path: str):
    categories = df["shadow_casting"]
    categories_items = df["language"].str.capitalize()
    color_palette = {
        "Shadow Object": DARK_VIOLET,
        "No Shadow Object": DARK_GREEN,
        "Non Processed": GREY,
    }

    mapping = {
        True: "Shadow Object",
        False: "No Shadow Object",
        np.nan: "Non Processed",
    }

    cat_mapped = categories.map(mapping)

    plot_histogram(
        cat_items=categories_items,
        cat_mapped=cat_mapped,
        save_path=save_path,
        title="Shadows",
        x_label="Shadows",
        y_label="Number of Samples",
        plot_name="documents_shadows_recount",
        color_palette=color_palette,
    )


def plot_stain_samples_histogram(df: pd.DataFrame, save_path: str):
    categories = df["printer_stains"]
    categories_items = df["language"].str.capitalize()
    color_palette = {
        "Printer Stains": DARK_VIOLET,
        "Clean Printout": DARK_GREEN,
        "Non Processed": GREY,
    }

    mapping = {
        True: "Printer Stains",
        False: "Clean Printout",
        np.nan: "Non Processed",
    }

    cat_mapped = categories.map(mapping)

    plot_histogram(
        cat_items=categories_items,
        cat_mapped=cat_mapped,
        save_path=save_path,
        title="Printer Stains",
        x_label="Stains in Printout",
        y_label="Number of Samples",
        plot_name="documents_stains_recount",
        color_palette=color_palette,
    )


def plot_background_elements_histogram(df: pd.DataFrame, save_path: str, b_props: dict):
    categories = df["background_elements"]
    categories_items = df["language"].str.capitalize()

    mapping = {np.nan: "Non Processed", "False": "No object"}

    for object in b_props["blender"]["background_objects"]:
        mapping[object] = object.replace("_", " ").capitalize()

    color_palette = {"Non Processed": GREY, "No object": GREY}

    for i, map in enumerate(mapping.values()):

        if i % 2 == 0:
            color = DARK_GREEN
        else:
            color = DARK_VIOLET

        color_palette[map.capitalize()] = color

    cat_mapped = categories.map(mapping)

    plot_histogram(
        cat_items=categories_items,
        cat_mapped=cat_mapped,
        save_path=save_path,
        title="Background Objects",
        x_label="Object",
        y_label="Number of Samples",
        plot_name="documents_backgroun_objects_recount",
        color_palette=color_palette,
    )


def plot_background_materials_histogram(
    df: pd.DataFrame, save_path: str, b_props: dict
):
    categories = df["background_material"]
    categories_items = df["language"].str.capitalize()

    # TODO: read background materials from b_prop
    """mapping = {np.nan: "Non Processed"}

    # for object in b_props["blender"]["background_objects"]:
    #     mapping[object] = object.replace("_", " ").capitalize()

    # color_palette = {"Non Processed": GREY}

    # for i, map in enumerate(mapping.values()):

    #     if i % 2 == 0:
    #         color = DARK_GREEN
    #     else:
    #         color = DARK_VIOLET

    #     color_palette[map.capitalize()] = color"""

    # TODO: delete these lines when updated with b_prop
    mapping = {
        "white_plastic": "White Plastic",
        "wood": "Concrete",
        np.nan: "Non Processed",
    }

    color_palette = {
        "White Plastic": DARK_VIOLET,
        "Concrete": DARK_GREEN,
        "Non Processed": GREY,
    }
    #

    cat_mapped = categories.map(mapping)

    plot_histogram(
        cat_items=categories_items,
        cat_mapped=cat_mapped,
        save_path=save_path,
        title="Background Material",
        x_label="Material",
        y_label="Number of Samples",
        plot_name="documents_background_material_recount",
        color_palette=color_palette,
    )


def plot_histogram(
    cat_items,
    cat_mapped: dict,
    save_path: str,
    title: str,
    x_label: str,
    y_label: str,
    plot_name: str,
    color_palette: dict = None,
):

    adhoc_df = pd.DataFrame({"Items": cat_items, "Categories": cat_mapped})

    # This is to order the columns from left to right in descending order
    sum_per_category = adhoc_df.groupby("Categories").sum()
    ordered_categories = sum_per_category.sort_values(by="Items", ascending=True).index

    count = adhoc_df.groupby(["Categories", "Items"]).size().unstack(fill_value=0)

    normalized_counts = count.div(count.sum(axis=1), axis=0)
    ordered_count = count.reindex(ordered_categories)

    # Define plot
    _, ax = plt.subplots(figsize=(10, 6))
    bars = ordered_count.plot(kind="bar", stacked=True, ax=ax, legend=None)

    labels = [item.get_text() for item in ax.get_xticklabels()]

    if not color_palette:
        color_palette = get_color_generic_palette(labels)

    # Config the columns based on their x_pos
    columns_dict = config_histogram_columns(color_palette, labels, bars)

    # Get the height of the highest column
    max_bar_height = 0
    for bar in bars.patches:
        patch_height = bar.get_y() + bar.get_height()

        if max_bar_height < patch_height:
            max_bar_height = patch_height

    max_bar_height = int(max_bar_height * 1.0225)

    # Draw the blocks that compose every column
    for bar, lang in zip(
        bars.patches,
        pd.Series(normalized_counts.columns).repeat(len(normalized_counts)),
    ):
        bar_x = bar.get_x()
        base_color = get_base_color(columns=columns_dict, x=bar_x)
        column_height = get_column_height(columns=columns_dict, x=bar_x)
        intensity = bar.get_height() / column_height
        adjusted_color = adjust_color_intensity(base_color, intensity)
        bar.set_facecolor(adjusted_color)

        # Add text
        if bar.get_height() != 0:
            text_x = bar.get_x() + bar.get_width() / 2
            text_y = bar.get_y() + bar.get_height() / 2
            num_x = bar.get_x() + bar.get_width()
            num_y = bar.get_y() + bar.get_height()
            num_samples = int(bar.get_height())
            plt.text(text_x, text_y, f"{lang}", ha="center", va="center", color="white")
            plt.text(
                num_x,
                num_y,
                f" + {num_samples}",
                ha="left",
                va="center",
                color="grey",
                fontsize=9,
            )

        # Add total number of samples if it has not been done earlier
        if (
            not get_flag_column_total_num_samples_plotted(columns=columns_dict, x=bar_x)
            and bar.get_height() != 0
        ):
            # text_x = bar.get_x() + bar.get_width() / 2
            plt.text(
                text_x,
                max_bar_height,
                f"{int(column_height)}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                bbox=dict(facecolor=base_color, edgecolor="none", pad=1.0),
            )
            set_flag_column_total_num_samples_plotted(columns=columns_dict, x=bar_x)

    # Tune the plot
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=0)

    # Save the plot
    plot_save_path = os.path.join(save_path, "".join([plot_name, ".pdf"]))
    plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    requirements_path = os.path.join(
        Path(__file__).resolve().parents[1],
        "src",
        "replication_pipeline",
        "assets",
        "requirements.json",
    )
    blender_props_path = os.path.join(
        Path(__file__).resolve().parents[1],
        "src",
        "blender_mod",
        "assets",
        "properties.json",
    )

    save_path = os.path.join(
        Path(__file__).resolve().parents[0], "dataset_figs", "metrics"
    )

    # Load data
    reqs = read_json(requirements_path)
    blender_props = read_json(blender_props_path)
    blueprint_df = get_blueprint()

    """PLOTS"""

    """ General information"""
    # Number of samples per school
    plot_samples_per_school(blueprint_df, save_path)
    # Number os students per school
    plot_students_per_school(blueprint_df, save_path)

    # """ Visual appearence of samples"""
    # Original vs. Blender Mod replicas
    plot_visual_categories_histogram(blueprint_df, save_path)
    # Rendering style
    plot_rendering_styles_histogram(blueprint_df, save_path, blender_props)
    # Samples with modified meshes
    plot_modified_meshes_histogram(blueprint_df, save_path)
    # Samples with shadows
    plot_shadow_samples_histogram(blueprint_df, save_path)
    # Samples with printer stains
    plot_stain_samples_histogram(blueprint_df, save_path)
    # Samples with elements in background
    plot_background_elements_histogram(blueprint_df, save_path, blender_props)
    # Background materials in sample
    plot_background_materials_histogram(blueprint_df, save_path, blender_props)

    """ Layout of samples"""
    # Number of samples per layout model
    plot_samples_per_layout_model_histogram(blueprint_df, save_path)

    """ Biases"""
    # Number of student distribution per names origin, language and gender
    plot_students_per_name_origin_lang_gender(blueprint_df, save_path)
    # Number of samples distribution per names origin, language and gender
    plot_samples_per_name_origin_lang_gender(blueprint_df, save_path)
    # Grade distributions per language, origin and gender
    # plot_grade_violins(blueprint_df, reqs, save_path)
