from typing import List, Tuple, Type, Any, Dict
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import json
import os
import math


SHOW_PLOT = True


def get_bboxes_as_points(form: List[Dict[str, Any]], properties: dict) -> np.array:
    """
    Get a list containing all the points of the word's bounding boxes as pixel coordina-
    tes (x, y).

    Args:
        form (list): A list containing all the segments of a form. Each segment should
            be a dictionary containing (among others) a 'words' key with bounding
            box(es) data.
        properties (dict): A dictionary containing the properties of the A4 page
            including its dimensions in pixels.

    Returns:
        np.array: An array containing all the key points as pixel coordinates, where
            each row represents a point and has two columns for the x and y coordinates
            respectively.

    Example:
        >>> form = {
                [
                    {
                        "box": [x1_min, y1_min, x2_max, y2_max],
                        "text": "some_text",
                        "label": "some_label",
                        "words": [
                            {"box": [x1_min, y1_min, x1_max, y1_max]},
                            {"box": [x2_min, y2_min, x2_max, y2_max]},
                        ],
                        "linking": link,
                        "id": i,
                    },
                    ...
                    {
                        ...
                    }
                ]
            }

        >>> get_bboxes_as_points(form)
        array([[x1_min, y1_min],
               [x1_max, y1_max],
               ...
               [xn_min, yn_min],
               [xn_max, yn_max]])
    """

    bboxes_points = []
    for form_entry in form:
        bboxes_points.extend(extract_boxes(form_entry, properties))

    return np.array(bboxes_points)


def reference_system_correction(point: list, adjust: str, properties: dict) -> list:
    """
    Adjust a point's coordinate based on the reference system of the document.

    This method modifies a single coordinate (x or y) of a given point according to the
    dimensions specified in the properties dictionary. The purpose is to ensure that the
    point's coordinates are consistent with the reference system used in Blender.

    Args:
        point (list): A list containing the coordinates of the point to be adjusted.
        adjust (str): A string indicating which coordinate ('x' or 'y') to adjust.
        properties (dict): A dictionary containing the configuration values, including
                           the dimensions of the document.

    Returns:
        list: A list containing the updated coordinates of the point.
    """

    if adjust == "x":
        index_correction = 0
        a4_pixel_dim = properties["delaunay"]["a4_pixel_dims"]["width"]
    elif adjust == "y":
        index_correction = 1
        a4_pixel_dim = properties["delaunay"]["a4_pixel_dims"]["width"]

    point[index_correction] = a4_pixel_dim - point[index_correction]

    return point


def extract_boxes(form_entry: dict, properties: dict) -> List[Tuple[int, int]]:
    """
    Extract the pixel coordinates of the bounding boxes from the words in the JSON dict.

    The `form_entry` dictionary should have a key 'words' which contains a list of
    dictionaries, each with a key 'box' that holds a list of 4 integers representing
    the coordinates of the bounding box.

    Args:
        form_entry (dict): A dataset segment containing words and their bounding boxes.
        properties (dict): A dictionary containing the properties of the A4 page
            including its dimensions in pixels.

    Returns:
        list[tuple[int, int]]: A list with the points of the each bounding box. Each
            point is represented as a tuple of two integers (x, y).
    """

    points = []
    for word in form_entry["words"]:
        second_point = [word["box"][0], word["box"][3]]
        fourth_point = [word["box"][2], word["box"][1]]

        # Append points in a counter-clockwise order
        points.append(reference_system_correction(word["box"][0:2], "x", properties))
        points.append(reference_system_correction(second_point, "x", properties))
        points.append(reference_system_correction(word["box"][2:4], "x", properties))
        points.append(reference_system_correction(fourth_point, "x", properties))

    return points


def compute_grid(properties: dict, sampling_x: int = 100, sampling_y: int = 100):
    """
    Generate a grid of points within the bounding box defined by the dimensions of
    an A4 page.

    The function creates a grid of points by sampling the space within the bounds of
    an A4 page. The interval between adjacent grid points along the x and y axes are
    defined by `sampling_x` and `sampling_y` respectively. These intervals determine
    the density of the grid points.

    Args:
        properties (dict): A dictionary containing the properties of the A4 page
            including its dimensions in pixels.
        sampling_x (int): The sampling interval along the x-axis. Default is 100.
        sampling_y (int): The sampling interval along the y-axis.Default is 100.

    Returns:
        np.array: An array of tuples where each tuple contains the x and y coordinates
            of a point on the grid. The grid points are ordered first along the x-axis,
            then along the y-axis.
    """

    a4_pixel_width = properties["delaunay"]["a4_pixel_dims"]["width"]
    a4_pixel_height = properties["delaunay"]["a4_pixel_dims"]["height"]

    steps_x = math.ceil((a4_pixel_width) / sampling_x) + 1
    steps_y = math.ceil((a4_pixel_height) / sampling_y) + 1

    x = np.linspace(0, a4_pixel_width, num=steps_x, dtype=int)
    x_step = x[1] + 1
    y = np.linspace(0, a4_pixel_height, num=steps_y, dtype=int)
    y_step = y[1] + 1

    x_grid, y_grid = np.meshgrid(x, y)

    points = np.vstack((x_grid.ravel(), y_grid.ravel())).T

    grid = [tuple(point) for point in points]
    grid_marks = {"x": x, "y": y}

    return np.array(grid), x_step, y_step, grid_marks


def read_json(name: str):
    """
    Read a JSON file from the current working directory and returns its content.

    Args:
        name (str): The name of the JSON file (including the .json extension).

    Returns:
        dict: A dictionary containing the data from the JSON file.

    """

    file_path = os.path.join(os.getcwd(), name)

    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def show_plot(vertices: np.array, mesh: Type[Delaunay], name: str):
    """
    Display a plot showing a Delaunay triangulation (vertices come from both a default
    grid and the words' bounding boxes in the document of interest).

    Args:
        vertices (np.array): A 2D numpy array of shape (n, 2) representing the vertices
            coordinates (it includes default_grid + words' bounding boxes vertices).
            default_grid (np.array): A 2D numpy array of shape (m, 2) representing the
            grid points coordinates.
        mesh (Type[Delaunay]): A Delaunay object representing the triangulation mesh.
        name (str): The title to be displayed at the top of the plot.
    """

    plt.style.use("ggplot")
    plt.title(label=name)
    plt.triplot(vertices[:, 0], vertices[:, 1], mesh.simplices)
    plt.plot(vertices[:, 0], vertices[:, 1], "o", markersize=2, color="purple")
    plt.xlabel("Horizontal pixels")
    plt.ylabel("Vertical pixels")
    plt.axis("equal")
    plt.show()


def list_to_tuple_items(vector: list):
    """
    Converts each item of a given list to a tuple.

    Args:
        vector (list): The list containing items to be converted to tuples.

    Returns:
        list: A new list with each item from the input list converted to a tuple.
    """
    return [tuple(item) for item in vector]


def pixel_to_m(vector: list, properties: dict):
    """
    Converts a list of coordinates from pixels to meters.

    This function iterates through a list of tuples, each containing coordinates in
    pixels, and converts these coordinates to meters based on the conversion factor
    provided in the 'properties' dictionary. The conversion involves two steps:

    1. Converting pixels to millimeters using the '1 / mm_2_pixel' conversion factor.
    2. Converting millimeters to meters by dividing by 1000.

    Args:
        vector (list): A list of tuples where each tuple contains coordinates in pixels.
        properties (dict): A dictionary containing the conversion factor under the keys
            'delaunay' and 'mm_2_pixel'.

    Returns:
        np.array: A NumPy array of tuples where each tuple contains the converted
            coordinates in meters.
    """

    mm_2_pixel = properties["delaunay"]["mm_2_pixel"]
    pixel_to_mm = 1 / mm_2_pixel
    pixel_to_m = pixel_to_mm / 1000

    transformed = []
    for vertex in vector:
        vertex = tuple(np.array(vertex) * pixel_to_m)
        transformed.append(vertex)
    return np.array(transformed)


def coord_m_to_pixel(coord: float, properties: dict) -> int:
    mm_2_pixel = properties["delaunay"]["mm_2_pixel"]
    coord_mm = coord * 1000
    coord_px = coord_mm * mm_2_pixel

    return int(coord_px)


def coord_pixel_to_m(coord: int, properties: dict) -> float:
    mm_2_pixel = properties["delaunay"]["mm_2_pixel"]
    pixel_to_mm = 1 / mm_2_pixel
    pixel_to_m = pixel_to_mm / 1000
    coord_m = coord * pixel_to_m

    return coord_m


def load_properties(root: str):
    """
    Load the properties from a JSON file located in the "assets" directory.

    This function reads a JSON file named "properties.json" located in the "assets"
    directory under the specified root directory. It utilizes the `read_json` function
    to read the file and return the properties as a dictionary.

    Args:
        root (str): The root directory where the "assets" directory is located.

    Returns:
        dict: A dictionary containing the properties loaded from the JSON file.
    """

    proprerties_path = os.path.join(root, "assets", "properties.json")
    return read_json(proprerties_path)
