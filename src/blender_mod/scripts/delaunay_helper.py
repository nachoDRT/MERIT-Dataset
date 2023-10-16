from typing import List, Tuple, Type, Any, Dict
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import json
import os


SHOW_PLOT = True


def get_bboxes_as_points(form: List[Dict[str, Any]]) -> np.array:
    """
    Get a list containing all the points of the word's bounding boxes as pixel coordina-
    tes (x, y).

    Args:
        form (list): A list containing all the segments of a form. Each segment should
            be a dictionary containing (among others) a 'words' key with bounding
            box(es) data.

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
        bboxes_points.extend(extract_boxes(form_entry))

    return np.array(bboxes_points)


def extract_boxes(form_entry: dict) -> List[Tuple[int, int]]:
    """
    Extract the pixel coordinates of the bounding boxes from the words in the JSON dict.

    The `form_entry` dictionary should have a key 'words' which contains a list of
    dictionaries, each with a key 'box' that holds a list of 4 integers representing
    the coordinates of the bounding box.

    Args:
        form_entry (dict): A dataset segment containing words and their bounding boxes.

    Returns:
        list[tuple[int, int]]: A list with the points of the each bounding box. Each
            point is represented as a tuple of two integers (x, y).
    """

    points = []
    for word in form_entry["words"]:
        points.append(word["box"][0:2])
        points.append(word["box"][2:4])

    return points


def compute_grid(properties: dict, sampling_x: int = 50, sampling_y: int = 50):
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
        sampling_x (int): The sampling interval along the x-axis. Default is 50.
        sampling_y (int): The sampling interval along the y-axis.Default is 50.

    Returns:
        np.array: An array of tuples where each tuple contains the x and y coordinates
            of a point on the grid. The grid points are ordered first along the x-axis,
            then along the y-axis.
    """

    a4_pixel_width = properties["delaunay"]["a4_pixel_dims"]["width"]
    a4_pixel_height = properties["delaunay"]["a4_pixel_dims"]["height"]

    x = np.arange(0, a4_pixel_width, sampling_x)
    y = np.arange(0, a4_pixel_height, sampling_y)

    x_grid, y_grid = np.meshgrid(x, y)

    points = np.vstack((x_grid.ravel(), y_grid.ravel())).T

    grid = [tuple(point) for point in points]

    return np.array(grid)


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


def show_plot(
    vertices: np.array, default_grid: np.array, mesh: Type[Delaunay], name: str
):
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
    plt.plot(default_grid[:, 0], default_grid[:, 1], "o", markersize=1)
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
