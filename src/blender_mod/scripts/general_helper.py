import numpy as np
import os
import json
import cv2
from typing import List

SHOW_RESULTS = True


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


def get_value_from_normal(avg: float, max_delta: float, z_score: int = 3):
    """
    Generate a random sample from a normal distribution defined by an average value
    and a maximum deviation from the average.

    This function calculates the standard deviation of the distribution based on the
    specified maximum deviation (`max_delta`) and z-score, and then generates a random
    sample from this distribution using NumPy's `np.random.normal` function.

    Args:
        avg (float): The average (mean) value of the distribution.
        max_delta (float): The maximum deviation from the average value that defines the
            range within which the sample is likely to fall based on the specified
            z-score.
        z_score (int, optional): The number of standard deviations from the mean that
            defines the range of likely values. Default is 3, which corresponds to a
            99.7% confidence interval in a normal distribution.

    Returns:
        float: A random sample from the defined normal distribution.

    Note:
        - The standard deviation (`sigma`) is calculated based on the formula:
          sigma = (max_val - min_val) / (2 * z_score)
    """

    max_val = avg + max_delta
    min_val = avg - max_delta
    sigma = (max_val - min_val) / (2 * z_score)

    sample = np.random.normal(loc=avg, scale=sigma)

    return sample


def load_requirements(root: str):
    """
    Load the requirements from a JSON file located in the "assets" directory.

    This function reads a JSON file named "requirements.json" located in the "assets"
    directory under the specified root directory. It utilizes the `read_json` function
    to read the file and return the requirements as a dictionary.

    Args:
        root (str): The root directory where the "assets" directory is located.

    Returns:
        dict: A dictionary containing the Blender modification requirements.
    """
    requiremetns_path = os.path.join(root, "assets", "requirements.json")
    return read_json(requiremetns_path)


def draw_point(img: np.array, point: tuple):
    """
    Draw a point on a given image using OpenCV.

    This method takes a numpy array representing an image and a tuple representing
    the coordinates of a point. It uses the OpenCV function `cv2.circle` to draw
    a green point of radius 2 pixels at the specified coordinates on the image.

    Args:
        img (np.array): A np array (image) on which the point will be drawn.
        point (tuple): A tuple with (x, y) coordinates of the point to be drawn.

    Returns:
        np.array: The input image with the point drawn on it.
    """
    x, y = point
    img_with_point = cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    return img_with_point


def draw_bboxes(img: np.array, rectangles: List[np.ndarray[np.int32]]) -> np.array:
    """
    Draw the bboxes on a given image using the vertices in the 'rectangles' list.

    This method creates a semi-transparent overlay with the bboxes drawn on it,
    then blends this overlay with the original image to produce the output image. The
    output image is helpful to visually check the words bboxes are properly defined
    after Blender transformations.

    Args:
        img (np.array): The rendered image where bboxes will be drawn.
        rectangles (List[np.ndarray[np.int32]]): A list of np.ndarray objects where each
                                                 ndarray contains the vertices of a
                                                 transformed bbox.

    Returns:
        np.array: The image with the bboxes drawn on it.

    Note:
        - The 'rectangles' list should contain ndarrays of shape (4, 2) representing the
          coordinates of the rectangle vertices.
    """

    overlay = np.zeros_like(img)
    for rect_points in rectangles:
        # Draw a solid color rectangle on the empty image
        cv2.fillPoly(overlay, [rect_points], color=(0, 255, 0))

        # Draw the rectangle corners
        for point in rect_points:
            img = draw_point(img, point)

    # Blend images
    alpha = 0.5
    beta = 1 - alpha
    output_img = cv2.addWeighted(overlay, alpha, img, beta, 0)

    if SHOW_RESULTS:
        show_img(title="Trnasformed bboxes", img=output_img)

    return output_img


def show_img(title: str, img: np.array) -> None:
    """
    Display an image.

    Args:
        title (str): The title of the window.
        img (np.array): The image to be displayed.
    """

    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
