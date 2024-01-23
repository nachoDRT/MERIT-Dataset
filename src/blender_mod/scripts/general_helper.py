import numpy as np
import os
import json
import cv2
import copy
from typing import List, Tuple

SHOW_RESULTS = False


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


def draw_point(img: np.array, point: tuple, color: tuple):
    """
    Draw a point on a given image using OpenCV.

    This method takes a numpy array representing an image and a tuple representing
    the coordinates of a point. It uses the OpenCV function `cv2.circle` to draw
    a green point of radius 2 pixels at the specified coordinates on the image.

    Args:
        img (np.array): A np array (image) on which the point will be drawn.
        point (tuple): A tuple with (x, y) coordinates of the point to be drawn.
        color (tuple): A BGR color.

    Returns:
        np.array: The input image with the point drawn on it.
    """
    x, y = point
    img_with_point = cv2.circle(img, (x, y), 2, color, -1)

    return img_with_point


def draw_bboxes(
    img: np.array, rectangles: List[np.ndarray[np.int32]], color: tuple
) -> np.array:
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
        color (tuple): A BGR color.

    Returns:
        np.array: The image with the bboxes drawn on it.

    Note:
        - The 'rectangles' list should contain ndarrays of shape (4, 2) representing the
          coordinates of the rectangle vertices.
    """

    overlay = np.zeros_like(img)
    for rect_points in rectangles:
        # Draw a solid color rectangle on the empty image
        cv2.fillPoly(overlay, [rect_points], color=color)

        # Draw the rectangle corners
        for point in rect_points:
            img = draw_point(img, point, color)

    # Blend images
    alpha = 0.5
    beta = 1 - alpha
    output_img = cv2.addWeighted(overlay, alpha, img, beta, 0)

    if SHOW_RESULTS:
        show_img(title="Transformed bboxes", img=output_img)

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


def write_json(data: dict, file_path: str) -> None:
    """
    Write a dictionary to a JSON file.

    Args:
        data (dict): The data to be written to the file.
        file_path (str): The path of the file to write to.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def collect_bbox_pts_in_list(word_index: int, points: List) -> List:
    """
    Flatten a list of lists into a single list.

    Parameters:
        word_index (int): Index of the item in the `points` list.
        points (List): List containing points or coordinates.

    Returns:
        List: Flattened list of coordinates.
    """
    try:
        points_list = [
            coordinate for point in points[word_index].tolist() for coordinate in point
        ]
    except AttributeError:
        points_list = [
            coordinate for point in points[word_index] for coordinate in point
        ]

    return points_list


def collect_transformed_segment_bbox(bboxes: List) -> List:
    """
    Create a new bounding box encompassing a text segment.

    Parameters:
        bboxes (List): List containing bounding boxes.

    Returns:
        List: List of lists containing the coordinates of the four corners of
        the new segment bounding box.
    """
    pt0 = [bboxes[0][0], bboxes[0][1]]
    pt1 = [bboxes[0][2], bboxes[0][3]]
    pt2 = [bboxes[-1][-4], bboxes[-1][-3]]
    pt3 = [bboxes[-1][-2], bboxes[-1][-1]]
    segment_bbox = [pt0, pt1, pt2, pt3]

    return segment_bbox


def edit_json_labels(json_path: str, points: List) -> Tuple[dict, List]:
    """
    Read the JSON file with the original labels and replace the original bounding boxes
    with the new ones (after Blender modifications: camera rotatation, traslation, etc.)

    Parameters:
        json_path (str): Path to the original JSON file containing the labels.
        points (List): A list of lists (bbounding boxes with four points delimiting
                       every word bbox)

    Returns:
        Tuple[dict, List]: Tuple containing the edited JSON object and a list of the new
                           segments bounding boxes.
    """

    labels = read_json(name=json_path)
    labels_edited = copy.deepcopy(labels)

    word_index = 0
    segments_bboxes = []
    for segment_index, segment in enumerate(labels["form"]):
        segment_bboxes = []
        for word_index_in_segment, _ in enumerate(segment["words"]):
            edited_bbox = collect_bbox_pts_in_list(word_index, points)
            labels_edited["form"][segment_index]["words"][word_index_in_segment][
                "box"
            ] = edited_bbox
            segment_bboxes.append(edited_bbox)
            word_index += 1

        # Obtain the transformed segment bbounding box
        segment_bboxes = collect_transformed_segment_bbox(bboxes=segment_bboxes)
        edited_segment_bbox = collect_bbox_pts_in_list(
            word_index=0, points=[segment_bboxes]
        )
        labels_edited["form"][segment_index]["box"] = edited_segment_bbox
        segments_bboxes.append(np.array(segment_bboxes))

    return labels_edited, segments_bboxes
