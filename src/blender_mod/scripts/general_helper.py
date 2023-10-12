import numpy as np
import os
import json


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
