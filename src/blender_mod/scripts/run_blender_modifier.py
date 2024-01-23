import subprocess
import os
from os.path import dirname, abspath
from pathlib import Path
import json

root = Path(dirname(abspath(__file__))).parent

try:
    blender_exe_path_keeper = os.path.join(
        root,
        "assets",
        "blender_executable_path.json",
    )

    with open(blender_exe_path_keeper, "r") as file:
        data = json.load(file)

    blender_exe_path = data["path"]

except FileNotFoundError:
    blender_exe_path = "blender"

blend_file = os.path.join(root, "mod.blend")
python_script = os.path.join(root, "scripts", "blender_modifier.py")

command = [blender_exe_path, blend_file, "--background", "--python", python_script]
subprocess.run(command)
