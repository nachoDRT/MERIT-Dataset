import bpy
import numpy as np
from scipy.spatial import Delaunay
import sys
import os
import math
import colorsys
from pathlib import Path
import random

file_dir = os.path.join(bpy.path.abspath("//"), "scripts")

if file_dir not in sys.path:
    sys.path.append(file_dir)

import delaunay_helper as dhelp
import general_helper as ghelp

SHOW_PLOT = True


def define_units(scale_length: float = 1.0, lenght_unit: str = "METERS"):
    """
    Sets the unit settings for the current Blender scene to metric, with specified scale
    and unit of length.

    This function updates the unit settings of the current Blender scene to use the
    metric system, and allows for setting a specific scale length and unit of length.
    The scale length is a multiplier that adjusts the size of the 3D space in Blender,
    and the unit of length specifies the name of the units being used (e.g., meters).

    Args:
        scale_length (float): The scale length to set in Blender. Defaults to 1.0.
        length_unit (str): The unit of length to use (e.g., "METERS", "MILLIMETERS").
            Defaults to "METERS".
    """

    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = scale_length
    bpy.context.scene.unit_settings.length_unit = lenght_unit


def compute_mesh(sample_path: str, properties: dict):
    """
    Compute a Delaunay mesh based on the data from a labeled document JSON file.

    Given the path of a sample JSON file, this function reads the file to obtain the
    data, computes a grid, and generates a Delaunay mesh from the vertices obtained from
    both the document points and the grid. If the global variable `SHOW_PLOT` is set to
    True, it also plots the resulting Delaunay mesh.

    Args:
        sample_path (str): The name of the sample JSON file containing the (clean)
            template data.
        properties (dict): The JSON dict with configuration values.

    Returns:
        tuple: A tuple containing:
            - vertices (np.array): The array of vertices used to compute the Delaunay
            mesh (it contains the document vertices + the grid vertices).
            - delaunay_mesh (Delaunay): The computed Delaunay mesh.
    """

    json_info = dhelp.read_json(name=sample_path)
    document_points = dhelp.get_bboxes_as_points(form=json_info["form"])
    grid = dhelp.compute_grid(properties=properties)
    vertices = np.concatenate((document_points, grid), axis=0)
    delaunay_mesh = Delaunay(vertices)
    if SHOW_PLOT:
        dhelp.show_plot(
            vertices=vertices,
            default_grid=grid,
            mesh=delaunay_mesh,
            name=sample_path[:-5],
        )

    return vertices, delaunay_mesh


def create_mesh_object(vertices: np.array, mesh: Delaunay, properties: dict):
    """
    Create a new Blender mesh using a Delaunay mesh object and its vertices.

    This function initiates a new mesh data block and object in Blender, links the mesh
    object to the current collection, and sets it as the active and selected object.
    It processes the vertices and faces to format them correctly for Blender, fills the
    mesh data block with this information, and updates the mesh and the scene to reflect
    these changes. The function finally unwraps the UV map of the object.

    Args:
        vertices (np.array): The array of 2D vertices used to create the mesh.
        mesh (Delaunay): The Delaunay mesh computed from the vertices.
        properties (dict): The JSON dict with configuration values.

    Note:
        - The vertices should be 2D, and they will be converted to 3D by adding a
        z-dimension of 0.
    """

    # Create a new mesh data block and object
    mesh_data = bpy.data.meshes.new(name="Document Mesh")
    mesh_obj = bpy.data.objects.new("Document", mesh_data)

    # Link the mesh object to the current collection
    bpy.context.collection.objects.link(mesh_obj)

    # Ensure the mesh object is selected and active
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    # Add the z dimension to the 2D vertices and format them
    three_d_vertices = [tuple(np.append(vertice, 0)) for vertice in vertices]
    three_d_vertices = dhelp.pixel_to_m(vector=three_d_vertices, properties=properties)
    three_d_vertices = dhelp.list_to_tuple_items(three_d_vertices)

    # Define faces (clockwise lists of vertex indices delimiting them) and format them
    faces = mesh.simplices
    faces = dhelp.list_to_tuple_items(faces)

    # Fill the mesh data block with the vertices and faces
    mesh_data.from_pydata(three_d_vertices, [], faces)
    mesh_data.update()

    # Update the scene
    bpy.context.view_layer.update()
    create_uv_map()


def create_uv_map():
    """
    Unwraps the UV map of the currently active object in Blender.

    The angle-based unwrapping method tries to minimize texture stretching, making it
    suitable for most generic models. A small margin is set for separating the UV is-
    lands to avoid texture bleeding.
    """

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)
    bpy.ops.object.mode_set(mode="OBJECT")


def apply_texture(document: str, paper: str):
    """
    Applies a texture to the active object by creating a new material and
    setting up a node tree to handle the texture mapping, mixing, and shading.

    Args:
    document (str): The file path of the document texture image.
    paper (str): The file path of the paper texture image.
    """

    # TODO Include normals map
    # Get the active object
    obj = bpy.context.active_object

    # Create a new material
    mat = bpy.data.materials.new(name="Document_Material")

    # Ensure node use is enabled
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create necessary nodes
    doc_texture_node = nodes.new(type="ShaderNodeTexImage")
    paper_texture_node = nodes.new(type="ShaderNodeTexImage")
    mix_rgb_node = nodes.new(type="ShaderNodeMixRGB")
    mix_rgb_node.inputs[0].default_value = 1.0
    mapping_node = nodes.new(type="ShaderNodeMapping")
    mapping_node.inputs["Scale"].default_value[0] = 1.4
    coord_node = nodes.new(type="ShaderNodeTexCoord")
    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # Set node locations to prevent overlapping
    doc_texture_node.location = (-300, 300)
    paper_texture_node.location = (-300, 0)
    mix_rgb_node.location = (200, 200)
    mapping_node.location = (-600, 300)
    coord_node.location = (-800, 300)
    principled_node.location = (500, 300)
    output_node.location = (900, 300)

    # Load images into the texture node
    doc_image = bpy.data.images.load(document)
    doc_texture_node.image = doc_image

    paper_image = bpy.data.images.load(paper)
    paper_texture_node.image = paper_image

    mix_rgb_node.blend_type = "MULTIPLY"

    # Connect the nodes
    links = mat.node_tree.links
    links.new(coord_node.outputs["UV"], mapping_node.inputs["Vector"])
    links.new(mapping_node.outputs["Vector"], doc_texture_node.inputs["Vector"])
    links.new(doc_texture_node.outputs["Color"], mix_rgb_node.inputs["Color1"])
    links.new(paper_texture_node.outputs["Color"], mix_rgb_node.inputs["Color2"])
    links.new(mix_rgb_node.outputs["Color"], principled_node.inputs["Base Color"])
    links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])

    # Assign the material to the object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def create_background(texture_path: str, normals_path: str, back_data: dict):
    """
    Creates a background plane in the Blender scene and applies a texture and normal map
    to it.

    This function creates a plane named "Background" in the Blender scene if it doesn't
    exist already. It then creates a new material named "Background_Material" and sets
    up nodes to use the specified texture and normal map files to create a textured sur-
    face on the plane.

    Args:
        texture_path (str): The file path to the texture image.
        normals_path (str): The file path to the normal map image.
        data (dict): A dictionary containing info about the position or scale of the
        plane background.
    """

    # Create plane
    if not bpy.data.objects.get("Plane"):
        bpy.ops.mesh.primitive_plane_add()
    plane = bpy.data.objects["Plane"]
    plane.scale = (back_data["scale_x"], back_data["scale_y"], 1)
    plane.location = (back_data["pos_x"], back_data["pos_y"], back_data["pos_z"])

    # New background material
    mat = bpy.data.materials.new(name="Background_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create necessary nodes
    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    texture_node = nodes.new(type="ShaderNodeTexImage")
    normals_node = nodes.new(type="ShaderNodeNormalMap")
    normals_node.inputs["Strength"].default_value = 0.5
    normals_image_node = nodes.new(type="ShaderNodeTexImage")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # Set node locations to prevent overlapping
    principled_node.location = (200, 300)
    texture_node.location = (-300, 300)
    normals_node.location = (0, -50)
    normals_image_node.location = (-300, -50)
    output_node.location = (500, 300)

    texture_node.image = bpy.data.images.load(texture_path)
    normals_image_node.image = bpy.data.images.load(normals_path)

    # Link nodes
    links.new(texture_node.outputs["Color"], principled_node.inputs["Base Color"])
    links.new(normals_image_node.outputs["Color"], normals_node.inputs["Color"])
    links.new(normals_node.outputs["Normal"], principled_node.inputs["Normal"])
    links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])

    # Asign material
    if plane.data.materials:
        plane.data.materials[0] = mat
    else:
        plane.data.materials.append(mat)


def compute_pos(pos_data: dict, distribution: str = "normal"):
    """
    Computes position coordinates based on specified distribution parameters.

    This function generates a position vector based on the provided data for each di-
    mension (x, y, z). Currently, it only supports a normal distribution for generating
    these values, utilizing a helper function `ghelp.get_value_from_normal` to obtain a
    value from a normal distribution based on the average and max_delta values for each
    dimension.

    Args:
        pos_data (dict): A dictionary containing the average position ('x', 'y', 'z')
            and the maximum deviation ('dx', 'dy', 'dz') for each dimension.
        distribution (str): The type of distribution to use for generating position
            values. Defaults to "normal".

    Returns:
        list: A list containing the position values for each dimension [x, y, z].

    Note:
        - Only the "normal" distribution is currently implemented. Any other distribu-
          tion type will result in a printed message and no change to the position value
          for that dimension.
    """
    dims = ["x", "y", "z"]
    pos = []

    for dim in dims:
        pos_i = pos_data[dim]
        pos_di = pos_data["".join(["d", dim])]
        if distribution == "normal":
            pos_i = ghelp.get_value_from_normal(avg=pos_i, max_delta=pos_di)
        else:
            print(f"{distribution} distribution is not implemented")
        pos.append(pos_i)

    return pos


def config_camera(camera_data: dict):
    """
    Positions and orients a camera in the Blender scene based on specified style.

    This function ensures a camera object exists in the Blender scene, then computes
    the camera's location and rotation based on the input style. The camera_data
    dict should contain nested dictionaries describing the position and rotation
    parameters for the camera, including average values and maximum deviations
    for each parameter. The function uses a helper method `ghelp.get_value_from_normal`
    to generate a value from a normal distribution defined by the average value and
    the max, min values.

    Args:
        camera_data (dict): A dictionary containing the following nested structure:

            {
                "pos_meters": {
                    "x": float, "dx": float,
                    "y": float, "dy": float,
                    "z": float, "dz": float
                },
                "rot_degrees": {
                    "x": float, "dx": float,
                    "y": float, "dy": float,
                    "z": float, "dz": float
                }
            }
    """

    # Make sure a camera object exists
    if not bpy.data.objects.get("Camera"):
        bpy.ops.object.camera_add()
    camera = bpy.data.objects["Camera"]

    # Compute camera location
    pos = compute_pos(pos_data=camera_data["pos_meters"])

    # Compute camera rotation
    rot_x = camera_data["rot_degrees"]["x"]
    rot_dx = camera_data["rot_degrees"]["dx"]
    rot_x = math.radians(ghelp.get_value_from_normal(avg=rot_x, max_delta=rot_dx))

    rot_y = camera_data["rot_degrees"]["y"]
    rot_dy = camera_data["rot_degrees"]["dy"]
    rot_y = math.radians(ghelp.get_value_from_normal(avg=rot_y, max_delta=rot_dy))

    rot_z = camera_data["rot_degrees"]["z"]
    rot_dz = camera_data["rot_degrees"]["dz"]
    rot_z = math.radians(ghelp.get_value_from_normal(avg=rot_z, max_delta=rot_dz))

    # Set location and rotation
    camera.location = (pos[0], pos[1], pos[2])
    camera.rotation_euler = (rot_x, rot_y, rot_z)


def config_lights(lights_data: dict):
    """
    Create and configure a number of light objects in the Blender scene based on the
    provided data.

    Args:
        lights_data (dict): A dictionary containing the data to configure lights.
    """

    # TODO Random light style
    n_lights = lights_data["number"]

    for light_i in range(n_lights):
        # Create a new light datablock
        light = bpy.data.lights.new(name=f"Light_{light_i}", type="POINT")
        light.energy = lights_data["power"]
        light.diffuse_factor = lights_data["diffuse"]
        light.specular_factor = lights_data["specular"]
        light.shadow_soft_size = lights_data["radius"]

        # Create a new light object and link it to the collection
        light_object = bpy.data.objects.new(f"Light_{light_i}", object_data=light)
        bpy.context.collection.objects.link(light_object)

        # Set light location
        light_pos = compute_pos(lights_data["pos_meters"])
        light_object.location = (light_pos[0], light_pos[1], light_pos[2])

        # Light color
        color_info = lights_data["color"]
        rgb_color = colorsys.hsv_to_rgb(
            color_info["hue"], color_info["saturation"], color_info["value"]
        )
        light.color = rgb_color

        # Contact shadows
        light.use_contact_shadow = True
        light.shadow_buffer_bias = 0.001


def render_scene(dst_folder: str, name: str, img_dims: dict):
    """
    Render the current scene in Blender using the EEVEE render engine,
    and save the rendered image to the specified destination folder with the given name.

    Args:
        dst_folder (str): The directory path where the rendered image will be saved.
        name (str): The name of the rendered image file (including file extension).
    """
    camera = bpy.data.objects.get("Camera")
    if camera:
        bpy.context.scene.camera = camera
    else:
        raise RuntimeError("No camera found in the scene")

    bpy.context.scene.render.resolution_x = img_dims["width"]
    bpy.context.scene.render.resolution_y = img_dims["height"]

    bpy.context.scene.render.filepath = os.path.join(dst_folder, name)
    bpy.ops.render.render(write_still=True)


def modify_mesh(mesh_data: dict):
    """
    Modify a mesh object in Blender by subdividing it and smoothing the vertices.

    This function takes a dictionary containing subdivision and vertex smoothing
    information, selects the mesh object named "Document" in the current Blender scene,
    and applies the specified modifications.

    Args:
        mesh_data (dict): A dictionary containing the following keys:

            {
                'subdivision': {
                    'cuts': float
                    'fractal': float
                },
                'vert_smooth': {
                    'repeat': float
                    'factor': float
                }
            }

    """
    bpy.data.objects["Document"].select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")

    subdiv = mesh_data["subdivision"]
    smooth = mesh_data["vert_smooth"]

    # Mesh modifications
    bpy.ops.mesh.subdivide(
        number_cuts=subdiv["cuts"],
        fractal=subdiv["fractal"],
        seed=random.randint(1, 1000),
    )
    bpy.ops.mesh.vertices_smooth(repeat=smooth["repeat"], factor=smooth["factor"])
    bpy.ops.object.mode_set(mode="OBJECT")


if __name__ == "__main__":
    # Config
    define_units()

    # Load requirements
    requirements = ghelp.load_requirements(root=bpy.path.abspath("//"))

    # Load properties
    properties = dhelp.load_properties(root=bpy.path.abspath("//"))

    # Load assets
    sample_name = os.path.join(
        Path(bpy.path.abspath("//")).parent.parent, "data", "original", "sample.json"
    )
    print(sample_name)
    document_texture = os.path.join(
        Path(bpy.path.abspath("//")).parent.parent, "data", "original", "sample.png"
    )
    paper_texture = os.path.join(
        bpy.path.abspath("//"), "assets", "textures", "papers", "paper.png"
    )
    background_texture = os.path.join(
        bpy.path.abspath("//"),
        "assets",
        "textures",
        "backgrounds",
        "white_oak",
        "texture.png",
    )
    background_normal = os.path.join(
        bpy.path.abspath("//"),
        "assets",
        "textures",
        "backgrounds",
        "white_oak",
        "normals.png",
    )

    # Mesh and object
    vertices, delaunay_mesh = compute_mesh(
        sample_path=sample_name, properties=properties
    )
    create_mesh_object(vertices=vertices, mesh=delaunay_mesh, properties=properties)

    # Textures
    apply_texture(document=document_texture, paper=paper_texture)

    # Modify mesh
    mesh_data = properties["blender"]["document_mesh_mod"]
    modify_mesh(mesh_data=mesh_data)

    # Set background
    background_data = properties["blender"]["common"]["background"]
    create_background(
        texture_path=background_texture,
        normals_path=background_normal,
        back_data=background_data,
    )

    # Set light
    lights_data = properties["blender"][requirements["styles"][0]]["lights"]
    config_lights(lights_data=lights_data)

    # Set camera
    camera_data = properties["blender"][requirements["styles"][0]]["camera"]
    config_camera(camera_data=camera_data)

    # Render scene
    dst_folder = os.path.join(
        Path(bpy.path.abspath("//")).parent.parent, "data", "modified"
    )
    render_scene(
        dst_folder=dst_folder, name="test.png", img_dims=requirements["img_output"]
    )
