import bpy
import numpy as np
from scipy.spatial import Delaunay
import sys
import os
import math
import colorsys
from pathlib import Path
import random
import bpy_extras
import cv2
import copy
import pandas as pd
import tqdm
import bmesh
from mathutils import Vector
from datetime import datetime


file_dir = os.path.join(bpy.path.abspath("//"), "scripts")

if file_dir not in sys.path:
    sys.path.append(file_dir)

import delaunay_helper as dhelp
import general_helper as ghelp

SHOW_PLOT = False


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
            - n_document_points (int): Number of document vertices (word's bboxes)
                used in the computation.
            - vertices (np.array): The array of vertices used to compute the Delaunay
            mesh (it contains the document vertices + the grid vertices).
            - delaunay_mesh (Delaunay): The computed Delaunay mesh.
    """
    grid_sampling_x = properties["delaunay"]["grid_sampling"]["x"]
    grid_sampling_y = properties["delaunay"]["grid_sampling"]["y"]
    json_info = dhelp.read_json(name=sample_path)
    document_points = dhelp.get_bboxes_as_points(
        form=json_info["form"], properties=properties
    )

    grid_vertices, x_step, y_step, grid_marks = dhelp.compute_grid(
        properties=properties, sampling_x=grid_sampling_x, sampling_y=grid_sampling_y
    )
    mesh_steps = {"x": x_step, "y": y_step}
    delaunay_mesh = Delaunay(grid_vertices)
    if SHOW_PLOT:
        dhelp.show_plot(
            vertices=grid_vertices,
            mesh=delaunay_mesh,
            name=sample_path[:-5],
        )

    return (
        grid_vertices,
        delaunay_mesh,
        mesh_steps,
        grid_marks,
        document_points,
    )


def create_mesh_object(
    vertices: np.array,
    mesh: Delaunay,
    properties: dict,
    mesh_name: str,
):
    """
    Create a new Blender mesh using a Delaunay mesh object and its vertices.

    This function initiates a new mesh data block and object in Blender, links the mesh
    object to the current collection, and sets it as the active and selected object.
    It processes the vertices and faces to format them correctly for Blender, fills the
    mesh data block with this information, and updates the mesh and the scene to reflect
    these changes. The function finally unwraps the UV map of the object.

    Args:
        n_bboxes_vertices (int): Number of (exclusively) words' bboxes vertices.
        vertices (np.array): The array of 2D vertices used to create the mesh.
        mesh (Delaunay): The Delaunay mesh computed from the vertices.
        properties (dict): The JSON dict with configuration values.

    Note:
        - The vertices should be 2D, and they will be converted to 3D by adding a
        z-dimension of 0.
    """

    # Create a new mesh data block and object
    mesh_data = bpy.data.meshes.new(name=mesh_name)
    mesh_obj = bpy.data.objects.new("Document", mesh_data)

    # Link the mesh object to the current collection
    bpy.context.collection.objects.link(mesh_obj)

    # Ensure the mesh object is selected and active
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    # Add the z dimension to the 2D vertices and format them
    three_d_vertices = [tuple(np.append(vertex, 0)) for vertex in vertices]
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


def create_printer_stains(nodes, links, mix_shader):

    # Printer dots
    dots_coord_node = nodes.new(type="ShaderNodeTexCoord")
    dots_mapping_node = nodes.new(type="ShaderNodeMapping")
    dots_voronoi_node = nodes.new(type="ShaderNodeTexVoronoi")
    dots_voronoi_node.inputs["Scale"].default_value = 500
    dots_voronoi_node.inputs["Randomness"].default_value = 1
    dots_color_ramp = nodes.new(type="ShaderNodeValToRGB")
    dots_color_ramp.color_ramp.interpolation = "LINEAR"
    dots_color_ramp.color_ramp.elements.remove(dots_color_ramp.color_ramp.elements[1])
    dots_color_ramp_colors_positions = [
        (0.964, (0, 0, 0, 1)),
        (0.982, (1, 1, 1, 1)),
    ]

    for pos, color in dots_color_ramp_colors_positions:
        element = dots_color_ramp.color_ramp.elements.new(position=pos)
        element.color = color

    # Printer lines
    lines_coord_node = nodes.new(type="ShaderNodeTexCoord")
    lines_mapping_node = nodes.new(type="ShaderNodeMapping")
    lines_mapping_node.inputs["Scale"].default_value[0] = -0.2
    lines_mapping_node.inputs["Scale"].default_value[1] = -2000
    lines_voronoi_node = nodes.new(type="ShaderNodeTexVoronoi")
    lines_voronoi_node.inputs["Scale"].default_value = 5
    lines_voronoi_node.inputs["Randomness"].default_value = 1
    lines_color_ramp = nodes.new(type="ShaderNodeValToRGB")
    lines_color_ramp.color_ramp.interpolation = "LINEAR"
    lines_color_ramp.color_ramp.elements.remove(lines_color_ramp.color_ramp.elements[1])
    lines_color_ramp_colors_positions = [
        (0.627, (0, 0, 0, 1)),
        (0.673, (1, 1, 1, 1)),
    ]

    for pos, color in lines_color_ramp_colors_positions:
        element = lines_color_ramp.color_ramp.elements.new(position=pos)
        element.color = color

    # Printer lines vertical bands mask
    bands_coord_node = nodes.new(type="ShaderNodeTexCoord")
    bands_mapping_node = nodes.new(type="ShaderNodeMapping")
    bands_mapping_node.inputs["Scale"].default_value[0] = 1.415
    bands_gradient_texture = nodes.new(type="ShaderNodeTexGradient")
    bands_gradient_texture.gradient_type = "LINEAR"
    bands_color_ramp = nodes.new(type="ShaderNodeValToRGB")
    bands_color_ramp.color_ramp.interpolation = "EASE"
    bands_color_ramp.color_ramp.elements.remove(bands_color_ramp.color_ramp.elements[1])
    bands_color_ramp_colors_positions = [
        (0.1, (0, 0, 0, 1)),
        (0.25, (0, 0, 0, 1)),
        (0.40, (0, 0, 0, 1)),
        (0.55, (0, 0, 0, 1)),
        (0.70, (0, 0, 0, 1)),
        (0.85, (0, 0, 0, 1)),
        (0.9, (0, 0, 0, 1)),
    ]

    for _ in range(random.randint(0, 3)):
        bands_color_ramp_colors_positions.extend([(random.uniform(0, 1), (1, 1, 1, 1))])

    for pos, color in bands_color_ramp_colors_positions:
        element = bands_color_ramp.color_ramp.elements.new(position=pos)
        element.color = color

    # Mixer: Printer lines and bands
    mixer_lines_bands = nodes.new(type="ShaderNodeMixRGB")
    mixer_lines_bands.blend_type = "MULTIPLY"
    mixer_lines_bands.inputs[0].default_value = 1.0

    # Mixer: Dots and lines
    mixer_dots_lines = nodes.new(type="ShaderNodeMixRGB")
    mixer_dots_lines.blend_type = "ADD"
    mixer_dots_lines.inputs[0].default_value = 1.0

    # Set node locations to prevent overlapping: Printer dots
    dots_coord_node.location = (-800, 1500)
    dots_mapping_node.location = (-600, 1500)
    dots_voronoi_node.location = (-300, 1500)
    dots_color_ramp.location = (0, 1500)
    mixer_dots_lines.location = (500, 1250)

    # Set node locations to prevent overlapping: Printer lines
    lines_coord_node.location = (-800, 1000)
    lines_mapping_node.location = (-600, 1000)
    lines_voronoi_node.location = (-300, 1000)
    lines_color_ramp.location = (0, 1000)
    mixer_lines_bands.location = (300, 750)

    # Set node locations to prevent overlapping: Printer bands
    bands_coord_node.location = (-800, 600)
    bands_mapping_node.location = (-600, 600)
    bands_gradient_texture.location = (-300, 600)
    bands_color_ramp.location = (0, 600)

    # Connect the nodes: Printer dots
    links.new(dots_coord_node.outputs["UV"], dots_mapping_node.inputs["Vector"])
    links.new(dots_mapping_node.outputs["Vector"], dots_voronoi_node.inputs["Vector"])
    links.new(dots_voronoi_node.outputs["Color"], dots_color_ramp.inputs["Fac"])
    links.new(dots_color_ramp.outputs["Color"], mixer_dots_lines.inputs[1])

    # Connect the nodes: Printer lines
    links.new(lines_coord_node.outputs["UV"], lines_mapping_node.inputs["Vector"])
    links.new(lines_mapping_node.outputs["Vector"], lines_voronoi_node.inputs["Vector"])
    links.new(lines_voronoi_node.outputs["Color"], lines_color_ramp.inputs["Fac"])
    links.new(lines_color_ramp.outputs["Color"], mixer_lines_bands.inputs[1])

    # Connect the nodes: Printer bands
    links.new(bands_coord_node.outputs["UV"], bands_mapping_node.inputs["Vector"])
    links.new(
        bands_mapping_node.outputs["Vector"], bands_gradient_texture.inputs["Vector"]
    )
    links.new(bands_gradient_texture.outputs["Color"], bands_color_ramp.inputs["Fac"])
    links.new(bands_color_ramp.outputs["Color"], mixer_lines_bands.inputs[2])
    links.new(mixer_lines_bands.outputs["Color"], mixer_dots_lines.inputs[2])
    links.new(mixer_dots_lines.outputs["Color"], mix_shader.inputs["Fac"])


def apply_document_texture(document: str, mods_dict: dict):
    """
    Applies a texture to the active object by creating a new material and
    setting up a node tree to handle the texture mapping, mixing, and shading.

    Args:
    document (str): The file path of the document texture image.
    mods_dict (dict): What to modify.
    """
    paper_texture = "".join([mods_dict["paper_texture"], ".png"])
    paper_texture_path = os.path.join(
        bpy.path.abspath("//"), "assets", "textures", "papers", paper_texture
    )

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
    mapping_node.inputs["Location"].default_value[0] = -0.001
    mapping_node.inputs["Scale"].default_value[0] = 1.415
    mapping_node.inputs["Scale"].default_value[1] = 1
    mapping_node.inputs["Rotation"].default_value[2] = math.radians(180)
    coord_node = nodes.new(type="ShaderNodeTexCoord")
    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled_node.inputs["Specular"].default_value = 0.1
    principled_node.inputs["Roughness"].default_value = 1.0
    principled_node.inputs["Metallic"].default_value = 1.0
    principled_node_printer = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled_node_printer.inputs["Base Color"].default_value = (
        0.092,
        0.092,
        0.092,
        1,
    )
    principled_node_printer.inputs["Specular"].default_value = 0.1
    principled_node_printer.inputs["Roughness"].default_value = 1.0
    principled_node_printer.inputs["Metallic"].default_value = 1.0
    mix_shader = nodes.new(type="ShaderNodeMixShader")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # Set node locations to prevent overlapping: Document texture
    doc_texture_node.location = (-300, 300)
    paper_texture_node.location = (-300, 0)
    mix_rgb_node.location = (200, 200)
    mapping_node.location = (-600, 300)
    coord_node.location = (-800, 300)
    principled_node.location = (500, 300)
    output_node.location = (900, 300)

    # Set node locations to prevent overlapping: Second Principle BSDF
    principled_node_printer.location = (500, -400)
    mix_shader.location = (750, 300)

    # Load images into the texture node
    doc_image = bpy.data.images.load(document)
    doc_texture_node.image = doc_image

    paper_image = bpy.data.images.load(paper_texture_path)
    paper_texture_node.image = paper_image

    mix_rgb_node.blend_type = "MULTIPLY"

    # Connect the nodes: Document texture
    links = mat.node_tree.links
    links.new(coord_node.outputs["UV"], mapping_node.inputs["Vector"])
    links.new(mapping_node.outputs["Vector"], doc_texture_node.inputs["Vector"])
    links.new(doc_texture_node.outputs["Color"], mix_rgb_node.inputs["Color1"])
    links.new(paper_texture_node.outputs["Color"], mix_rgb_node.inputs["Color2"])
    links.new(mix_rgb_node.outputs["Color"], principled_node.inputs["Base Color"])

    # Printer stain nodes
    printer_stains = mods_dict["printer_stains"]
    if printer_stains and printer_stains != "N/A" and printer_stains != "False":
        create_printer_stains(nodes, links, mix_shader)

    links.new(principled_node.outputs["BSDF"], mix_shader.inputs[1])
    links.new(principled_node_printer.outputs["BSDF"], mix_shader.inputs[2])

    links.new(mix_shader.outputs["Shader"], output_node.inputs["Surface"])

    # Assign the material to the object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def create_background(back_data: dict, mods_dict: dict):
    """
    Creates a background plane in the Blender scene and applies a texture and normal map
    to it.

    This function creates a plane named "Background" in the Blender scene if it doesn't
    exist already. It then creates a new material named "Background_Material" and sets
    up nodes to use the specified texture and normal map files to create a textured sur-
    face on the plane.

    Args:
        data (dict): A dictionary containing info about the position or scale of the
        plane background.
        mods_dict (dict): The dictionary with the info about what to modify according to
        the blueprint
    """

    # Create plane
    if not bpy.data.objects.get("Plane"):
        bpy.ops.mesh.primitive_plane_add()
    plane = bpy.data.objects["Plane"]
    plane.scale = (back_data["scale_x"], back_data["scale_y"], 1)
    plane.location = (back_data["pos_x"], back_data["pos_y"], back_data["pos_z"])
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # New background material
    mat = bpy.data.materials.new(name="Background_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create necessary nodes and adjust values
    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled_node.inputs["Specular"].default_value = 0.25
    principled_node.inputs["Roughness"].default_value = 0.25
    principled_node.inputs["Metallic"].default_value = 0.8
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

    background_mat = mods_dict["background_material"]
    background_mat_root = os.path.join(
        bpy.path.abspath("//"),
        "assets",
        "textures",
        "backgrounds",
        background_mat,
    )
    texture_path = os.path.join(
        background_mat_root,
        "texture.png",
    )
    normals_path = os.path.join(
        background_mat_root,
        "normals.png",
    )
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

    # Create a new camera
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = "Document_camera"

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
    Render the current scene in Blender using the EEVEE render engine, save the rendered
    image to the specified destination folder with the given name, and return the rende-
    red image as a NumPy array

    Args:
        dst_folder (str): The directory path where the rendered image will be saved.
        name (str): The name of the rendered image file (including file extension).
    Returns:
        np.array: The rendered image as an cv2 image.
    """

    camera = bpy.data.objects.get("Document_camera")
    if camera:
        bpy.context.scene.camera = camera
    else:
        raise RuntimeError("No camera found in the scene")

    # Dimensions swapped to get a vertical orientation for the output (image and labels)
    # bpy.context.scene.render.resolution_x = img_dims["height"]
    # bpy.context.scene.render.resolution_y = img_dims["width"]

    rendered_img_path = os.path.join(dst_folder, name)
    bpy.context.scene.render.filepath = rendered_img_path
    bpy.ops.render.render(write_still=True)

    rendered_img = cv2.imread(rendered_img_path)

    return rendered_img


def get_vertices_id_from_group(object_name: str, group_name: str):
    """
    Retrieve the indices of the vertices belonging to an specific vertex group of a
    given object.

    This function activates the object, switches to edit mode, and selects the vertices
    belonging to an specific vertex group. It then retrieves the indices of these verti-
    ces and returns them in a list.

    Args:
        object_name (str): The name of the object whose vertex group is being queried.
        group_name (str): The name of the vertex group within the object.

    Returns:
        list: A list of integers representing the indices of the vertices belonging to
        the specified vertex group.
    """

    doc_object = bpy.data.objects[object_name]
    if group_name not in doc_object.vertex_groups:
        raise ValueError(f"Vertex group {group_name} not found in object {object_name}")

    vertex_group = doc_object.vertex_groups[group_name]
    vertices_in_group = []

    for v in doc_object.data.vertices:
        try:
            if vertex_group.weight(v.index) > 0:
                vertices_in_group.append(v.index)

        # Vertex not in group, skip to next vertex
        except RuntimeError:
            pass

    return vertices_in_group


def retrieve_bboxes_pixel_points(img: np.array, mesh_name: str, mapped_ids):
    """
    Retrieve the pixel coordinates of vertices belonging to a vertex group (words
    bboxes) and draw these points on the rendered image.

    This method identifies the pixel coordinates of vertices from a specified vertex
    group in a Blender object. It uses the Blender camera settings to convert the 3D
    world coordinates of these vertices to 2D pixel coordinates. It then draws these
    points on the input image and returns both the list of pixel coordinates and the
    modified image.

    Args:
        img (np.array): A numpy array representing the (original) rendered image.

    Returns:
        tuple: A tuple containing:
            - rectangles (list): A list of lists (bbounding boxes with four points deli-
                                 miting every word bbox).
            - img (np.array): The input image with the vertices drawn on it as points.
    """

    camera_name = "Document_camera"
    obj_name = "Document"
    mesh_name = mesh_name

    camera = bpy.data.objects[camera_name]
    doc_object = bpy.data.objects[obj_name]
    world_location = doc_object.location

    mesh = bpy.data.meshes[mesh_name]
    vertices_px = []

    # Bboxes vertices: transform 3D world coordinates to 2D pixel coordinates
    for word in mapped_ids:
        for corner in word:
            vertex_id = corner[1]
            world_location = mesh.vertices[vertex_id].co

            render_coordinates = bpy_extras.object_utils.world_to_camera_view(
                bpy.context.scene, camera, world_location
            )

            point_x = int(render_coordinates.x * bpy.context.scene.render.resolution_x)
            point_y = int(
                (1 - render_coordinates.y) * bpy.context.scene.render.resolution_y
            )

            point = (point_x, point_y)
            vertices_px.append(point)

    all_vertices_indices = []
    all_vertices_px = []
    for index in range(len(mesh.vertices)):
        all_vertices_indices.append(index)
        world_location = mesh.vertices[index].co

        render_coordinates = bpy_extras.object_utils.world_to_camera_view(
            bpy.context.scene, camera, world_location
        )

        point_x = int(render_coordinates.x * bpy.context.scene.render.resolution_x)
        point_y = int(
            (1 - render_coordinates.y) * bpy.context.scene.render.resolution_y
        )

        point = (point_x, point_y)
        all_vertices_px.append(point)

    # Draw the bboxes over the rendered img for visual check
    rectangles = []

    # One bbox is defined by 4 points
    for i in range(0, len(vertices_px), 4):
        rect_points = np.array(list(vertices_px[i : i + 4]), dtype=np.int32)
        rectangles.append(rect_points)

    img = ghelp.draw_mesh_points(img=img, pts=all_vertices_px, color=(255, 0, 0), r=2)
    img = ghelp.draw_bboxes(img=img, rectangles=rectangles, color=(0, 255, 0))

    return rectangles, img


def get_blueprint():
    """
    Retrieves the dataset blueprint from a CSV file.

    This method builds the path to the CSV blueprint ('dataset_blueprint.csv'),
    located in the 'dashboard' directory. It then reads the CSV file into a pandas
    DataFrame.

    Returns:
        tuple: A tuple containing two elements:
            - pandas.DataFrame: The DataFrame created from the CSV file.
            - str: The file path to the CSV file.
    """

    blueprint_path = os.path.join(
        Path(__file__).resolve().parents[2], "dashboard", "dataset_blueprint.csv"
    )
    blueprint_df = pd.read_csv(blueprint_path)

    return blueprint_df, blueprint_path


def get_sample_paths_and_names(sample_name: str):
    language = sample_name.split("_")[0]
    school = sample_name.split("_")[1]

    root = os.path.join(
        Path(bpy.path.abspath("//")).parent.parent,
        "data",
        "original",
        language,
        school,
        "dataset_output",
    )

    sample_img_path = os.path.join(root, "images", "".join([sample_name, ".png"]))
    sample_labels_path = os.path.join(
        root, "annotations", "".join([sample_name, ".json"])
    )

    dst_folder = os.path.join(
        Path(bpy.path.abspath("//")).parent.parent,
        "data",
        "modified",
        language,
        school,
        "dataset_output",
    )

    mod_sample_name = "".join([sample_name, "_blender_mod.png"])
    mod_labels_name = "".join([sample_name, "_blender_mod.json"])
    bboxes_sample_name = "".join([sample_name, "_blender_mod_bbox_debug.png"])
    segments_sample_name = "".join([sample_name, "_blender_mod_segment_debug.png"])

    return (
        sample_img_path,
        sample_labels_path,
        dst_folder,
        mod_sample_name,
        mod_labels_name,
        bboxes_sample_name,
        segments_sample_name,
    )


def move_doc_uppward(z: float):
    obj = bpy.data.objects["Document"]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)

    # Transform mesh triangles to quads where possible
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.tris_convert_to_quads()
    bpy.ops.object.mode_set(mode="OBJECT")

    # Solidify the object
    # solidify_modifier = obj.modifiers.new(name="DocumentSolidifier", type="SOLIDIFY")
    # solidify_modifier.thickness = 0.001

    # Change its position
    obj.location.z = z


def compute_skewness(obj: bpy.data.objects):
    print("Computing mesh skewness")

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    mesh.verts.ensure_lookup_table()
    mesh.edges.ensure_lookup_table()
    mesh.faces.ensure_lookup_table()

    smoothness_scores = []

    for vert in mesh.verts:
        adj_faces = vert.link_faces
        if not adj_faces:
            continue
        avg_normal = sum((face.normal for face in adj_faces), Vector()) / len(adj_faces)

        deviation = (vert.normal - avg_normal).length
        smoothness_scores.append(deviation)

    mesh.free()

    average_dev_skewness = (
        sum(smoothness_scores) / len(smoothness_scores) if smoothness_scores else 0
    )

    print(f"Mesh skewness: {average_dev_skewness}")
    return average_dev_skewness


def run_cloth_sim():
    obj = bpy.data.objects.get("Document")

    # Make other objects interact
    for other_obj in bpy.context.scene.objects:
        if other_obj.type == "MESH" and other_obj != obj:
            if "Collision" not in other_obj.modifiers:
                collision_modifier = other_obj.modifiers.new(
                    name="Collision", type="COLLISION"
                )

                collision_modifier.settings.thickness_outer = 0.001
                collision_modifier.settings.thickness_inner = 0.05

    # Make sure the object is active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Add the object a cloth modifier
    cloth_modifier = obj.modifiers.new(name="Cloth", type="CLOTH")

    # Set the physical properties:
    cloth_modifier.settings.mass = 0.001
    cloth_modifier.settings.air_damping = 3
    cloth_modifier.settings.bending_model = "LINEAR"

    # Paper mechanical
    cloth_modifier.settings.bending_stiffness = 50
    cloth_modifier.settings.bending_damping = 50

    # Set the object collition distance to other object
    cloth_modifier.collision_settings.distance_min = 0.001

    # Set simulation quality parameters
    cloth_modifier.settings.quality = 10
    cloth_modifier.collision_settings.collision_quality = 10

    # Activate selfcollisions and set the threshold
    # cloth_modifier.collision_settings.use_self_collision = True
    # cloth_modifier.collision_settings.self_distance_min = 0.001

    # Run the simulation step by step to obtain a proper deformed mesh
    bpy.context.scene.frame_end = 75
    for frame in range(1, bpy.context.scene.frame_end + 1):
        bpy.context.scene.frame_set(frame)

    # Apply the cloth modifier
    bpy.context.scene.frame_set(bpy.context.scene.frame_end)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="OBJECT")

    for modifier in obj.modifiers:
        if modifier.type == "CLOTH":
            bpy.ops.object.modifier_apply(modifier=modifier.name)

    # Smooth the mesh
    bpy.ops.object.shade_smooth()

    # Compute skewness
    compute_skewness(obj)


def import_background_object(mods_dict: dict):

    object_name = mods_dict["background_elements"]
    if object_name and object_name != "N/A" and object_name != "False":

        object_file_path = os.path.join(
            bpy.path.abspath("//"),
            "assets",
            "objects",
            "".join([object_name, ".blend"]),
        )

        directory = os.path.join(object_file_path, "Object")

        bpy.ops.wm.append(
            filepath=directory + object_name,
            directory=directory,
            filename=object_name,
        )

        # Select the object
        obj = bpy.data.objects[object_name]
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)

        # Tune pos/rot data
        x = random.uniform(-0.1, 0.2)
        y = random.uniform(-0.1, 0.3)
        z_angle = random.uniform(0, 360)

        # Change its position and rotation
        obj.location = (x, y, 0)
        obj.rotation_euler = (0, 0, math.radians(z_angle))


def get_modifications_dict(blueprint_df: pd.DataFrame, sample: str) -> dict:
    properties = [
        "rendering_style",
        "shadow_casting",
        "printer_stains",
        "background_elements",
        "modify_mesh",
        "background_material",
        "paper_texture",
    ]

    mod_dict = {}

    for sample_property in properties:
        mod_dict[sample_property] = blueprint_df.loc[
            blueprint_df["file_name"] == sample[1], sample_property
        ].iloc[0]

    return mod_dict


def hide_elements_except(object_name: str):

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")

    for obj in bpy.context.scene.objects:
        if obj.name != object_name:
            obj.hide_render = True
        else:
            obj.hide_render = False


def modify_document_mesh(mods_dict: dict):

    if (
        mods_dict["modify_mesh"]
        and mods_dict["modify_mesh"] != "N/A"
        and mods_dict["rendering_style"] != "scanner"
    ):
        move_doc_uppward(z=0.025)
        run_cloth_sim()
        move_doc_uppward(z=0)
        hide_elements_except("Document")

    elif mods_dict["rendering_style"] == "scanner":
        move_doc_uppward(z=0.1)


def approx_bboxes_points_to_grid(
    mesh_steps: dict, grid_marks: dict, document_points: list
):
    words_id_map = []
    bbox_map = []
    word_index = 0

    for index, point in enumerate(document_points):
        coord_x_px = point[0]
        coord_y_px = point[1]
        _, index_column = remap_coordinate(coord_x_px, mesh_steps["x"], grid_marks["x"])
        _, index_row = remap_coordinate(coord_y_px, mesh_steps["y"], grid_marks["y"])

        mapped_index = index_row * len(grid_marks["x"]) + index_column

        # Update the mapping
        bbox_map.append((index, mapped_index))
        word_index += 1

        if word_index == 4:
            words_id_map.append(bbox_map)
            bbox_map = []
            word_index = 0

        elif word_index > 4:
            raise Warning(f"A bounding box is defined by 4 vertices")

    return words_id_map


def remap_coordinate(coord, step, grid_axis_marks):
    division = coord // step
    remainder = coord % step

    if remainder < step / 2:
        index = division
    else:
        index = division + 1
    coord = grid_axis_marks[index]

    return coord, index


def cast_shadow(mods_dict: dict):
    if mods_dict["shadow_casting"] and mods_dict["shadow_casting"] != "N/A":
        object_name = "rigged_male_body"
        object_file_path = os.path.join(
            bpy.path.abspath("//"),
            "assets",
            "shadows",
            "".join([object_name, ".blend"]),
        )

        directory = os.path.join(object_file_path, "Collection")

        bpy.ops.wm.append(
            filepath=directory + object_name,
            directory=directory,
            filename=object_name,
        )

        # Select the object
        obj = bpy.data.objects[object_name]
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)

        # Tune pos/rot data
        x = random.uniform(-0.1, 0.4)
        y = random.uniform(0.8, 1)
        z_angle = random.uniform(-30, 30)

        # Change its position and rotation
        obj.location = (x, y, -0.84)
        obj.rotation_euler = (math.radians(90), 0, math.radians(z_angle))


def set_lighting_syle(mods_dict):
    lighting_style = mods_dict["rendering_style"]

    if lighting_style and lighting_style != "N/A" and lighting_style != "False":
        # Load de HDR image
        hdr_file = "".join([lighting_style, ".hdr"])
        hdr_path = os.path.join(bpy.path.abspath("//"), "assets", "hdr", hdr_file)

        hdr_img = bpy.data.images.load(hdr_path)

        # Config World/Tree Nodes
        world = bpy.context.scene.world
        if not world.use_nodes:
            world.use_nodes = True

        node_tree = world.node_tree
        nodes = node_tree.nodes

        for node in nodes:
            nodes.remove(node)

        # Add Background an Texture Environment Nodes
        bg_node = nodes.new(type="ShaderNodeBackground")
        env_node = nodes.new(type="ShaderNodeTexEnvironment")

        # Add Extra Nodes to Control HDR Orientation
        mapping_node = nodes.new(type="ShaderNodeMapping")
        tex_coord_node = nodes.new(type="ShaderNodeTexCoord")

        # Rotate the HDR
        z_rotation_value = math.radians(random.uniform(0, 360))
        mapping_node.inputs["Rotation"].default_value[2] = z_rotation_value

        # Connect Nodes
        node_tree.links.new(
            tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"]
        )
        node_tree.links.new(mapping_node.outputs["Vector"], env_node.inputs["Vector"])
        node_tree.links.new(env_node.outputs["Color"], bg_node.inputs["Color"])
        node_tree.links.new(
            bg_node.outputs["Background"],
            nodes.new(type="ShaderNodeOutputWorld").inputs["Surface"],
        )

        env_node.image = hdr_img

        bpy.context.scene.eevee.use_ssr = True
        bpy.context.scene.eevee.use_ssr_refraction = True
        bpy.context.view_layer.update()


def modify_samples(samples_to_mod_df: pd.DataFrame, blueprint_df: pd.DataFrame):
    for sample in tqdm.tqdm(samples_to_mod_df["file_name"].items()):
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
        (
            img_path,
            labels_path,
            dst_folder,
            mod_sample_name,
            mod_labels_name,
            bboxes_sample_name,
            segments_sample_name,
        ) = get_sample_paths_and_names(sample[1])

        mods = get_modifications_dict(blueprint_df=blueprint_df, sample=sample)

        (
            grid_vertices,
            delaunay_mesh,
            mesh_sampling,
            grid_marks,
            document_points,
        ) = compute_mesh(sample_path=labels_path, properties=properties)

        mesh_name = "".join([sample[1], "_", datetime.now().isoformat()])
        create_mesh_object(
            vertices=grid_vertices,
            mesh=delaunay_mesh,
            properties=properties,
            mesh_name=mesh_name,
        )

        bboxes_mapped_ids = approx_bboxes_points_to_grid(
            mesh_sampling, grid_marks, document_points
        )

        # Textures
        apply_document_texture(document=img_path, mods_dict=mods)

        # Set background
        background_data = properties["blender"]["common"]["background"]
        create_background(back_data=background_data, mods_dict=mods)
        # Import Background Object
        import_background_object(mods_dict=mods)

        # Set light
        lights_data = properties["blender"][requirements["styles"][0]]["lights"]
        config_lights(lights_data=lights_data)
        set_lighting_syle(mods_dict=mods)

        # Set camera
        camera_data = properties["blender"][requirements["styles"][0]]["camera"]
        config_camera(camera_data=camera_data)

        # Modify Document Mesh
        modify_document_mesh(mods_dict=mods)

        # Cast shadow
        cast_shadow(mods_dict=mods)

        # Render scene
        rendered_img = render_scene(
            dst_folder=os.path.join(dst_folder, "images"),
            name=mod_sample_name,
            img_dims=requirements["img_output"],
        )

        # Obtain new bbox pixel coordinates and write the output
        bboxes_px_points, img = retrieve_bboxes_pixel_points(
            img=copy.deepcopy(rendered_img),
            mesh_name=mesh_name,
            mapped_ids=bboxes_mapped_ids,
        )
        bboxes_sample_path = os.path.join(
            dst_folder, "debug_bboxes", bboxes_sample_name
        )
        os.makedirs(os.path.dirname(bboxes_sample_path), exist_ok=True)
        cv2.imwrite(bboxes_sample_path, img)

        # Modify original labels with the new bboxes layout
        modified_labels, segment_rectangles = ghelp.edit_json_labels(
            json_path=labels_path, points=bboxes_px_points
        )

        # Draw the segments bboxes and write the output
        segments_img = ghelp.draw_bboxes(
            img=copy.deepcopy(rendered_img),
            rectangles=segment_rectangles,
            color=(0, 0, 255),
        )
        segments_sample_path = os.path.join(
            dst_folder, "debug_bboxes", segments_sample_name
        )
        os.makedirs(os.path.dirname(segments_sample_path), exist_ok=True)
        cv2.imwrite(segments_sample_path, segments_img)

        # Write a JSON with the bboxes modified according to the Blender scene modifications
        ghelp.write_json(
            data=modified_labels,
            file_path=os.path.join(dst_folder, "annotations", mod_labels_name),
        )

        # Delete every object in the scene
        print(a)
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

        # Update blueprint
        blueprint_df.loc[
            blueprint_df["file_name"] == sample[1], "modification_done"
        ] = True
        blueprint_df.to_csv(blueprint_path, index=False)


if __name__ == "__main__":
    # Config
    define_units()

    # Load requirements
    requirements = ghelp.load_requirements(root=bpy.path.abspath("//"))

    # Load properties
    properties = dhelp.load_properties(root=bpy.path.abspath("//"))

    # Load blueprint
    blueprint_df, blueprint_path = get_blueprint()

    # Select only those samples still to be generated
    mask = blueprint_df["modification_done"] == False
    filtered_df = blueprint_df[mask]

    modify_samples(filtered_df, blueprint_df)
