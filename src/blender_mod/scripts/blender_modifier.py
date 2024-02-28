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
            - n_document_points (int): Number of document vertices (word's bboxes)
                used in the computation.
            - vertices (np.array): The array of vertices used to compute the Delaunay
            mesh (it contains the document vertices + the grid vertices).
            - delaunay_mesh (Delaunay): The computed Delaunay mesh.
    """

    json_info = dhelp.read_json(name=sample_path)
    document_points = dhelp.get_bboxes_as_points(
        form=json_info["form"], properties=properties
    )
    n_document_points = len(document_points)
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

    return n_document_points, vertices, delaunay_mesh


def create_mesh_object(
    n_bboxes_vertices: int, vertices: np.array, mesh: Delaunay, properties: dict
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
    mesh_data = bpy.data.meshes.new(name="Document Mesh")
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

    # Create a group so we can easily track vertices later
    bboxes_vertices_group = mesh_obj.vertex_groups.new(name="bboxes_v_group")
    # We take advantage of the order of indices in "vertices": first those related to
    # word bounding boxes, followed by the rest (grid)
    indices_selection = [index for index in range(n_bboxes_vertices)]
    bboxes_vertices_group.add(indices_selection, 1.0, "ADD")

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
    mapping_node.inputs["Scale"].default_value[0] = 1.39
    mapping_node.inputs["Scale"].default_value[1] = 0.98
    mapping_node.inputs["Rotation"].default_value[2] = math.radians(180)
    coord_node = nodes.new(type="ShaderNodeTexCoord")
    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled_node_printer = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled_node_printer.inputs["Base Color"].default_value = (
        0.092,
        0.092,
        0.092,
        1,
    )
    mix_shader = nodes.new(type="ShaderNodeMixShader")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # Printer stain nodes

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
    bands_mapping_node.inputs["Scale"].default_value[0] = 1.39
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
    mix_shader.location = (750, 300)

    # Load images into the texture node
    doc_image = bpy.data.images.load(document)
    doc_texture_node.image = doc_image

    paper_image = bpy.data.images.load(paper)
    paper_texture_node.image = paper_image

    mix_rgb_node.blend_type = "MULTIPLY"

    # Connect the nodes: Document texture
    links = mat.node_tree.links
    links.new(coord_node.outputs["UV"], mapping_node.inputs["Vector"])
    links.new(mapping_node.outputs["Vector"], doc_texture_node.inputs["Vector"])
    links.new(doc_texture_node.outputs["Color"], mix_rgb_node.inputs["Color1"])
    links.new(paper_texture_node.outputs["Color"], mix_rgb_node.inputs["Color2"])
    links.new(mix_rgb_node.outputs["Color"], principled_node.inputs["Base Color"])

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
    links.new(principled_node.outputs["BSDF"], mix_shader.inputs[1])
    links.new(principled_node_printer.outputs["BSDF"], mix_shader.inputs[2])

    links.new(mix_shader.outputs["Shader"], output_node.inputs["Surface"])

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
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

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
    Render the current scene in Blender using the EEVEE render engine, save the rendered
    image to the specified destination folder with the given name, and return the rende-
    red image as a NumPy array

    Args:
        dst_folder (str): The directory path where the rendered image will be saved.
        name (str): The name of the rendered image file (including file extension).
    Returns:
        np.array: The rendered image as an cv2 image.
    """

    camera = bpy.data.objects.get("Camera")
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


def retrieve_bboxes_pixel_points(img: np.array):
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

    camera_name = "Camera"
    obj_name = "Document"
    mesh_name = "Document Mesh"

    camera = bpy.data.objects[camera_name]
    doc_object = bpy.data.objects[obj_name]
    world_location = doc_object.location

    bbox_vertices_indices = get_vertices_id_from_group(
        object_name="Document", group_name="bboxes_v_group"
    )

    mesh = bpy.data.meshes[mesh_name]
    bboxes_vertices_px = []

    # Bboxes vertices: transform 3D world coordinates to 2D pixel coordinates
    for index in bbox_vertices_indices:
        world_location = mesh.vertices[index].co

        render_coordinates = bpy_extras.object_utils.world_to_camera_view(
            bpy.context.scene, camera, world_location
        )

        point_x = int(render_coordinates.x * bpy.context.scene.render.resolution_x)
        point_y = int(
            (1 - render_coordinates.y) * bpy.context.scene.render.resolution_y
        )

        point = (point_x, point_y)
        bboxes_vertices_px.append(point)

    # Draw the bboxes over the rendered img for visual check
    rectangles = []

    # One bbox is defined by 4 points
    for i in range(0, len(bboxes_vertices_px), 4):
        rect_points = np.array(list(bboxes_vertices_px[i : i + 4]), dtype=np.int32)
        rectangles.append(rect_points)

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


def prepare_for_cloth_sim():
    obj = bpy.data.objects["Document"]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)

    # Transform mesh triangles to quads where possible
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.tris_convert_to_quads()
    bpy.ops.object.mode_set(mode="OBJECT")

    # Change its position
    obj.location.z = 0.015


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
    cloth_modifier.settings.quality = 5
    cloth_modifier.collision_settings.collision_quality = 1

    # Activate selfcollisions and set the threshold
    # cloth_modifier.collision_settings.use_self_collision = True
    # cloth_modifier.collision_settings.self_distance_min = 0.001

    # Run the simulation step by step to obtain a proper deformed mesh
    bpy.context.scene.frame_end = 35
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


def import_object(file_path: str, object_name: str):

    directory = os.path.join(file_path, "Object")

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
    x = random.uniform(-0.1, 0.3)
    y = random.uniform(-0.1, 0.4)
    z_angle = random.uniform(0, 360)

    # Change its position and rotation
    obj.location = (x, y, 0)
    obj.rotation_euler = (0, 0, math.radians(z_angle))


def modify_samples(
    samples_to_mod_df: pd.DataFrame,
    blueprint_df: pd.DataFrame,
    paper_texture: str,
    background_texture: str,
    background_normal: str,
):
    for sample in tqdm.tqdm(samples_to_mod_df["file_name"].items()):
        (
            img_path,
            labels_path,
            dst_folder,
            mod_sample_name,
            mod_labels_name,
            bboxes_sample_name,
            segments_sample_name,
        ) = get_sample_paths_and_names(sample[1])

        n_bboxes_vertices, all_vertices, delaunay_mesh = compute_mesh(
            sample_path=labels_path, properties=properties
        )
        create_mesh_object(
            n_bboxes_vertices=n_bboxes_vertices,
            vertices=all_vertices,
            mesh=delaunay_mesh,
            properties=properties,
        )

        # Textures
        apply_texture(document=img_path, paper=paper_texture)

        # Modify mesh
        mesh_data = properties["blender"]["document_mesh_mod"]
        # modify_mesh(mesh_data=mesh_data)

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

        # Import Background Object
        object_name = random.choice(properties["blender"]["background_objects"])
        object_file_path = os.path.join(
            bpy.path.abspath("//"),
            "assets",
            "objects",
            "".join([object_name, ".blend"]),
        )
        import_object(file_path=object_file_path, object_name=object_name)

        prepare_for_cloth_sim()
        run_cloth_sim()
        print(a)

        # Render scene
        rendered_img = render_scene(
            dst_folder=os.path.join(dst_folder, "images"),
            name=mod_sample_name,
            img_dims=requirements["img_output"],
        )

        # Obtain new bbox pixel coordinates and write the output
        bboxes_px_points, img = retrieve_bboxes_pixel_points(
            img=copy.deepcopy(rendered_img)
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

    # Load Assets
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

    modify_samples(
        filtered_df, blueprint_df, paper_texture, background_texture, background_normal
    )
