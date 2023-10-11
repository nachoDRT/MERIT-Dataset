import bpy
import numpy as np
from scipy.spatial import Delaunay
import sys
import os
from pathlib import Path

file_dir = os.path.join(bpy.path.abspath("//"), "scripts")

if file_dir not in sys.path:
    sys.path.append(file_dir)

import delaunay_helper as dhelp

SHOW_PLOT = True


def define_units(scale_length: float = 1.0, lenght_unit: str = "METERS"):
    """
    Sets the unit settings for the current Blender scene to metric, with specified scale
    and unit of length.

    This function updates the unit settings of the current Blender scene to use the
    metric system, and allows for setting a specific scale length and unit of length.
    The scale length is a multiplier that adjusts the size of the 3D space in Blender,
    and the unit of length specifies the name of the units being used (e.g., meters).

    Parameters:
        scale_length (float): The scale length to set in Blender. Defaults to 1.0.
        length_unit (str): The unit of length to use (e.g., "METERS", "MILLIMETERS").
        Defaults to "METERS".
    """

    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = scale_length
    bpy.context.scene.unit_settings.length_unit = lenght_unit


def compute_mesh(sample_path: str):
    """
    Compute a Delaunay mesh based on the data from a labeled document JSON file.

    Given the path of a sample JSON file, this function reads the file to obtain the
    data, computes a grid, and generates a Delaunay mesh from the vertices obtained from
    both the document points and the grid. If the global variable `SHOW_PLOT` is set to
    True, it also plots the resulting Delaunay mesh.

    Parameters:
        sample_path (str): The name of the sample JSON file containing the data.

    Returns:
        tuple: A tuple containing:
            - vertices (np.array): The array of vertices used to compute the Delaunay
            mesh (it contains the document vertices + the grid vertices).
            - delaunay_mesh (Delaunay): The computed Delaunay mesh.
    """

    json_info = dhelp.read_json(name=sample_path)
    document_points = dhelp.get_bboxes_as_points(form=json_info["form"])
    grid = dhelp.compute_grid()
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


def create_mesh_object(vertices: np.array, mesh: Delaunay):
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
    three_d_vertices = dhelp.pixel_to_m(three_d_vertices)
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

    Parameters:
    document (str): The file path of the document texture image.
    paper (str): The file path of the paper texture image.
    """

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


def create_background(texture: str, normals: str):
    pass


if __name__ == "__main__":
    # Config
    define_units()

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
    background_texture = ""
    background_normal = ""

    # Mesh and object
    vertices, delaunay_mesh = compute_mesh(sample_path=sample_name)
    create_mesh_object(vertices=vertices, mesh=delaunay_mesh)

    # Textures
    apply_texture(document=document_texture, paper=paper_texture)

    # Modify mesh

    # Set background
    # create_background()

    # Set light

    # Set camera

    # Render scene
