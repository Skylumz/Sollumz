import bpy
from bpy.types import (
    Mesh
)
from mathutils import Vector
from bpy_extras.mesh_utils import edge_loops_from_edges
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, NamedTuple

from ..tools.meshhelper import flip_uvs
from ..cwxml.drawable import VertexBuffer
from ..shared.math import distance_point_to_line
from .cable import CableAttr, is_cable_mesh, mesh_get_cable_attribute_values

# Vertex Layout:
#   Position
#   Normal = tangent
#   Colour0
#    r = phase offset 1
#    g = phase offset 2
#    b = (unused)
#    a = diffuse, lerp factor between shader_cableDiffuse and shader_cableDiffuse2
#   TexCoord0
#    x = radius (signed)
#    y = distance from line * micromovement scale (further from the line and larger scale more affected by wind)
class CableVertexBufferBuilder:
    """Builds Geometry vertex buffers from a cable curve."""

    def __init__(self, mesh: Mesh):
        assert is_cable_mesh(mesh), "Non-cable mesh passed to CableVertexBufferBuilder"
        self.mesh = mesh

    def build(self) -> NDArray:
        return self._build_vertex_buffer()

    def _build_vertex_buffer(self) -> NDArray:
        verts_position = np.empty((len(self.mesh.vertices), 3), dtype=np.float32)
        self.mesh.attributes["position"].data.foreach_get("vector", verts_position.ravel())
        verts_radius = mesh_get_cable_attribute_values(self.mesh, CableAttr.RADIUS)
        verts_diffuse_factor = mesh_get_cable_attribute_values(self.mesh, CableAttr.DIFFUSE_FACTOR)
        verts_um_scale = mesh_get_cable_attribute_values(self.mesh, CableAttr.UM_SCALE)
        verts_phase_offset = mesh_get_cable_attribute_values(self.mesh, CableAttr.PHASE_OFFSET)

        verts_phase_offset.clip(0.0, 1.0, out=verts_phase_offset)
        verts_diffuse_factor.clip(0.0, 1.0, out=verts_diffuse_factor)

        pieces = edge_loops_from_edges(self.mesh)

        # Each segment between two points is composed of 2 triangles (6 vertices)
        num_output_verts = sum((len(vertices) - 1) * 6 for vertices in pieces)

        struct_dtype = [VertexBuffer.VERT_ATTR_DTYPES[attr_name]
                        for attr_name in ("Position", "Normal", "Colour0", "TexCoord0")]
        output_vertex_arr = np.empty(num_output_verts, dtype=struct_dtype)
        v_pos = output_vertex_arr["Position"]
        v_tan = output_vertex_arr["Normal"]
        v_col = output_vertex_arr["Colour0"]
        v_tex = output_vertex_arr["TexCoord0"]

        vert_idx = 0
        for vertices in pieces:
            # TODO: might not be the proper way to calculate tangents
            tangents = [None] * len(vertices)
            distances = [None] * len(vertices)
            start = Vector(verts_position[vertices[0]])
            end = Vector(verts_position[vertices[-1]])
            for i in range(len(vertices)):
                v0 = vertices[i]
                p0 = Vector(verts_position[v0])
                if i == 0:
                    v1 = vertices[i + 1]
                    p1 = Vector(verts_position[v1])
                    tangent = (p1 - p0).normalized()
                elif i == (len(vertices) - 1):
                    vM1 = vertices[i - 1]
                    pM1 = Vector(verts_position[vM1])
                    tangent = (p0 - pM1).normalized()
                else:
                    v1 = vertices[i + 1]
                    vM1 = vertices[i - 1]
                    p1 = Vector(verts_position[v1])
                    pM1 = Vector(verts_position[vM1])
                    tangent = (p1 - pM1).normalized()

                tangents[i] = tangent

                distances[i] = distance_point_to_line(start, end, p0)

            for i0 in range(len(vertices) - 1):
                i1 = i0 + 1
                v0 = vertices[i0]
                v1 = vertices[i1]

                # First triangle (v0 -> v1 -> v0)
                v_pos[vert_idx] = verts_position[v0]
                v_tan[vert_idx] = tangents[i0]
                v_col[vert_idx][0] = int(verts_phase_offset[v0][0] * 255)
                v_col[vert_idx][1] = int(verts_phase_offset[v0][1] * 255)
                v_col[vert_idx][2] = 0 # unused
                v_col[vert_idx][3] = int(verts_diffuse_factor[v0] * 255)
                v_tex[vert_idx][0] = -verts_radius[v0] # negative
                v_tex[vert_idx][1] = distances[i0] * verts_um_scale[v0]

                v_pos[vert_idx + 1] = verts_position[v1]
                v_tan[vert_idx + 1] = tangents[i1]
                v_col[vert_idx + 1][0] = int(verts_phase_offset[v1][0] * 255)
                v_col[vert_idx + 1][1] = int(verts_phase_offset[v1][1] * 255)
                v_col[vert_idx + 1][2] = 0 # unused
                v_col[vert_idx + 1][3] = int(verts_diffuse_factor[v1] * 255)
                v_tex[vert_idx + 1][0] = -verts_radius[v1] # negative
                v_tex[vert_idx + 1][1] = distances[i1] * verts_um_scale[v0]

                #  same as first vertex but with positive radius
                output_vertex_arr[vert_idx + 2] = output_vertex_arr[vert_idx]
                v_tex[vert_idx + 2][0] = verts_radius[v0] # positive

                # Second triangle
                output_vertex_arr[vert_idx + 3] = output_vertex_arr[vert_idx + 2]
                output_vertex_arr[vert_idx + 4] = output_vertex_arr[vert_idx + 1]
                output_vertex_arr[vert_idx + 5] = output_vertex_arr[vert_idx + 1]
                v_tex[vert_idx + 5][0] = verts_radius[v1] # positive

                vert_idx += 6

        return output_vertex_arr
