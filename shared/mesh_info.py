import numpy as np
from mathutils import Vector
from collections import defaultdict


def get_centroid_of_cylinder(radius: float, length: float):
    half_length = length * 0.5

    # Assume a cylinder placed at origin
    centroid = Vector((0.0, 0.0, 0.0))

    # For cylinders the radius of the sphere that circumscribes the cylinder is equal to
    # half the diagonal of the cylinder
    # diagonal = length of hypotenuse of a right triangle with sides equal to cylinder
    #            diameter and length (or radius and half length)
    radius_around_centroid = np.sqrt(radius * radius + half_length * half_length)

    return centroid, radius_around_centroid


def get_mass_properties_of_cylinder(radius: float, length: float):
    radius2 = radius * radius
    length2 = length * length

    # Assume a cylinder placed at origin
    cg = Vector((0.0, 0.0, 0.0))

    volume = np.pi * radius2 * length

    # https://scienceworld.wolfram.com/physics/MomentofInertiaCylinder.html
    ixx = length2 / 12 + radius2 / 4
    iyy = radius2 / 2
    izz = ixx
    inertia = Vector((ixx, iyy, izz))

    return volume, cg, inertia


def get_centroid_of_capsule(radius: float, length: float):
    half_length = length * 0.5

    # Assume a capsule placed at origin
    centroid = Vector((0.0, 0.0, 0.0))
    radius_around_centroid = half_length + radius

    return centroid, radius_around_centroid


def get_mass_properties_of_capsule(radius: float, length: float):
    radius2 = radius * radius
    radius3 = radius2 * radius
    length2 = length * length
    length3 = length2 * length

    # Assume a capsule placed at origin
    cg = Vector((0.0, 0.0, 0.0))

    # volume capsule = volume cylinder + sphere
    volume_sphere = (4 / 3) * np.pi * radius3
    volume_cylinder = np.pi * radius2 * length
    volume = volume_cylinder + volume_sphere

    # https://www.wolframalpha.com/input?i=moment+of+inertia+of+a+capsule
    ixx = (5 * length3 + 20 * length2 * radius + 45 * length * radius2 + 32 * radius3) / (60 * length + 80 * radius)
    iyy = (radius2 * (15 * length + 16 * radius)) / (30 * length + 40 * radius)
    izz = ixx
    inertia = Vector((ixx, iyy, izz))

    return volume, cg, inertia


def get_centroid_of_sphere(radius: float):
    # Assume a sphere placed at origin
    centroid = Vector((0.0, 0.0, 0.0))
    radius_around_centroid = radius

    return centroid, radius_around_centroid


def get_mass_properties_of_sphere(radius: float):
    radius2 = radius * radius
    radius3 = radius2 * radius

    # Assume a sphere placed at origin
    cg = Vector((0.0, 0.0, 0.0))

    volume = (4 / 3) * np.pi * radius3

    # https://scienceworld.wolfram.com/physics/MomentofInertiaSphere.html
    ixx = iyy = izz = 2 * radius2 / 5
    inertia = Vector((ixx, iyy, izz))

    return volume, cg, inertia


def get_centroid_of_box(box_min: Vector, box_max: Vector):
    # Assume a box placed at origin
    centroid = Vector((0.0, 0.0, 0.0))
    radius_around_centroid = 0.5 * (box_max - box_min).length

    return centroid, radius_around_centroid


def get_mass_properties_of_box(box_min: Vector, box_max: Vector):
    x, y, z = box_max - box_min
    x2, y2, z2 = x * x, y * y, z * z

    # Assume a box placed at origin
    cg = Vector((0.0, 0.0, 0.0))

    volume = x * y * z

    # moment of inertia of a solid rectangular cuboid
    # https://en.wikipedia.org/wiki/List_of_moments_of_inertia#List_of_3D_inertia_tensors
    ixx = (y2 + z2) / 12
    iyy = (z2 + x2) / 12
    izz = (x2 + y2) / 12
    inertia = Vector((ixx, iyy, izz))

    return volume, cg, inertia


def get_centroid_of_mesh(mesh_vertices):
    from .miniball import get_bounding_ball
    C, r2 = get_bounding_ball(mesh_vertices)
    centroid = Vector(C)
    radius_around_centroid = np.sqrt(r2)
    return centroid, radius_around_centroid


def calc_face_projected_areas(v0, v1, v2):
    sx = 0.0
    sy = 0.0
    sz = 0.0
    for v_start, v_end in ((v0, v1), (v1, v2), (v2, v0)):
        vdiff = v_start - v_end
        vsum = v_start + v_end
        sx += vdiff[1] * vsum[2]
        sy += vdiff[2] * vsum[0]
        sz += vdiff[0] * vsum[1]

    sx /= 2.0
    sy /= 2.0
    sz /= 2.0

    return sx, sy, sz


def get_mass_properties_of_mesh_shell(mesh_vertices, mesh_faces):
    """
    Alternative for meshes with holes.
    """
    triangles = mesh_vertices[mesh_faces]

    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    v0x = triangles[:, 0, 0]
    v0y = triangles[:, 0, 1]
    v0z = triangles[:, 0, 2]
    v1x = triangles[:, 1, 0]
    v1y = triangles[:, 1, 1]
    v1z = triangles[:, 1, 2]
    v2x = triangles[:, 2, 0]
    v2y = triangles[:, 2, 1]
    v2z = triangles[:, 2, 2]

    # Since the mesh is open, we approximate the center of gravity with the average of the triangle
    # centroids scaled by their area.
    tri_areas = np.linalg.norm(np.cross(v0 - v1, v2 - v1, axis=1), axis=1) / 2
    tri_cgs = (v0 + v1 + v2) / 3
    tri_cgs *= tri_areas[:, np.newaxis]
    cg = tri_cgs.sum(axis=0) / tri_areas.sum()
    cg = Vector(cg)

    # The mesh is open so it doesn't really have a volume, but the following formula matches the values
    # in the original assets .
    #
    # The idea is we create a tetrahedron for each triangle in the mesh using the vertices of the
    # triangle and a vertex at the origin (0, 0, 0). Then, we calculate the signed volume of each
    # tetrahedron and sum them up to get the total volume of our mesh. Depending on the orientation
    # of the triangles (facing towards the origin or away), each tetrahedron volume may end up positive
    # or negative so summing them up removes the volume part outside the mesh and we are left with the
    # volume only inside our mesh.
    # And we do similarly with the moment of inertia.
    #
    # http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf

    # formula 6
    tri_tetrahedron_volumes = (v0 * np.cross(v1, v2, axis=1)).sum(axis=1) / 6
    volume = abs(tri_tetrahedron_volumes.sum())

    # formulas 9a, 9b and 9c: https://thescipub.com/pdf/jmssp.2005.8.11.pdf
    a = (v0y * v0y + v0y * v1y + v1y * v1y + v0y * v2y + v1y * v2y + v2y * v2y +
         v0z * v0z + v0z * v1z + v1z * v1z * v0z * v2z + v1z * v2z + v2z * v2z)
    b = (v0z * v0z + v0z * v1z + v1z * v1z + v0z * v2z + v1z * v2z + v2z * v2z +
         v0x * v0x + v0x * v1x + v1x * v1x * v0x * v2x + v1x * v2x + v2x * v2x)
    c = (v0x * v0x + v0x * v1x + v1x * v1x + v0x * v2x + v1x * v2x + v2x * v2x +
         v0y * v0y + v0y * v1y + v1y * v1y * v0y * v2y + v1y * v2y + v2y * v2y)
    a = (6 * np.abs(tri_tetrahedron_volumes) * a).sum() / 60
    b = (6 * np.abs(tri_tetrahedron_volumes) * b).sum() / 60
    c = (6 * np.abs(tri_tetrahedron_volumes) * c).sum() / 60
    ixx = a / volume
    iyy = b / volume
    izz = c / volume
    inertia = Vector((ixx, iyy, izz))
    # print("> get_mass_properties_of_mesh_shell")
    # print(f"{volume=:.5f}")
    # print(f"cg={cg.x:.5f}, {cg.y:.5f}, {cg.z:.5f}")
    # print(f"inertia={inertia.x:.5f}, {inertia.y:.5f}, {inertia.z:.5f}")
    return volume, cg, inertia


def get_mass_properties_of_mesh_solid(mesh_vertices, mesh_faces):
    """
    Expects a closed-solid mesh. If there are holes in the mesh, the results won't make sense and might throw
    division-by-zero errors.

    Explanation: https://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf
    """
    triangles = mesh_vertices[mesh_faces]

    v0x = triangles[:, 0, 0]
    v0y = triangles[:, 0, 1]
    v0z = triangles[:, 0, 2]
    v1x = triangles[:, 1, 0]
    v1y = triangles[:, 1, 1]
    v1z = triangles[:, 1, 2]
    v2x = triangles[:, 2, 0]
    v2y = triangles[:, 2, 1]
    v2z = triangles[:, 2, 2]

    def subexpr(w0, w1, w2):
        temp0 = w0 + w1
        f1 = temp0 + w2
        temp1 = w0 * w0
        temp2 = temp1 + w1 * temp0
        f2 = temp2 + w2 * f1
        f3 = w0 * temp1 + w1 * temp2 + w2 * f2

        return f1, f2, f3

    f1x, f2x, f3x = subexpr(v0x, v1x, v2x)
    f1y, f2y, f3y = subexpr(v0y, v1y, v2y)
    f1z, f2z, f3z = subexpr(v0z, v1z, v2z)

    a1, b1, c1 = v1x - v0x, v1y - v0y, v1z - v0z
    a2, b2, c2 = v2x - v0x, v2y - v0y, v2z - v0z
    d0 = b1 * c2 - b2 * c1
    d1 = a2 * c1 - a1 * c2
    d2 = a1 * b2 - a2 * b1

    intg = np.zeros(7)
    intg[0] = np.sum(d0 * f1x)
    intg[1] = np.sum(d0 * f2x)
    intg[2] = np.sum(d1 * f2y)
    intg[3] = np.sum(d2 * f2z)
    intg[4] = np.sum(d0 * f3x)
    intg[5] = np.sum(d1 * f3y)
    intg[6] = np.sum(d2 * f3z)

    intg /= np.array([6, 24, 24, 24, 60, 60, 60])

    volume = intg[0]
    if volume == 0.0:
        nan = float("nan")
        cg = Vector((nan, nan, nan))
        inertia = Vector((nan, nan, nan))
    else:
        cg = Vector((intg[1], intg[2], intg[3])) / volume
        ixx = intg[5] + intg[6] - volume * (cg.y * cg.y + cg.z * cg.z)
        iyy = intg[4] + intg[6] - volume * (cg.z * cg.z + cg.x * cg.x)
        izz = intg[4] + intg[5] - volume * (cg.x * cg.x + cg.y * cg.y)
        inertia = Vector((ixx, iyy, izz)) / volume
    # print("> get_mass_properties_of_mesh_solid")
    # print(f"{volume=:.5f}")
    # print(f"cg={cg.x:.5f}, {cg.y:.5f}, {cg.z:.5f}")
    # print(f"inertia={inertia.x:.5f}, {inertia.y:.5f}, {inertia.z:.5f}")

    return volume, cg, inertia


def is_mesh_solid(mesh_vertices, mesh_faces) -> bool:
    """Gets whether the mesh is a closed oriented manifold."""

    # TODO: this can be optimized, we're doing a lot of unnecesary work for easier debugging

    def _get_edge_to_neighbour_faces_map():
        """Returns an array indexed by edge indices, with a list of faces connected to each edge."""
        edge_to_neighbour_faces = defaultdict(list)
        for face_index, (v0, v1, v2) in enumerate(mesh_faces):
            e0 = (v0, v1)
            e1 = (v1, v2)
            e2 = (v2, v0)
            for edge in (e0, e1, e2):
                edge_reversed = (edge[1], edge[0])
                if edge_reversed in edge_to_neighbour_faces:
                    edge_to_neighbour_faces[edge_reversed].append(face_index)
                else:
                    edge_to_neighbour_faces[edge].append(face_index)

        return edge_to_neighbour_faces

    def _classify_edges_by_manifold():
        edge_to_neighbour_faces = _get_edge_to_neighbour_faces_map()

        # Boundary edges: Edges that are connected to only one face.
        # Manifold edges: Edges that are connected to exactly two faces.
        # Non-manifold edges: Edges that are connected to more than two faces, or no faces at all.
        boundary_edges = []
        manifold_edges = []
        non_manifold_edges = []
        for edge, neighbour_faces in edge_to_neighbour_faces.items():
            num_faces = len(neighbour_faces)
            if num_faces == 1:
                boundary_edges.append(edge)
            elif num_faces == 2:
                manifold_edges.append(edge)
            else:
                non_manifold_edges.append(edge)

        return boundary_edges, manifold_edges, non_manifold_edges

    boundary_edges, manifold_edges, non_manifold_edges = _classify_edges_by_manifold()
    # print(f"  {boundary_edges=}")
    # print(f"  {manifold_edges=}")
    # print(f"  {non_manifold_edges=}")

    return len(boundary_edges) == 0 and len(non_manifold_edges) == 0
