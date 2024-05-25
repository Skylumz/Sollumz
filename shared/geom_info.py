import numpy as np
from mathutils import Vector
from typing import NamedTuple


class Centroid(NamedTuple):
    centroid: Vector
    radius_around_centroid: float


class MassProperties(NamedTuple):
    volume: float
    center_of_gravity: Vector
    inertia: Vector


def get_centroid_of_cylinder(radius: float, length: float) -> Centroid:
    half_length = length * 0.5

    # Assume a cylinder placed at origin
    centroid = Vector((0.0, 0.0, 0.0))

    # For cylinders the radius of the sphere that circumscribes the cylinder is equal to
    # half the diagonal of the cylinder
    # diagonal = length of hypotenuse of a right triangle with sides equal to cylinder
    #            diameter and length (or radius and half length)
    radius_around_centroid = np.sqrt(radius * radius + half_length * half_length)

    return Centroid(centroid, radius_around_centroid)


def get_mass_properties_of_cylinder(radius: float, length: float) -> MassProperties:
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

    return MassProperties(volume, cg, inertia)


def get_centroid_of_disc(radius: float) -> Centroid:
    # Assume a disc placed at origin
    centroid = Vector((0.0, 0.0, 0.0))

    # For discs just the radius is used for radius_around_centroid, unlike cylinders
    radius_around_centroid = radius

    return Centroid(centroid, radius_around_centroid)


def get_mass_properties_of_disc(radius: float, length: float) -> MassProperties:
    # Disc mass properties are the same as a cylinder
    return get_mass_properties_of_cylinder(radius, length)


def get_centroid_of_capsule(radius: float, length: float) -> Centroid:
    half_length = length * 0.5

    # Assume a capsule placed at origin
    centroid = Vector((0.0, 0.0, 0.0))
    radius_around_centroid = half_length + radius

    return Centroid(centroid, radius_around_centroid)


def get_mass_properties_of_capsule(radius: float, length: float) -> MassProperties:
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

    return MassProperties(volume, cg, inertia)


def get_centroid_of_sphere(radius: float) -> Centroid:
    # Assume a sphere placed at origin
    centroid = Vector((0.0, 0.0, 0.0))
    radius_around_centroid = radius

    return Centroid(centroid, radius_around_centroid)


def get_mass_properties_of_sphere(radius: float) -> MassProperties:
    radius2 = radius * radius
    radius3 = radius2 * radius

    # Assume a sphere placed at origin
    cg = Vector((0.0, 0.0, 0.0))

    volume = (4 / 3) * np.pi * radius3

    # https://scienceworld.wolfram.com/physics/MomentofInertiaSphere.html
    ixx = iyy = izz = 2 * radius2 / 5
    inertia = Vector((ixx, iyy, izz))

    return MassProperties(volume, cg, inertia)


def get_centroid_of_box(box_min: Vector, box_max: Vector) -> Centroid:
    # Assume a box placed at origin
    centroid = Vector((0.0, 0.0, 0.0))
    radius_around_centroid = 0.5 * (box_max - box_min).length

    return Centroid(centroid, radius_around_centroid)


def get_mass_properties_of_box(box_min: Vector, box_max: Vector) -> MassProperties:
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

    return MassProperties(volume, cg, inertia)


def get_centroid_of_mesh(mesh_vertices) -> Centroid:
    from . import miniball

    while True:
        try:  # ugly, miniball can sometimes fail, so try again...
            C, r2 = miniball.get_bounding_ball(mesh_vertices)
            break
        except np.linalg.LinAlgError:
            continue
    centroid = Vector(C)
    radius_around_centroid = np.sqrt(r2)
    return Centroid(centroid, radius_around_centroid)


def get_mass_properties_of_mesh(mesh_vertices, mesh_faces):
    triangles = mesh_vertices[mesh_faces]

    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]

    tri_tetrahedron_volumes = (v0 * np.cross(v1, v2, axis=1)).sum(axis=1) / 6
    volume = abs(tri_tetrahedron_volumes.sum())

    if is_mesh_solid(mesh_vertices, mesh_faces):
        tri_tetrahedron_cgs = (v0 + v1 + v2) / 4
        tri_tetrahedron_cgs *= tri_tetrahedron_volumes[:, np.newaxis]
        cg = tri_tetrahedron_cgs.sum(axis=0) / volume
    else:
        # Since the mesh is open, we approximate the center of gravity with the average of the triangle
        # centroids scaled by their area.
        tri_areas = np.linalg.norm(np.cross(v0 - v1, v2 - v1, axis=1), axis=1) / 2
        tri_cgs = (v0 + v1 + v2) / 3
        tri_cgs *= tri_areas[:, np.newaxis]
        cg = tri_cgs.sum(axis=0) / tri_areas.sum()

    cg = Vector(cg)

    ixx = 0.0
    iyy = 0.0
    izz = 0.0
    for tri_idx, (v0, v1, v2) in enumerate(triangles):
        # Based on https://github.com/bulletphysics/bullet3/blob/e9c461b0ace140d5c73972760781d94b7b5eee53/src/BulletCollision/CollisionShapes/btConvexTriangleMeshShape.cpp#L236
        # TODO: vectorize with numpy
        a = Vector(v0) - cg
        b = Vector(v1) - cg
        c = Vector(v2) - cg

        i = [0.0, 0.0, 0.0]
        vol_neg = -tri_tetrahedron_volumes[tri_idx]
        for j in range(3):
            i[j] = vol_neg * (
                0.1 * (a[j] * a[j] + b[j] * b[j] + c[j] * c[j]) +
                0.05 * (a[j] * b[j] + a[j] * b[j] + a[j] * c[j] + a[j] * c[j] + b[j] * c[j] + b[j] * c[j])
            )

        i00 = -i[0]
        i11 = -i[1]
        i22 = -i[2]

        ixx += i11 + i22
        iyy += i22 + i00
        izz += i00 + i11

    ixx /= volume
    iyy /= volume
    izz /= volume

    inertia = Vector((ixx, iyy, izz))
    return MassProperties(volume, cg, inertia)


def is_mesh_solid(mesh_vertices, mesh_faces) -> bool:
    """Gets whether the mesh is a closed oriented manifold."""

    # TODO: this can be optimized, we're doing a lot of unnecesary work for easier debugging

    def _get_edge_to_neighbour_faces_map():
        """Returns an array indexed by edge indices, with a list of faces connected to each edge."""
        from collections import defaultdict
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
    return len(boundary_edges) == 0 and len(non_manifold_edges) == 0


def calculate_composite_inertia(
    root_cg: Vector,
    parts_cg: list[Vector],
    parts_mass: list[float],
    parts_inertia: list[Vector]
) -> Vector:
    assert len(parts_cg) == len(parts_mass)
    assert len(parts_cg) == len(parts_inertia)

    total_inertia = Vector((0.0, 0.0, 0.0))
    for cg, mass, inertia in zip(parts_cg, parts_mass, parts_inertia):
        x, y, z = cg - root_cg
        x2, y2, z2 = x * x, y * y, z * z
        inertia.x += mass * (y2 + z2)
        inertia.y += mass * (z2 + x2)
        inertia.z += mass * (x2 + y2)

        total_inertia += inertia

    return total_inertia
