import bpy
from bpy.types import (
    Context,
    Operator,
)
from bpy.props import (
    FloatProperty
)
from bpy_extras.mesh_utils import edge_loops_from_edges
import numpy as np
from .cable import (
    CableAttr,
    mesh_add_cable_attribute,
    mesh_has_cable_attribute,
    is_cable_mesh_object,
)


class CableEditRestrictedHelper:
    @classmethod
    def poll(cls, context: Context):
        cls.poll_message_set("Must be in Edit mode with a cable drawable model.")
        obj = context.active_object
        return obj is not None and obj.mode == "EDIT" and is_cable_mesh_object(obj)


class CableRestrictedHelper:
    @classmethod
    def poll(cls, context: Context):
        cls.poll_message_set("Must have a cable drawable model selected.")
        objs = context.selected_objects
        return any(is_cable_mesh_object(obj) for obj in objs)


class CableSetAttributeBase(CableEditRestrictedHelper):
    bl_options = {"REGISTER", "UNDO"}

    attribute: CableAttr
    value: float

    def execute(self, context):
        obj = context.active_object

        mode = obj.mode
        # we need to switch from Edit mode to Object mode so the selection gets updated
        bpy.ops.object.mode_set(mode="OBJECT")

        mesh = obj.data
        if not mesh_has_cable_attribute(mesh, self.attribute):
            mesh_add_cable_attribute(mesh, self.attribute)

        attr = mesh.attributes[self.attribute]
        for v in mesh.vertices:
            if not v.select:
                continue

            attr.data[v.index].value = self.value

        bpy.ops.object.mode_set(mode=mode)
        return {"FINISHED"}


class SOLLUMZ_OT_cable_set_radius(Operator, CableSetAttributeBase):
    bl_idname = "sollumz.cable_set_radius"
    bl_label = "Set Cable Radius"
    bl_description = (
        "Sets the radius of the cable at the selected vertices.\n\n"
    ) + f"{CableAttr.RADIUS.label}: {CableAttr.RADIUS.description}"

    attribute = CableAttr.RADIUS
    value: FloatProperty(
        name=CableAttr.RADIUS.label, description=CableAttr.RADIUS.description,
        min=0.0001, default=CableAttr.RADIUS.default_value,
        subtype="DISTANCE"
    )


class SOLLUMZ_OT_cable_set_diffuse_factor(Operator, CableSetAttributeBase):
    bl_idname = "sollumz.cable_set_diffuse_factor"
    bl_label = "Set Cable Diffuse Factor"
    bl_description = (
        "Sets the diffuse factor of the cable at the selected vertices.\n\n"
    ) + f"{CableAttr.DIFFUSE_FACTOR.label}: {CableAttr.DIFFUSE_FACTOR.description}"

    attribute = CableAttr.DIFFUSE_FACTOR
    value: FloatProperty(
        name=CableAttr.DIFFUSE_FACTOR.label, description=CableAttr.DIFFUSE_FACTOR.description,
        min=0.0, max=1.0, default=CableAttr.DIFFUSE_FACTOR.default_value,
        subtype="FACTOR"
    )


class SOLLUMZ_OT_cable_set_um_scale(Operator, CableSetAttributeBase):
    bl_idname = "sollumz.cable_set_um_scale"
    bl_label = "Set Cable Micromovements Scale"
    bl_description = (
        "Sets the micromovements scale of the cable at the selected vertices.\n\n"
    ) + f"{CableAttr.UM_SCALE.label}: {CableAttr.UM_SCALE.description}"

    attribute = CableAttr.UM_SCALE
    value: FloatProperty(
        name=CableAttr.UM_SCALE.label, description=CableAttr.UM_SCALE.description,
        min=0.0, default=CableAttr.UM_SCALE.default_value
    )


class SOLLUMZ_OT_cable_randomize_phase_offset(Operator, CableRestrictedHelper):
    bl_idname = "sollumz.cable_randomize_phase_offset"
    bl_label = "Randomize Cable Phase Offset"
    bl_description = (
        "Sets the phase offset to a different random value for each cable piece (i.e. connected vertices receive the "
        "same random value).\n\n"
    ) + f"{CableAttr.PHASE_OFFSET.label}: {CableAttr.PHASE_OFFSET.description}"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: Context):
        for obj in context.selected_objects:
            if not is_cable_mesh_object(obj):
                continue

            mode = obj.mode
            bpy.ops.object.mode_set(mode="OBJECT")

            mesh = obj.data
            if not mesh_has_cable_attribute(mesh, CableAttr.PHASE_OFFSET):
                mesh_add_cable_attribute(mesh, CableAttr.PHASE_OFFSET)

            attr = mesh.attributes[CableAttr.PHASE_OFFSET]
            pieces = edge_loops_from_edges(mesh)
            phase_offsets = np.random.default_rng().random((len(pieces), 2))
            for i, piece in enumerate(pieces):
                x, y = phase_offsets[i]
                for vi in piece:
                    attr.data[vi].vector = x, y, 0.0

            bpy.ops.object.mode_set(mode=mode)

        return {"FINISHED"}
