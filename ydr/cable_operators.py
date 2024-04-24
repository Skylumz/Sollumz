import bpy
from bpy.types import (
    Context,
    Operator,
)
from bpy.props import (
    FloatProperty
)
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


class SOLLUMZ_OT_cable_test(Operator, CableEditRestrictedHelper):
    bl_idname = "sollumz.cable_test"
    bl_label = "Cable Test"
    bl_description = "Cable Test"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        print(f"{obj=}")
        return {"FINISHED"}

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
    bl_description = "Sets the radius of the cable at the selected vertices"

    attribute = CableAttr.RADIUS
    value: FloatProperty(
        name=CableAttr.RADIUS.label, description=CableAttr.RADIUS.description,
        min=0.0001, default=CableAttr.RADIUS.default_value,
        subtype="DISTANCE"
    )


class SOLLUMZ_OT_cable_set_diffuse_factor(Operator, CableSetAttributeBase):
    bl_idname = "sollumz.cable_set_diffuse_factor"
    bl_label = "Set Cable Diffuse Factor"
    bl_description = "Sets the diffuse factor of the cable at the selected vertices"

    attribute = CableAttr.DIFFUSE_FACTOR
    value: FloatProperty(
        name=CableAttr.DIFFUSE_FACTOR.label, description=CableAttr.DIFFUSE_FACTOR.description,
        min=0.0, max=1.0, default=CableAttr.DIFFUSE_FACTOR.default_value,
        subtype="FACTOR"
    )

class SOLLUMZ_OT_cable_set_um_scale(Operator, CableSetAttributeBase):
    bl_idname = "sollumz.cable_set_um_scale"
    bl_label = "Set Cable Micromovements Scale"
    bl_description = "Sets the micromovements scale of the cable at the selected vertices"

    attribute = CableAttr.UM_SCALE
    value: FloatProperty(
        name=CableAttr.UM_SCALE.label, description=CableAttr.UM_SCALE.description,
        min=0.0, default=CableAttr.UM_SCALE.default_value
    )

# class SOLLUMZ_OT_set_bounds_from_selection(SOLLUMZ_OT_base, bpy.types.Operator):
#     """Set room bounds from selection (must be in edit mode)"""
#     bl_idname = "sollumz.setroomboundsfromselection"
#     bl_label = "Set Bounds From Selection"
#
#     @classmethod
#     def poll(cls, context):
#         return get_selected_room(context) is not None and (context.active_object and context.active_object.mode == "EDIT")
#
#     def run(self, context):
#         selected_archetype = get_selected_archetype(context)
#         selected_room = get_selected_room(context)
#         selected_verts = []
#         for obj in context.objects_in_mode:
#             selected_verts.extend(get_selected_vertices(obj))
#         if not len(selected_verts) > 1:
#             self.message("You must select at least 2 vertices!")
#             return False
#         if not selected_archetype.asset:
#             self.message("You must set an asset for the archetype.")
#             return False
#
#         pos = selected_archetype.asset.location
#
#         return True
