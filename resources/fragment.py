from abc import ABC as AbstractClass
from xml.etree import ElementTree as ET
from .codewalker_xml import *
from .drawable import Drawable, LightsProperty
from .bound import BoundsComposite


class YFT:

    file_extension = ".yft.xml"

    @staticmethod
    def from_xml_file(filepath):
        return Fragment.from_xml_file(filepath)

    @staticmethod
    def write_xml(fragment, filepath):
        return fragment.write_xml(filepath)


class BoneTransformItem(MatrixProperty):
    tag_name = "Item"

    def __init__(self, tag_name: str, value=None):
        super().__init__(tag_name, value or Matrix())


class BoneTransformsListProperty(ListProperty):
    list_type = BoneTransformItem
    tag_name = "BoneTransforms"

    def __init__(self, tag_name=None):
        super().__init__(tag_name or BoneTransformsListProperty.tag_name)
        self.unk = AttributeProperty("unk", 1)


class ArchetypeProperty(ElementTree):
    tag_name = "Archetype"

    def __init__(self):
        super().__init__()
        self.name = TextProperty("Name")
        self.mass = ValueProperty("Mass")
        self.mass_inv = ValueProperty("MassInv")
        self.unknown_48 = ValueProperty("Unknown48")
        self.unknown_4c = ValueProperty("Unknown4C")
        self.unknown_50 = ValueProperty("Unknown50")
        self.unknown_54 = ValueProperty("Unknown54")
        self.inertia_tensor = VectorProperty("InertiaTensor")
        self.inertia_tensor_inv = VectorProperty("InertiaTensorInv")
        self.bounds = BoundsComposite()


class TransformItem(MatrixProperty):
    tag_name = "Item"

    def __init__(self, tag_name: str, value=None):
        super().__init__(tag_name, value or Matrix())


class TransformsListProperty(ListProperty):
    list_type = TransformItem
    tag_name = "Transforms"

    def __init__(self, tag_name=None):
        super().__init__(tag_name or TransformsListProperty.tag_name)


class ChildrenItem(ElementTree):
    tag_name = "Item"

    def __init__(self):
        super().__init__()
        self.group_index = ValueProperty("GroupIndex")
        self.bone_tag = ValueProperty("BoneTag")
        self.pristine_mass = ValueProperty("PristineMass")
        self.damaged_mass = ValueProperty("DamagedMass")
        self.unk_float = ValueProperty("UnkFloat")
        self.unk_vec = VectorProperty("UnkVec")
        self.inertia_tensor = QuaternionProperty("InertiaTensor")
        # self.event_set = None # ?????????? FIND
        self.drawable = FragmentDrawable()


class ChildrenListProperty(ListProperty):
    list_type = ChildrenItem
    tag_name = "Children"


class GroupItem(ElementTree):
    tag_name = "Item"

    def __init__(self):
        super().__init__()
        self.name = TextProperty("Name")
        self.parent_index = ValueProperty("ParentIndex")
        self.glass_window_index = ValueProperty("GlassWindowIndex")
        self.glass_flags = ValueProperty("GlassFlags")
        self.strength = ValueProperty("Strength")
        self.force_transmission_scale_up = ValueProperty(
            "ForceTransmissionScaleUp")
        self.force_transmission_scale_down = ValueProperty(
            "ForceTransmissionScaleDown")
        self.joint_stiffness = ValueProperty("JointStiffness")
        self.min_soft_angle_1 = ValueProperty("MinSoftAngle1")
        self.max_soft_angle_1 = ValueProperty("MaxSoftAngle1")
        self.max_soft_angle_2 = ValueProperty("MaxSoftAngle2")
        self.max_soft_angle_3 = ValueProperty("MaxSoftAngle3")
        self.rotation_speed = ValueProperty("RotationSpeed")
        self.rotation_strength = ValueProperty("RotationStrength")
        self.restoring_strength = ValueProperty("RestoringStrength")
        self.restoring_max_torque = ValueProperty("RestoringMaxTorque")
        self.latch_strength = ValueProperty("LatchStrength")
        self.mass = ValueProperty("Mass")
        self.min_damage_force = ValueProperty("MinDamageForce")
        self.damage_health = ValueProperty("DamageHealth")
        self.unk_float_5c = ValueProperty("UnkFloat5C")
        self.unk_float_60 = ValueProperty("UnkFloat60")
        self.unk_float_64 = ValueProperty("UnkFloat64")
        self.unk_float_68 = ValueProperty("UnkFloat68")
        self.unk_float_6c = ValueProperty("UnkFloat6C")
        self.unk_float_70 = ValueProperty("UnkFloat70")
        self.unk_float_74 = ValueProperty("UnkFloat74")
        self.unk_float_78 = ValueProperty("UnkFloat78")
        self.unk_float_a8 = ValueProperty("UnkFloatA8")


class GroupsListProperty(ListProperty):
    list_type = GroupItem
    tag_name = "Groups"


class LODProperty(ElementTree):
    tag_name = "LOD"

    def __init__(self, tag_name="LOD"):
        super().__init__()
        self.tag_name = tag_name
        self.unknown_14 = ValueProperty("Unknown14")
        self.unknown_18 = ValueProperty("Unknown18")
        self.unknown_1c = ValueProperty("Unknown1C")
        self.position_offset = VectorProperty("PositionOffset")
        self.unknown_40 = VectorProperty("Unknown40")
        self.unknown_50 = VectorProperty("Unknown50")
        self.damping_linear_c = VectorProperty("DampingLinearC")
        self.damping_linear_v = VectorProperty("DampingLinearV")
        self.damping_linear_v2 = VectorProperty("DampingLinearV2")
        self.damping_angular_c = VectorProperty("DampingAngularC")
        self.damping_angular_v = VectorProperty("DampingAngularV")
        self.damping_angular_v2 = VectorProperty("DampingAngularV2")
        self.archetype = ArchetypeProperty()
        self.transforms = TransformsListProperty()
        self.groups = GroupsListProperty()
        self.children = ChildrenListProperty()


class PhysicsProperty(ElementTree):
    tag_name = "Physics"

    def __init__(self):
        super().__init__()
        self.lod1 = LODProperty("LOD1")
        self.lod2 = LODProperty("LOD2")
        self.lod3 = LODProperty("LOD3")


class WindowItem(ElementTree):
    tag_name = "Window"

    def __init__(self):
        super().__init__()
        self.item_id = ValueProperty("ItemID")
        self.unk_ushort_1 = ValueProperty("UnkUshort1")
        self.unk_ushort_4 = ValueProperty("UnkUshort4")
        self.unk_ushort_5 = ValueProperty("UnkUshort5")
        self.unk_float_1 = ValueProperty("UnkFloat1")
        self.unk_float_2 = ValueProperty("UnkFloat2")
        self.unk_float_3 = ValueProperty("UnkFloat3")
        self.unk_float_5 = ValueProperty("UnkFloat5")
        self.unk_float_6 = ValueProperty("UnkFloat6")
        self.unk_float_7 = ValueProperty("UnkFloat7")
        self.unk_float_9 = ValueProperty("UnkFloat9")
        self.unk_float_10 = ValueProperty("UnkFloat10")
        self.unk_float_11 = ValueProperty("UnkFloat11")
        self.unk_float_13 = ValueProperty("UnkFloat13")
        self.unk_float_14 = ValueProperty("UnkFloat14")
        self.unk_float_15 = ValueProperty("UnkFloat15")
        self.unk_float_16 = ValueProperty("UnkFloat16")
        self.unk_float_17 = ValueProperty("UnkFloat17")
        self.unk_float_18 = ValueProperty("UnkFloat18")
        self.cracks_texture_tiling = ValueProperty("CracksTextureTiling")
        self.shattermap = TextProperty("ShatterMap")


class VehicleGlassWindows(ListProperty):
    list_type = WindowItem
    tag_name = "VehicleGlassWindows"


class FragmentDrawable(Drawable):

    def __init__(self):
        super().__init__()
        self.matrix = MatrixProperty("Matrix")


class Fragment(ElementTree, AbstractClass):
    tag_name = "Fragment"

    def __init__(self):
        super().__init__()
        self.name = TextProperty("Name")
        self.bounding_sphere_center = VectorProperty("BoundingSphereCenter")
        self.bounding_sphere_radius = ValueProperty("BoundingSphereRadius")
        self.unknown_b0 = ValueProperty("UnknownB0")
        self.unknown_b8 = ValueProperty("UnknownB8")
        self.unknown_bc = ValueProperty("UnknownBC")
        self.unknown_c0 = ValueProperty("UnknownC0")
        self.unknown_c4 = ValueProperty("UnknownC4")
        self.unknown_cc = ValueProperty("UnknownCC")
        self.gravity_factor = ValueProperty("GravityFactor")
        self.buoyancy_factor = ValueProperty("BuoyancyFactor")
        self.drawable = FragmentDrawable()
        self.bones_transforms = BoneTransformsListProperty()
        self.physics = PhysicsProperty()
        self.lights = LightsProperty()
        self.vehicle_glass_windows = VehicleGlassWindows()
