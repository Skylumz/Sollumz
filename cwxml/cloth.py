
from abc import ABC as AbstractClass, abstractmethod
from collections import defaultdict
from mathutils import Vector
from xml.etree import ElementTree as ET
from .element import (
    AttributeProperty,
    ElementTree,
    ElementProperty,
    FlagsProperty,
    ListProperty,
    MatrixProperty,
    ValueProperty,
    VectorProperty,
    TextProperty,
    TextListProperty,
    InlineValueListProperty,
    Vector4ListProperty,
)
from .drawable import Drawable


# class YLD:
#
#     file_extension = ".yld.xml"
#
#     @staticmethod
#     def from_xml_file(filepath):
#         return ClothDictionaryFile.from_xml_file(filepath)
#
#     @staticmethod
#     def write_xml(cloth_dict_file, filepath):
#         return cloth_dict_file.write_xml(filepath)


# class ClothDictionaryFile(ElementTree):
#     tag_name = "ClothDictionary"
#
#     def __init__(self):
#         super().__init__()

class ClothBridgeSimGfx(ElementTree):
    tag_name = "BridgeSimGfx"

    def __init__(self):
        super().__init__()
        self.vertex_count_high = ValueProperty("VertexCount", 0)
        self.vertex_count_med = ValueProperty("Unknown14", 0)
        self.vertex_count_low = ValueProperty("Unknown18", 0)
        self.vertex_count_vlow = ValueProperty("Unknown1C", 0)  # not imported/exported by CW
        self.pin_radius_high = InlineValueListProperty("Unknown20")
        self.pin_radius_med = InlineValueListProperty("Unknown30")
        self.pin_radius_low = InlineValueListProperty("Unknown40")
        self.pin_radius_vlow = InlineValueListProperty("Unknown50")  # not imported/exported by CW
        self.vertex_weights_high = InlineValueListProperty("Unknown60")
        self.vertex_weights_med = InlineValueListProperty("Unknown70")  # imported/exported as uint32 array by CW
        self.vertex_weights_low = InlineValueListProperty("Unknown80")  # imported/exported as uint32 array by CW
        self.vertex_weights_vlow = InlineValueListProperty("Unknown90")  # not imported/exported by CW
        self.inflation_scale_high = InlineValueListProperty("UnknownA0")
        self.inflation_scale_med = InlineValueListProperty("UnknownB0")  # imported/exported as uint32 array by CW
        self.inflation_scale_low = InlineValueListProperty("UnknownC0")  # imported/exported as uint32 array by CW
        self.inflation_scale_vlow = InlineValueListProperty("UnknownD0")  # not imported/exported by CW
        self.display_map_high = InlineValueListProperty("UnknownE0")  # mesh vertex index -> cloth vertex index map
        self.display_map_med = InlineValueListProperty("UnknownF0")
        self.display_map_low = InlineValueListProperty("Unknown100")
        self.display_map_vlow = InlineValueListProperty("Unknown110")  # not imported/exported by CW
        self.pinnable_list = InlineValueListProperty("Unknown128")  # actually a bitset


class MorphMapData(ElementTree):
    tag_name = None

    def __init__(self, tag_name="MorphMapData"):
        super().__init__()
        self.tag_name = tag_name
        self.poly_count = ValueProperty("Unknown180", 0)
        self.morph_map_high_weights = Vector4ListProperty("Unknown0")  # not imported/exported by CW
        self.morph_map_high_vertex_index = InlineValueListProperty("Unknown10")  # not imported/exported by CW
        self.morph_map_high_index0 = InlineValueListProperty("Unknown20")  # not imported/exported by CW
        self.morph_map_high_index1 = InlineValueListProperty("Unknown30")  # not imported/exported by CW
        self.morph_map_high_index2 = InlineValueListProperty("Unknown40")  # not imported/exported by CW
        self.morph_map_med_weights = Vector4ListProperty("Unknown50")
        self.morph_map_med_vertex_index = InlineValueListProperty("Unknown60")
        self.morph_map_med_index0 = InlineValueListProperty("Unknown70")
        self.morph_map_med_index1 = InlineValueListProperty("Unknown80")
        self.morph_map_med_index2 = InlineValueListProperty("Unknown90")
        self.morph_map_low_weights = Vector4ListProperty("UnknownA0")
        self.morph_map_low_vertex_index = InlineValueListProperty("UnknownB0")
        self.morph_map_low_index0 = InlineValueListProperty("UnknownC0")
        self.morph_map_low_index1 = InlineValueListProperty("UnknownD0")
        self.morph_map_low_index2 = InlineValueListProperty("UnknownE0")
        self.morph_map_vlow_weights = Vector4ListProperty("UnknownF0")  # not imported/exported by CW
        self.morph_map_vlow_vertex_index = InlineValueListProperty("Unknown100")  # not imported/exported by CW
        self.morph_map_vlow_index0 = InlineValueListProperty("Unknown110")  # not imported/exported by CW
        self.morph_map_vlow_index1 = InlineValueListProperty("Unknown120")  # not imported/exported by CW
        self.morph_map_vlow_index2 = InlineValueListProperty("Unknown130")  # not imported/exported by CW
        self.index_map_high = InlineValueListProperty("Unknown140")  # not imported/exported by CW
        self.index_map_med = InlineValueListProperty("Unknown150")
        self.index_map_low = InlineValueListProperty("Unknown160")
        self.index_map_vlow = InlineValueListProperty("Unknown170")  # not imported/exported by CW


class MorphController(ElementTree):
    tag_name = "MorphController"

    def __init__(self):
        super().__init__()
        self.map_data_high = MorphMapData("Unknown18")
        self.map_data_med = MorphMapData("Unknown20")
        self.map_data_low = MorphMapData("Unknown28")
        self.map_data_vlow = MorphMapData("Unknown30")  # not imported/exported by CW


class VerletClothVerticesProperty(ElementProperty):
    """List of Vector3s including a padding NaN component"""
    value_types = (list)

    def __init__(self, tag_name: str = "Vertices", value=None):
        super().__init__(tag_name, value or [])

    @staticmethod
    def from_xml(element: ET.Element):
        new = VerletClothVerticesProperty(element.tag, [])
        text = element.text.strip().split("\n")
        if len(text) > 0:
            for line in text:
                coords = line.strip().split(",")
                if not len(coords) == 4:
                    return VerletClothVerticesProperty.read_value_error(element)

                new.value.append(Vector((float(coords[0]), float(coords[1]), float(coords[2]))))  # coords[3] is padding

        return new

    def to_xml(self):
        element = ET.Element(self.tag_name)
        text = ["\n"]

        if not self.value:
            return

        for vertex in self.value:
            if not isinstance(vertex, Vector):
                raise TypeError(f"VerletClothVerticesProperty can only contain Vector objects, not '{type(self.value)}'!")
            text.append(f"{vertex.x}, {vertex.y}, {vertex.z}, NaN\n")  # padding component exported by CW

        element.text = "".join(text)

        return element


class VerletClothEdge(ElementTree):
    tag_name = "Item"

    def __init__(self):
        super().__init__()
        self.vertex0 = ValueProperty("Unknown0", 0)
        self.vertex1 = ValueProperty("Unknown2", 0)
        self.length_sqr = ValueProperty("Unknown4", 0.0)
        self.weight0 = ValueProperty("Unknown8", 0.5)
        self.compression_weight = ValueProperty("UnknownC", 0.25)


class VerletClothEdgeList(ListProperty):
    list_type = VerletClothEdge
    tag_name = "Constraints"

    def __init__(self, tag_name="Constraints"):
        super().__init__()
        self.tag_name = tag_name


class VerletCloth(ElementTree):
    tag_name = None

    def __init__(self, tag_name="VerletCloth"):
        super().__init__()
        self.tag_name = tag_name
        self.bb_min = VectorProperty("BBMin")
        self.bb_max = VectorProperty("BBMax")
        # self.unknown_3C = ValueProperty("Unknown3C")  # BBMin padding
        # self.unknown_4C = ValueProperty("Unknown4C")  # BBMax padding
        # self.unknown_50 = ValueProperty("Unknown50")  # rage::phClothData vftable pointer 4 lower bytes, not a float
        self.switch_distance_up = ValueProperty("UnknownA8", 9999.0)
        self.switch_distance_down = ValueProperty("UnknownAC", 9999.0)
        self.pinned_vertices_count = ValueProperty("UnknownE8")  # padding
        self.flags = ValueProperty("UnknownFA", 0)
        self.dynamic_pin_list_size = ValueProperty("Unknown148", 1)  # min: 1 - max: 31
        self.cloth_weight = ValueProperty("Unknown158", 1.0)
        self.vertex_positions = VerletClothVerticesProperty("Vertices")
        self.vertex_normals = VerletClothVerticesProperty("Vertices2")
        self.edges = VerletClothEdgeList("Constraints")
        self.custom_edges = VerletClothEdgeList("Constraints2")
        # self.bounds = ...("Bounds")

    def to_xml(self):
        element = super().to_xml()

        # The pointer that CW calls Behaviour is actually the elements pointer of the uint32 array CollisionInst. If the
        # tag is present, CW allocates a struct of 0x40 bytes for this pointer, which is equivalent to an array with
        # capacity 16.
        # Since CW hardcodes this array capacity to 16 (`ulong Unknown_138h = 0x100000;`), this tag should always be
        # present.
        element.append(ET.Element("Behaviour"))

        # Unknown140 is a similar case but with the DynamicPinList bits pointer. When present CW allocates a struct of
        # 16 bytes. More than enough to always fit dynamic_pin_list_size number of bits if the range from 1 to 31
        # documented in CW source is accurate.
        element.append(ET.Element("Unknown140"))

        return element


class ClothController(ElementTree):
    tag_name = "Controller"

    def __init__(self):
        super().__init__()
        self.name = TextProperty("Name")
        self.flags = ValueProperty("Type")
        self.unknown_78 = ValueProperty("Unknown78", 0.0)  # PinRadiusScale of characterClothController
        self.bridge = ClothBridgeSimGfx()
        self.morph_controller = MorphController()
        self.cloth_high = VerletCloth("VerletCloth1")
        self.cloth_med = VerletCloth("VerletCloth2")
        self.cloth_low = VerletCloth("VerletCloth3")
        self.cloth_vlow = VerletCloth("VerletCloth4")  # not currently exported/imported by CW :(


class ClothInstanceTuning(ElementTree):
    tag_name = "InstanceTuning"

    def __init__(self):
        super().__init__()


class EnvironmentCloth(ElementTree):
    tag_name = "Item"

    def __init__(self):
        super().__init__()
        self.flags = ValueProperty("Unknown78", 0)
        self.user_data = TextListProperty("UnknownData")
        self.controller = ClothController()
        self.tuning = ClothInstanceTuning()
        self.drawable = Drawable()


class EnvironmentClothList(ListProperty):
    list_type = EnvironmentCloth
    tag_name = "Cloths"
