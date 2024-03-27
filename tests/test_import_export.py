import pytest
import itertools
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from pathlib import Path
from xml.etree import ElementTree as ET
from .test_fixtures import fps_dependent
from .shared import SOLLUMZ_TEST_TMP_DIR, SOLLUMZ_TEST_GAME_ASSETS_DIR, SOLLUMZ_TEST_ASSETS_DIR
from ..ydr.ydrimport import import_ydr
from ..ydr.ydrexport import export_ydr
from ..yft.yftimport import import_yft
from ..yft.yftexport import export_yft
from ..ybn.ybnimport import import_ybn
from ..ybn.ybnexport import export_ybn
from ..ycd.ycdimport import import_ycd
from ..ycd.ycdexport import export_ycd


if SOLLUMZ_TEST_TMP_DIR is not None:
    def asset_path(file_name: str) -> Path:
        path = SOLLUMZ_TEST_ASSETS_DIR.joinpath(file_name)
        assert path.exists()
        return path

    def tmp_path(file_name: str) -> Path:
        path = SOLLUMZ_TEST_TMP_DIR.joinpath(file_name)
        return path

    def glob_assets(ext: str) -> list[tuple[Path, str]]:
        glob_pattern = f"*.{ext}.xml"
        assets = SOLLUMZ_TEST_ASSETS_DIR.rglob(glob_pattern)
        if SOLLUMZ_TEST_GAME_ASSETS_DIR is not None:
            game_assets = SOLLUMZ_TEST_GAME_ASSETS_DIR.rglob(glob_pattern)
            assets = itertools.chain(assets, game_assets)

        return list(map(lambda p: (p, str(p)), assets))

    def elem_to_vec(e: ET.Element):
        return np.array([float(e.get(k)) for k in ("x", "y", "z")])

    def elem_to_float(e: ET.Element):
        return np.array([float(e.get("value"))])

    @pytest.mark.parametrize("ydr_path, ydr_path_str", glob_assets("ydr"))
    def test_import_export_ydr(ydr_path: Path, ydr_path_str: str):
        obj = import_ydr(ydr_path_str)
        assert obj is not None

        out_path = tmp_path(ydr_path.name)
        success = export_ydr(obj, str(out_path))
        assert success
        assert out_path.exists()

    @pytest.mark.parametrize("yft_path, yft_path_str", glob_assets("yft"))
    def test_import_export_yft(yft_path: Path, yft_path_str: str):
        obj = import_yft(yft_path_str)
        assert obj is not None

        out_path = tmp_path(yft_path.name)
        success = export_yft(obj, str(out_path))
        assert success
        assert out_path.exists()

    @pytest.mark.parametrize("ybn_path, ybn_path_str", glob_assets("ybn"))
    def test_import_export_ybn(ybn_path: Path, ybn_path_str: str):
        obj = import_ybn(ybn_path_str)
        assert obj is not None

        out_path = tmp_path(ybn_path.name)
        success = export_ybn(obj, str(out_path))
        assert success
        assert out_path.exists()

    @pytest.mark.parametrize("ycd_path, ycd_path_str", glob_assets("ycd"))
    def test_import_export_ycd(ycd_path: Path, ycd_path_str: str):
        obj = import_ycd(ycd_path_str)
        assert obj is not None

        out_path = tmp_path(ycd_path.name)
        success = export_ycd(obj, str(out_path))
        assert success
        assert out_path.exists()

    # FPS settings equal or greater may output more frames than the original input file had.
    # This is expected because the created action will be longer (more frames) to reach
    # the defined duration at the given FPS. Tests will skip some checks in those cases.
    YCD_MAX_FPS_TO_CHECK_FRAME_COUNTS = 29.97

    def test_import_export_ycd_roundtrip_consistency_num_frames_and_duration(fps_dependent):
        ycd_path = asset_path("roundtrip_anim.ycd.xml")

        def _check_exported_ycd(path: Path):
            tree = ET.ElementTree()
            tree.parse(path)
            root = tree.getroot()

            start_times = [float(e.attrib["value"]) for e in root.findall("./Clips/Item/StartTime")]
            end_times = [float(e.attrib["value"]) for e in root.findall("./Clips/Item/EndTime")]
            rates = [float(e.attrib["value"]) for e in root.findall("./Clips/Item/Rate")]
            frame_counts = [int(e.attrib["value"]) for e in root.findall("./Animations/Item/FrameCount")]
            durations = [float(e.attrib["value"]) for e in root.findall("./Animations/Item/Duration")]

            # values from original roundtrip_anim.ycd.xml
            args = {"rtol": 1e-5, "err_msg": f"Roundtrip output '{path}' does not match original."}
            assert_allclose(start_times, [0.0, 13.33333], **args)
            assert_allclose(end_times, [13.3, 16.63333], **args)
            assert_allclose(rates, [1.0, 1.0], **args)
            if fps_dependent[0] < YCD_MAX_FPS_TO_CHECK_FRAME_COUNTS:
                assert_equal(frame_counts, [501], err_msg=args["err_msg"])
            assert_allclose(durations, [16.66666], **args)

        NUM_ROUNDTRIPS = 5
        curr_input_path = ycd_path
        for roundtrip in range(NUM_ROUNDTRIPS):
            obj = import_ycd(str(curr_input_path))
            assert obj is not None

            out_path = tmp_path(f"roundtrip_anim_{fps_dependent[1]}_{roundtrip}.ycd.xml")
            success = export_ycd(obj, str(out_path))
            assert success
            assert out_path.exists()

            _check_exported_ycd(out_path)

            curr_input_path = out_path

    def test_import_export_ycd_roundtrip_consistency_frame_values(fps_dependent):
        if fps_dependent[0] >= YCD_MAX_FPS_TO_CHECK_FRAME_COUNTS:
            # we check a array with length equal to the frame count, so cannot do this test with high FPS
            return

        ycd_path = asset_path("roundtrip_anim_values.ycd.xml")

        XPATH_QUANTIZE_FLOAT_CHANNEL = "./Animations/Item/Sequences/Item/SequenceData/Item/Channels/Item"

        orig_tree = ET.ElementTree()
        orig_tree.parse(ycd_path)
        orig_root = orig_tree.getroot()
        orig_quantize_float_channel = orig_root.find(XPATH_QUANTIZE_FLOAT_CHANNEL)
        orig_values = np.fromstring(orig_quantize_float_channel.find("Values").text, dtype=np.float32, sep=" ")
        assert len(orig_values) == 30  # quick check to verify we are reading the XML contents correctly
        orig_quantum = float(orig_quantize_float_channel.find("Quantum").attrib["value"])
        orig_offset = float(orig_quantize_float_channel.find("Offset").attrib["value"])

        def _check_exported_ycd(path: Path):
            tree = ET.ElementTree()
            tree.parse(path)
            root = tree.getroot()

            quantize_float_channel = root.find(XPATH_QUANTIZE_FLOAT_CHANNEL)
            values = np.fromstring(quantize_float_channel.find("Values").text, dtype=np.float32, sep=" ")
            quantum = float(quantize_float_channel.find("Quantum").attrib["value"])
            offset = float(quantize_float_channel.find("Offset").attrib["value"])

            args = {"rtol": 1e-5, "err_msg": f"Roundtrip output '{path}' does not match original."}
            assert_allclose(values, orig_values, **args)
            assert_allclose(quantum, orig_quantum, **args)
            assert_allclose(offset, orig_offset, **args)

        NUM_ROUNDTRIPS = 5
        curr_input_path = ycd_path
        for roundtrip in range(NUM_ROUNDTRIPS):
            obj = import_ycd(str(curr_input_path))
            assert obj is not None

            out_path = tmp_path(f"roundtrip_anim_values_{fps_dependent[1]}_{roundtrip}.ycd.xml")
            success = export_ycd(obj, str(out_path))
            assert success
            assert out_path.exists()

            _check_exported_ycd(out_path)

            curr_input_path = out_path

    def test_import_export_ycd_roundtrip_consistency_clip_anim_list(fps_dependent):
        ycd_path = asset_path("roundtrip_anim_clip_anim_list.ycd.xml")

        def _check_exported_ycd(path: Path):
            tree = ET.ElementTree()
            tree.parse(path)
            root = tree.getroot()

            clip_durations = [float(e.attrib["value"]) for e in root.findall("./Clips/Item/Duration")]
            start_times = [float(e.attrib["value"]) for e in root.findall("./Clips/Item/Animations/Item/StartTime")]
            end_times = [float(e.attrib["value"]) for e in root.findall("./Clips/Item/Animations/Item/EndTime")]
            rates = [float(e.attrib["value"]) for e in root.findall("./Clips/Item/Animations/Item/Rate")]
            frame_counts = [int(e.attrib["value"]) for e in root.findall("./Animations/Item/FrameCount")]
            durations = [float(e.attrib["value"]) for e in root.findall("./Animations/Item/Duration")]

            # values from original roundtrip_anim_clip_anim_list.ycd.xml
            args = {"rtol": 1e-5, "err_msg": f"Roundtrip output '{path}' does not match original."}
            assert_allclose(clip_durations, [1.33333], **args)
            assert_allclose(start_times, [9.966667, 7.3], **args)
            assert_allclose(end_times, [11.3, 8.7], **args)
            assert_allclose(rates, [1.0, 1.05], **args)
            if fps_dependent[0] < YCD_MAX_FPS_TO_CHECK_FRAME_COUNTS:
                assert_equal(frame_counts, [709, 551], err_msg=args["err_msg"])
            assert_allclose(durations, [23.6, 22.0], **args)

        NUM_ROUNDTRIPS = 5
        curr_input_path = ycd_path
        for roundtrip in range(NUM_ROUNDTRIPS):
            obj = import_ycd(str(curr_input_path))
            assert obj is not None

            out_path = tmp_path(f"roundtrip_anim_clip_anim_list_{fps_dependent[1]}_{roundtrip}.ycd.xml")
            success = export_ycd(obj, str(out_path))
            assert success
            assert out_path.exists()

            _check_exported_ycd(out_path)

            curr_input_path = out_path

    @pytest.mark.parametrize("yft_path, yft_path_str", glob_assets("yft"))
    def test_import_export_yft_link_attachment_calculation(yft_path: Path, yft_path_str: str):
        if "sollumz_cube" in yft_path_str:
            return

        obj = import_yft(yft_path_str)
        assert obj is not None

        out_path = tmp_path(yft_path.name)
        success = export_yft(obj, str(out_path))
        assert success
        assert out_path.exists()

        input_tree = ET.ElementTree()
        input_tree.parse(yft_path)
        input_root = input_tree.getroot()

        output_tree = ET.ElementTree()
        output_tree.parse(out_path)
        output_root = output_tree.getroot()

        XPATH_ROOT_CG_OFFSET = "./Physics/LOD1/PositionOffset"
        XPATH_LINK_ATTACHMENTS = "./Physics/LOD1/Transforms/Item"
        XPATH_GROUPS = "./Physics/LOD1/Groups/Item"
        XPATH_CHILDREN = "./Physics/LOD1/Children/Item"

        input_root_cg_offset = input_root.find(XPATH_ROOT_CG_OFFSET)
        input_link_attachments = input_root.findall(XPATH_LINK_ATTACHMENTS)
        input_groups = input_root.findall(XPATH_GROUPS)
        input_children = input_root.findall(XPATH_CHILDREN)
        output_root_cg_offset = output_root.find(XPATH_ROOT_CG_OFFSET)
        output_link_attachments = output_root.findall(XPATH_LINK_ATTACHMENTS)
        output_groups = output_root.findall(XPATH_GROUPS)
        output_children = output_root.findall(XPATH_CHILDREN)

        if True:
            import copy
            debug_xml_path = tmp_path(yft_path.stem + "_debug.xml")
            debug_root = ET.Element("FragmentDebug")
            debug_input_link_attachments = ET.SubElement(debug_root, "InputLinkAttachments")
            for i in input_link_attachments:
                debug_input_link_attachments.append(copy.deepcopy(i))
            debug_output_link_attachments = ET.SubElement(debug_root, "OutputLinkAttachments")
            for o in output_link_attachments:
                debug_output_link_attachments.append(copy.deepcopy(o))
            debug_tree = ET.ElementTree(debug_root)
            ET.indent(debug_tree, space="  ", level=0)
            debug_tree.write(debug_xml_path)

        def build_children_mapping(groups, children):
            mapping = {}
            for child_index, child in enumerate(children):
                bone_tag = int(child.find("BoneTag").get("value"))
                group_index = int(child.find("GroupIndex").get("value"))
                group_name = groups[group_index].find("Name").text
                child_key = (group_name, bone_tag)
                mapping[child_key] = child_index
            return mapping

        input_children_mapping = build_children_mapping(input_groups, input_children)
        output_children_mapping = build_children_mapping(output_groups, output_children)

        # Sanity checks
        assert input_children_mapping.keys() == output_children_mapping.keys(), "Children are different"
        assert len(input_link_attachments) == len(output_link_attachments), "Different number of link attachments"
        assert len(output_link_attachments) == len(output_children), "Number of link attachments different to number of children"

        input_root_cg_offset_vec = elem_to_vec(input_root_cg_offset)
        output_root_cg_offset_vec = elem_to_vec(output_root_cg_offset)
        assert_allclose(
            output_root_cg_offset_vec, input_root_cg_offset_vec,
            atol=1e-4,
            err_msg=(f"Fragment '{yft_path}', calculated root CG offset does not match original.\n"
                     f"   diff={output_root_cg_offset_vec - input_root_cg_offset_vec}")
        )

        for child_key in input_children_mapping.keys():
            input_child_index = input_children_mapping[child_key]
            output_child_index = output_children_mapping[child_key]

            input_link_attachment = input_link_attachments[input_child_index]
            output_link_attachment = output_link_attachments[output_child_index]

            input_values = np.fromstring(input_link_attachment.text, dtype=np.float32, sep=" ")
            output_values = np.fromstring(output_link_attachment.text, dtype=np.float32, sep=" ")

            input_values = input_values.reshape((4, 4))
            output_values = output_values.reshape((4, 4))

            mismatched = ~np.isclose(output_values, input_values, atol=1e-4)
            assert_allclose(
                output_values, input_values,
                atol=1e-4,
                err_msg=(f"Fragment '{yft_path}', calculated link attachment for child {child_key} does not match original.\n"
                         f"   {input_values=}\n"
                         f"   {output_values=}\n"
                         f"   {mismatched=}\n"
                         f"   {input_values[mismatched]=}\n"
                         f"   {output_values[mismatched]=}\n"
                         f"   diff={output_values[mismatched] - input_values[mismatched]}\n"
                         f"   root CG diff={output_root_cg_offset_vec - input_root_cg_offset_vec}")
            )
            #
            # err_msg=(f"Fragment '{yft_path}', calculated link attachment for child {child_key} does not match original.\n"
            #          f"   {input_values=}\n"
            #          f"   {output_values=}\n"
            #          f"   {mismatched=}\n"
            #          f"   {input_values[mismatched]=}\n"
            #          f"   {output_values[mismatched]=}\n"
            #          f"   diff={output_values[mismatched] - input_values[mismatched]}\n"
            #          f"   root CG diff={output_root_cg_offset_vec - input_root_cg_offset_vec}")
            # print(err_msg)

    @pytest.mark.parametrize("yft_path, yft_path_str", glob_assets("yft"))
    def test_import_export_yft_bounds_centroid_and_mass_properties_calculation(yft_path: Path, yft_path_str: str):
        if "sollumz_cube" in yft_path_str:
            return

        obj = import_yft(yft_path_str)
        assert obj is not None

        out_path = tmp_path(yft_path.name)
        success = export_yft(obj, str(out_path))
        assert success
        assert out_path.exists()

        input_tree = ET.ElementTree()
        input_tree.parse(yft_path)
        input_root = input_tree.getroot()

        output_tree = ET.ElementTree()
        output_tree.parse(out_path)
        output_root = output_tree.getroot()

        XPATH_BOUNDS = "./Physics/LOD1/Archetype/Bounds/Children/Item"
        XPATH_GROUPS = "./Physics/LOD1/Groups/Item"
        XPATH_CHILDREN = "./Physics/LOD1/Children/Item"

        input_bounds = input_root.findall(XPATH_BOUNDS)
        output_bounds = output_root.findall(XPATH_BOUNDS)

        input_groups = input_root.findall(XPATH_GROUPS)
        input_children = input_root.findall(XPATH_CHILDREN)
        output_groups = output_root.findall(XPATH_GROUPS)
        output_children = output_root.findall(XPATH_CHILDREN)

        assert len(input_bounds) == len(output_bounds), "Different number of bounds"

        def build_children_mapping(groups, children, bounds):
            # Not a perfect mapping, there can be multiple children in the same group and for the same bone.
            # Included bound type in key to try making it more unique.
            # Let's hope it's good enough for now...
            mapping = {}
            for child_index, child in enumerate(children):
                bone_tag = int(child.find("BoneTag").get("value"))
                group_index = int(child.find("GroupIndex").get("value"))
                group_name = groups[group_index].find("Name").text
                bound_type = bounds[child_index].get("type")
                child_key = (group_name, bone_tag, bound_type)
                mapping[child_key] = child_index
            return mapping

        input_children_mapping = build_children_mapping(input_groups, input_children, input_bounds)
        output_children_mapping = build_children_mapping(output_groups, output_children, output_bounds)

        for child_key in input_children_mapping.keys():
            input_child_index = input_children_mapping[child_key]
            output_child_index = output_children_mapping[child_key]

            input_bound = input_bounds[input_child_index]
            output_bound = output_bounds[output_child_index]

            input_type = input_bound.get("type")
            output_type = output_bound.get("type")
            assert input_type == output_type, "Different bound type"


            input_centroid = elem_to_vec(input_bound.find("BoxCenter"))
            output_centroid = elem_to_vec(output_bound.find("BoxCenter"))

            input_radius = elem_to_float(input_bound.find("SphereRadius"))
            output_radius = elem_to_float(output_bound.find("SphereRadius"))

            input_cg = elem_to_vec(input_bound.find("SphereCenter"))
            output_cg = elem_to_vec(output_bound.find("SphereCenter"))

            input_volume = elem_to_float(input_bound.find("Volume"))
            output_volume = elem_to_float(output_bound.find("Volume"))

            input_inertia = elem_to_vec(input_bound.find("Inertia"))
            output_inertia = elem_to_vec(output_bound.find("Inertia"))

            input_margin = elem_to_float(input_bound.find("Margin"))
            output_margin = elem_to_float(output_bound.find("Margin"))

            atol = 1e-3
            if input_type == "Geometry" or input_type == "GeometryBVH":
                input_bb_center = elem_to_vec(input_bound.find("GeometryCenter"))
                output_bb_center = elem_to_vec(output_bound.find("GeometryCenter"))

                assert_allclose(
                    input_bb_center, output_bb_center,
                    atol=atol,
                    err_msg=(f"Fragment '{yft_path}', calculated geometry bounding-box center does not match original.\n"
                             f"   diff={output_bb_center - input_bb_center}\n"
                             f"   bound type={output_type}\n"
                             f"   child={child_key}")
                )
                # skip the rest for now
                # continue

            assert_allclose(
                output_centroid, input_centroid,
                atol=atol,
                err_msg=(f"Fragment '{yft_path}', calculated centroid does not match original.\n"
                         f"   diff={output_centroid - input_centroid}\n"
                         f"   bound type={output_type}\n"
                         f"   child={child_key}")
            )

            assert_allclose(
                output_radius, input_radius,
                atol=atol,
                err_msg=(f"Fragment '{yft_path}', calculated radius around centroid does not match original.\n"
                         f"   diff={output_radius - input_radius}\n"
                         f"   bound type={output_type}\n"
                         f"   child={child_key}")
            )

            assert_allclose(
                output_cg, input_cg,
                atol=atol,
                err_msg=(f"Fragment '{yft_path}', calculated CG does not match original.\n"
                         f"   diff={output_cg - input_cg}\n"
                         f"   bound type={output_type}\n"
                         f"   child={child_key}")
            )

            assert_allclose(
                output_volume, input_volume,
                atol=atol,
                err_msg=(f"Fragment '{yft_path}', calculated volume does not match original.\n"
                         f"   diff={output_volume - input_volume}\n"
                         f"   bound type={output_type}\n"
                         f"   child={child_key}")
            )

            assert_allclose(
                output_inertia, input_inertia,
                atol=atol,
                err_msg=(f"Fragment '{yft_path}', calculated inertia does not match original.\n"
                         f"   diff={output_inertia - input_inertia}\n"
                         f"   bound type={output_type}\n"
                         f"   child={child_key}")
            )

            assert_allclose(
                output_margin, input_margin,
                atol=atol,
                err_msg=(f"Fragment '{yft_path}', calculated margin does not match original.\n"
                         f"   diff={output_margin - input_margin}\n"
                         f"   bound type={output_type}\n"
                         f"   child={child_key}")
            )
