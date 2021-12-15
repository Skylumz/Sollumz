import bpy
from mathutils import Matrix, Vector
from ..tools.drawablehelper import get_drawable_geometries, join_drawable_geometries
from ..resources.fragment import YFT
from ..ydr.ydrimport import drawable_to_obj, shadergroup_to_materials, light_to_obj
from ..ybn.ybnimport import composite_to_obj
from ..sollumz_properties import SOLLUMZ_UI_NAMES, SollumType
from ..tools.blenderhelper import split_object_by_vertex_groups


def get_bound_object_from_child_index(bobj, index):
    for bound in bobj.children:
        if bound.creation_index == index:
            return bound


def create_lod_obj(lod, filepath, materials):
    has_bounds = True if lod.archetype.bounds else False

    if not has_bounds:
        return

    bobj = composite_to_obj(lod.archetype.bounds,
                            SOLLUMZ_UI_NAMES[SollumType.BOUND_COMPOSITE], True)
    lobj = bpy.data.objects.new(lod.tag_name, None)
    lobj.empty_display_size = 0
    bpy.context.collection.objects.link(lobj)

    lobj.sollum_type = SollumType.FRAGLOD
    lobj.lod_properties.unknown_14 = lod.unknown_14
    lobj.lod_properties.unknown_18 = lod.unknown_18
    lobj.lod_properties.unknown_1c = lod.unknown_1c
    lobj.lod_properties.position_offset = lod.position_offset
    lobj.lod_properties.unknown_40 = lod.unknown_40
    lobj.lod_properties.unknown_40 = lod.unknown_50
    lobj.lod_properties.damping_linear_c = lod.damping_linear_c
    lobj.lod_properties.damping_linear_v = lod.damping_linear_v
    lobj.lod_properties.damping_linear_v2 = lod.damping_linear_v2
    lobj.lod_properties.damping_angular_c = lod.damping_angular_c
    lobj.lod_properties.damping_angular_v = lod.damping_angular_v
    lobj.lod_properties.damping_angular_v2 = lod.damping_angular_v2
    # archetype properties
    lobj.lod_properties.archetype_name = lod.archetype.name
    lobj.lod_properties.archetype_mass = lod.archetype.mass
    lobj.lod_properties.archetype_unknown_48 = lod.archetype.unknown_48
    lobj.lod_properties.archetype_unknown_4c = lod.archetype.unknown_4c
    lobj.lod_properties.archetype_unknown_50 = lod.archetype.unknown_50
    lobj.lod_properties.archetype_unknown_54 = lod.archetype.unknown_54
    lobj.lod_properties.archetype_inertia_tensor = lod.archetype.inertia_tensor

    gobjs = []
    for idx, group in enumerate(lod.groups):
        gobj = bpy.data.objects.new(group.name + "_group", None)
        gobj.empty_display_size = 0
        gobj.sollum_type = SollumType.FRAGGROUP
        bpy.context.collection.objects.link(gobj)

        gobj.group_properties.name = group.name
        gobj.group_properties.glass_window_index = group.glass_window_index
        gobj.group_properties.glass_flags = group.glass_flags
        gobj.group_properties.strength = group.strength
        gobj.group_properties.force_transmission_scale_up = group.force_transmission_scale_up
        gobj.group_properties.force_transmission_scale_down = group.force_transmission_scale_down
        gobj.group_properties.joint_stiffness = group.joint_stiffness
        gobj.group_properties.min_soft_angle_1 = group.min_soft_angle_1
        gobj.group_properties.max_soft_angle_1 = group.max_soft_angle_1
        gobj.group_properties.max_soft_angle_2 = group.max_soft_angle_2
        gobj.group_properties.max_soft_angle_3 = group.max_soft_angle_3
        gobj.group_properties.rotation_speed = group.rotation_speed
        gobj.group_properties.rotation_strength = group.rotation_strength
        gobj.group_properties.restoring_max_torque = group.restoring_max_torque
        gobj.group_properties.latch_strength = group.latch_strength
        gobj.group_properties.mass = group.mass
        gobj.group_properties.min_damage_force = group.min_damage_force
        gobj.group_properties.damage_health = group.damage_health
        gobj.group_properties.unk_float_5c = group.unk_float_5c
        gobj.group_properties.unk_float_60 = group.unk_float_60
        gobj.group_properties.unk_float_64 = group.unk_float_64
        gobj.group_properties.unk_float_68 = group.unk_float_68
        gobj.group_properties.unk_float_6c = group.unk_float_6c
        gobj.group_properties.unk_float_70 = group.unk_float_70
        gobj.group_properties.unk_float_74 = group.unk_float_74
        gobj.group_properties.unk_float_78 = group.unk_float_78
        gobj.group_properties.unk_float_a8 = group.unk_float_a8

        try:
            if group.parent_index == 255:
                gparent = lobj
            else:
                gparent = gobjs[group.parent_index]
            gobj.parent = gparent
        except:
            pass

        gobjs.append(gobj)

    # attach groups to bound objects
    for idx, child in enumerate(lod.children):
        cobj = bpy.data.objects.new(group.name + "_group", None)
        cobj.empty_display_size = 0
        cobj.sollum_type = SollumType.FRAGCHILD
        bpy.context.collection.objects.link(cobj)

        gobj = gobjs[child.group_index]

        cobj.parent = gobj
        cobj.name = gobj.name.replace("_group", "_child")
        cobj.child_properties.group = gobj
        cobj.child_properties.bone_tag = child.bone_tag
        cobj.child_properties.pristine_mass = child.pristine_mass
        cobj.child_properties.damaged_mass = child.damaged_mass
        cobj.child_properties.unk_vec = child.unk_vec
        cobj.child_properties.inertia_tensor = child.inertia_tensor

        bound = get_bound_object_from_child_index(bobj, idx)
        if bound:
            bound.parent = gobj
            bound.name = gobj.name.replace("_group", "_col")

        if len(child.drawable.drawable_models_high) > 0:
            a = drawable_to_obj(
                child.drawable, filepath, f"Drawable{idx}", None, materials)
            a.parent = cobj

        if lod.transforms:
            transform = lod.transforms[idx].value
            a = transform[3][0] + lod.position_offset.x
            b = transform[3][1] + lod.position_offset.y
            c = transform[3][2] + lod.position_offset.z
            transform[3][0] = a
            transform[3][1] = b
            transform[3][2] = c
            cobj.matrix_basis = transform.transposed()

    bpy.data.objects.remove(bobj, do_unlink=True)

    return lobj


def fragment_to_obj(fragment, filepath):
    fobj = bpy.data.objects.new(fragment.name, None)
    fobj.empty_display_size = 0
    fobj.sollum_type = SollumType.FRAGMENT
    bpy.context.collection.objects.link(fobj)

    fobj.fragment_properties.unk_b0 = fragment.unknown_b0
    fobj.fragment_properties.unk_b8 = fragment.unknown_b8
    fobj.fragment_properties.unk_bc = fragment.unknown_bc
    fobj.fragment_properties.unk_c0 = fragment.unknown_c0
    fobj.fragment_properties.unk_c4 = fragment.unknown_c4
    fobj.fragment_properties.unk_cc = fragment.unknown_cc
    fobj.fragment_properties.gravity_factor = fragment.gravity_factor
    fobj.fragment_properties.buoyancy_factor = fragment.buoyancy_factor

    materials = None
    if fragment.drawable:
        materials = shadergroup_to_materials(
            fragment.drawable.shader_group, filepath)
        fragment.drawable.lights = fragment.lights
        dobj = drawable_to_obj(
            fragment.drawable, filepath, fragment.drawable.name, None, materials)
        dobj.parent = fobj

    if len(fragment.physics.lod1.groups) > 0:
        lobj = create_lod_obj(fragment.physics.lod1, filepath, materials)
        lobj.lod_properties.type = 1
        lobj.parent = fobj
    if len(fragment.physics.lod2.groups) > 0:
        lobj = create_lod_obj(fragment.physics.lod2, filepath, materials)
        lobj.lod_properties.type = 2
        lobj.parent = fobj
    if len(fragment.physics.lod3.groups) > 0:
        lobj = create_lod_obj(fragment.physics.lod3, filepath, materials)
        lobj.lod_properties.type = 3
        lobj.parent = fobj

    allbmodels = []
    for child in dobj.children:
        allbmodels.append(child)
    allfmodels = fragment.drawable.all_models

    pose = fragment.bones_transforms
    if pose:
        modeltransforms = []
        pbc = len(pose)
        for i in range(pbc):
            modeltransforms.append(pose[i].value)

        for i in range(len(allfmodels)):
            model = allfmodels[i]
            bmodel = allbmodels[i]

            boneidx = model.bone_index
            m = modeltransforms[boneidx] if boneidx < len(
                modeltransforms) else Matrix()

            if not model.has_skin:
                bmodel.matrix_basis = m

    return fobj


def get_fragment_drawable(fragment):
    for child in fragment.children:
        if child.sollum_type == SollumType.DRAWABLE:
            return child


def import_yft(filepath, import_settings):
    yft_xml = YFT.from_xml_file(filepath)
    fobj = fragment_to_obj(yft_xml, filepath)

    if import_settings.split_by_bone:
        dobj = get_fragment_drawable(fobj)

        if not dobj:
            raise Exception("No fragment drawable found to split by bone!")

        for child in dobj.children:
            if child.sollum_type == SollumType.DRAWABLE_MODEL:
                join_drawable_geometries(child)
                geo = get_drawable_geometries(child)[0]
                split_object_by_vertex_groups(geo)
