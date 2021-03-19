import bpy
from bpy_extras.io_utils import ExportHelper
from bpy.types import Operator
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom
from mathutils import Vector
import os 
import sys 
import shutil
import ntpath
from datetime import datetime 

def prettify(elem):
    rough_string = tostring(elem, encoding='UTF-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent=" ")

def get_obj_children(obj):
    children = [] 
    for ob in bpy.data.objects: 
        if ob.parent == obj: 
            children.append(ob) 
    return children 

def order_vertex_list(list, vlayout):
    layout_map = {
        "Position": 0,
        "Normal": 1,
        "Colour0": 2,
        "Colour1": 3,
        "TexCoord0": 4,
        "TexCoord1": 5,
        "TexCoord2": 6,
        "TexCoord3": 7,
        "TexCoord4": 8,
        "TexCoord5": 9,
        "Tangent": 10,
        "BlendWeights": 11,
        "BlendIndices": 12,
    }

    newlist = []

    for i in range(len(vlayout)):
        layout_key = layout_map[vlayout[i]]
        if layout_key != None:
            newlist.append(list[layout_key])
        else:
            print('Incorrect layout element', vlayout[i])

    if (len(newlist) != len(vlayout)):
        print('Incorrect layout parse')

    return newlist

def vector_tostring(vector):
    try:
        string = ""
        string += str(vector.x) + " "   
        string += str(vector.y)
        if(hasattr(vector, "z")):   
            string += " " + str(vector.z) 
        else:
            string += " "
            
        return string 
    except:
        return None

def meshloopcolor_tostring(color):
    try:
        string = ""
        string += str(round(color[0] * 255)) + " "
        string += str(round(color[1] * 255)) + " "
        string += str(round(color[2] * 255)) + " "
        string += str(round(color[3] * 255))
        return string 
    except:
        return None
    
def process_uv(uv):
    u = uv[0]
    v = (uv[1] - 1.0) * -1

    return [u, v]

def get_vertex_string(mesh, vlayout):
    
    allstrings = []
    allstrings.append("\n") #makes xml a little prettier
    
    vertamount = len(mesh.vertices)
    vertices = [None] * vertamount
    normals = [None] * vertamount
    clr = [None] * vertamount
    clr1 = [None] * vertamount
    texcoords = {}
    tangents = [None] * vertamount
    blendw = [None] * vertamount
    blendi = [None] * vertamount

    for i in range(6):
        texcoords[i] = [None] * vertamount       
    
    mesh.calc_tangents()

    clr0_layer = None 
    if(mesh.vertex_colors == None):
        clr0_layer = mesh.vertex_colors.new()
    else:
        clr0_layer = mesh.vertex_colors[0]

    for uv_layer_id in range(len(mesh.uv_layers)):
        uv_layer = mesh.uv_layers[uv_layer_id].data
        for poly in mesh.polygons:
            for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                vi = mesh.loops[loop_index].vertex_index
                uv = process_uv(uv_layer[loop_index].uv)
                u = uv[0]
                v = uv[1]
                fixed_uv = Vector((u, v))
                texcoords[uv_layer_id][vi] = fixed_uv

            
    for poly in mesh.polygons:
        for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
            vi = mesh.loops[loop_index].vertex_index
            vertices[vi] = mesh.vertices[vi].co
            normals[vi] = mesh.vertices[vi].normal
            clr[vi] = clr0_layer.data[loop_index].color
            tangents[vi] = mesh.loops[loop_index].tangent 
            #FIXME: write actual blend weights and indices
            blendw[vi] = "0 0 255 0"
            blendi[vi] = "0 0 0 0"
            
    for i in range(len(vertices)):
        vstring = ""
        tlist = []
        tlist.append(vector_tostring(vertices[i])) 
        tlist.append(vector_tostring(normals[i])) 
        tlist.append(meshloopcolor_tostring(clr[i])) 
        tlist.append(meshloopcolor_tostring(clr1[i]))
        tlist.append(vector_tostring(texcoords[0][i])) 
        tlist.append(vector_tostring(texcoords[1][i]))
        tlist.append(vector_tostring(texcoords[2][i]))
        tlist.append(vector_tostring(texcoords[3][i]))
        tlist.append(vector_tostring(texcoords[4][i]))
        tlist.append(vector_tostring(texcoords[5][i]))
        tlist.append(vector_tostring(tangents[i]))
        tlist.append(blendw[i])
        tlist.append(blendi[i])
        
        layoutlist = order_vertex_list(tlist, vlayout)
        
        vstring = " " * 5
        for l in layoutlist:
            if(l != None):
                vstring += l 
                vstring += " " * 3
            else:
                print('Layoutlist elem is None!')
        vstring += "\n" 
        allstrings.append(vstring) 
            
    vertex_string = ""
    for s in allstrings:       
        vertex_string += s
    
    return vertex_string

def get_index_string(mesh):
    
    index_string = ""
    
    for poly in mesh.polygons:
        for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
            index_string += str(mesh.loops[loop_index].vertex_index) + " "
            
    return index_string

def fix_shader_name(name, no_extension = False): #because blender renames everything to .00X
    newname = ""
    n = name.split(".")
    if(len(n) == 3):
        newname += n[0]
        newname += "."  
        newname += n[1]
    else:
        newname = name
    if(no_extension):
        newname = newname[:-4]
    return newname

def get_vertex_layout(shader):
    shader = fix_shader_name(shader) 

    PT = ["Position", "TexCoord0"]
    PCTT = ["Position", "Colour0", "TexCoord0", "TexCoord1"]
    PNC = ["Position", "Normal", "Colour0"]
    PNCT = ["Position", "Normal", "Colour0", "TexCoord0"]
    PNCCT = ["Position", "Normal", "Colour0", "Colour1", "TexCoord0"]
    PNCTX = ["Position", "Normal", "Colour0", "TexCoord0", "Tangent"]
    PNCTTX = ["Position", "Normal", "Colour0", "TexCoord0", "TexCoord1", "Tangent"]
    PBBCCT = ["Position", "BlendWeights", "BlendIndices", "Colour0", "Colour1", "TexCoord0"]
    PBBNCTX = ["Position", "BlendWeights", "BlendIndices", "Normal", "Colour0", "TexCoord0", "Tangent"]
    PBBNCTTX = ["Position", "BlendWeights", "BlendIndices", "Normal", "Colour0", "TexCoord0", "TexCoord1", "Tangent"]
    PBBNCTT = ["Position", "BlendWeights", "BlendIndices", "Normal", "Colour0", "TexCoord0", "TexCoord1"]
    PBBNCT = ["Position", "BlendWeights", "BlendIndices", "Normal", "Colour0", "TexCoord0"]
    PNCCTTX = ["Position", "Normal", "Colour0", "Colour1", "TexCoord0", "TexCoord1", "Tangent"]
    PNCTTTX = ["Position", "Normal", "Colour0", "TexCoord0", "TexCoord1", "TexCoord2", "Tangent"]
    PNCCT3TX = ["Position", "Normal", "Colour0", "Colour1", "TexCoord0", "TexCoord3", "Tangent"]
    PNCTT3TX = ["Position", "Normal", "Colour0", "TexCoord0", "TexCoord1", "TexCoord3", "Tangent"]
    PBBNCTTT = ["Position", "BlendWeights", "BlendIndices", "Normal", "Colour0", "TexCoord0", "TexCoord1", "TexCoord2"]
    PNCCTT = ["Position", "Normal", "Colour0", "Colour1", "TexCoord0", "TexCoord1"]
    PNCCTTTT = ["Position", "Normal", "Colour0", "Colour1", "TexCoord0", "TexCoord1", "TexCoord2", "TexCoord3"]
    PNCCTTTX = ["Position", "Normal", "Colour0", "Colour1", "TexCoord0", "TexCoord1", "TexCoord3", "Tangent"]
    PNCT4T5TX = ["Position", "Normal", "Colour0", "TexCoord0", "TexCoord4", "TexCoord5", "Tangent"]
    PNCTT4T5TX = ["Position", "Normal", "Colour0", "TexCoord0", "TexCoord1", "TexCoord4", "TexCoord5", "Tangent"]

    if shader == "normal.sps": return PNCTX
    elif shader == "normal_spec.sps": return PNCTX
    elif shader == "normal_spec_detail.sps": return PNCTX
    elif shader == "normal_alpha.sps": return PNCTX
    elif shader == "default_spec.sps": return PNCT
    elif shader == "None": return PNCT
    elif shader == "emissive_alpha.sps": return PNCT
    elif shader == "spec.sps": return PNCT
    elif shader == "emissivestrong.sps": return PNCT
    elif shader == "cutout.sps": return PNCT
    elif shader == "default.sps": return PNCT
    elif shader == "spec_const.sps": return PNCT
    elif shader == "normal_decal.sps": return PNCTX
    elif shader == "decal.sps": return PNCT
    elif shader == "terrain_cb_w_4lyr.sps": return PNCCTX
    elif shader == "terrain_cb_w_4lyr_2tex_blend_pxm_spm.sps": return PNCCTTTX
    elif shader == "normal_detail.sps": return  PNCTX
    elif shader == "normal_spec_pxm.sps": return PNCTTTX
    elif shader == "terrain_cb_w_4lyr_pxm_spm.sps": return PNCCT3TX
    elif shader == "normal_tnt.sps": return  PNCTX
    elif shader == "glass_pv.sps": return  PNCTX
    elif shader == "normal_spec_decal.sps": return  PNCTX
    elif shader == "normal_decal_tnt.sps": return  PNCTX
    elif shader == "default_detail.sps": return  PNCT
    elif shader == "emissivenight.sps": return  PNCT
    elif shader == "decal_glue.sps": return  PNCT
    elif shader == "alpha.sps": return  PNCT
    elif shader == "emissive.sps": return  PNCT
    elif shader == "normal_decal_pxm.sps": return PNCTTTX
    elif shader == "normal_cutout.sps": return  PNCTX
    elif shader == "normal_spec_tnt.sps": return  PNCTX
    elif shader == "terrain_cb_w_4lyr_pxm.sps": return PNCCT3TX
    elif shader == "cpv_only.sps": return PNC
    elif shader == "emissive_speclum.sps": return  PNCT
    elif shader == "emissivestrong_alpha.sps": return  PNCT
    elif shader == "glass_env.sps": return  PNCTX
    elif shader == "mirror_decal.sps": return  PNCTX
    elif shader == "decal_normal_only.sps": return  PNCTX
    elif shader == "decal_dirt.sps": return  PNCT
    elif shader == "normal_spec_alpha.sps": return  PNCTX
    elif shader == "mirror_default.sps": return  PNCTX
    elif shader == "glass_normal_spec_reflect.sps": return  PNCTX
    elif shader == "cable.sps": return  PNCT
    elif shader == "normal_spec_cutout.sps": return  PNCTX
    elif shader == "normal_spec_reflect.sps": return  PNCTX
    elif shader == "radar.sps": return PCTT
    elif shader == "spec_tnt.sps": return  PNCT
    elif shader == "glass_pv_env.sps": return  PNCTX
    elif shader == "spec_reflect.sps": return  PNCT
    elif shader == "water_fountain.sps": return PT
    elif shader == "water_riverfoam.sps": return  PNCTX
    elif shader == "cutout_fence.sps": return  PNCT
    elif shader == "glass_spec.sps": return  PNCT
    elif shader == "emissivenight_alpha.sps": return  PNCT
    elif shader == "emissivenight_geomnightonly.sps": return  PNCT
    elif shader == "water_poolenv.sps": return PT
    elif shader == "normal_spec_detail_tnt.sps": return  PNCTX
    elif shader == "spec_alpha.sps": return  PNCT
    elif shader == "mirror_crack.sps": return  PNCTX
    elif shader == "glass_emissive.sps": return  PNCTX
    elif shader == "glass.sps": return  PNCTX
    elif shader == "normal_spec_decal_tnt.sps": return  PNCTX
    elif shader == "decal_tnt.sps": return  PNCT
    elif shader == "normal_pxm.sps": return PNCTTTX
    elif shader == "terrain_cb_w_4lyr_spec_pxm.sps": return PNCCT3TX
    elif shader == "emissive_tnt.sps": return  PNCT
    elif shader == "default_tnt.sps": return  PNCT
    elif shader == "terrain_cb_w_4lyr_spec_int.sps": return  PNCCTX
    elif shader == "cutout_fence_normal.sps": return  PNCTX
    elif shader == "normal_cutout_tnt.sps": return  PNCTX
    elif shader == "normal_pxm_tnt.sps": return PNCTTTX
    elif shader == "reflect.sps": return  PNCT
    elif shader == "spec_decal.sps": return  PNCT
    elif shader == "normal_reflect.sps": return  PNCTX
    elif shader == "emissive_additive_uv_alpha.sps": return  PNCT
    elif shader == "glass_reflect.sps": return  PNCT
    elif shader == "decal_spec_only.sps": return  PNCT
    elif shader == "emissive_additive_alpha.sps": return  PNCT
    elif shader == "normal_spec_emissive.sps": return  PNCTX
    elif shader == "decal_amb_only.sps": return  PNCT
    elif shader == "trees.sps": return PNCCT
    elif shader == "decal_diff_only_um.sps": return PBBCCT
    elif shader == "decal_shadow_only.sps": return  PNCTX
    elif shader == "normal_spec_decal_detail.sps": return  PNCTX
    elif shader == "reflect_decal.sps": return  PNCT
    elif shader == "normal_spec_decal_pxm.sps": return PNCTTX
    elif shader == "normal_spec_pxm_tnt.sps": return PNCTTTX
    elif shader == "normal_spec_cutout_tnt.sps": return  PNCTX
    elif shader == "glass_emissive_alpha.sps": return  PNCTX
    elif shader == "normal_diffspec.sps": return  PNCTX
    elif shader == "decal_emissive_only.sps": return  PNCT
    elif shader == "normal_spec_reflect_decal.sps": return  PNCTX
    elif shader == "emissive_alpha_tnt.sps": return  PNCT
    elif shader == "gta_radar.sps": return PCTT
    elif shader == "gta_normal.sps": return  PNCTX
    elif shader == "gta_default.sps": return  PNCT
    elif shader == "clouds_animsoft.sps": return  PNCTX
    elif shader == "clouds_altitude.sps": return  PNCTX
    elif shader == "clouds_fast.sps": return  PNCTX
    elif shader == "clouds_anim.sps": return  PNCTX
    elif shader == "clouds_soft.sps": return  PNCTX
    elif shader == "clouds_fog.sps": return  PNCTX
    elif shader == "default_um.sps": return PNCCT
    elif shader == "trees_lod.sps": return PNCCT
    elif shader == "trees_lod2.sps": return PNCCTTTT
    elif shader == "normal_spec_reflect_alpha.sps": return  PNCTX
    elif shader == "gta_reflect_alpha.sps": return  PNCT
    elif shader == "gta_spec.sps": return  PNCT
    elif shader == "grass_fur.sps": return  PNCTX
    elif shader == "default_terrain_wet.sps": return  PNCT
    elif shader == "terrain_cb_w_4lyr_spec.sps": return  PNCCTX
    elif shader == "emissive_clip.sps": return  PNCT
    elif shader == "grass_fur_mask.sps": return PNCTTTX
    elif shader == "cutout_tnt.sps": return  PNCT
    elif shader == "terrain_cb_w_4lyr_2tex_blend.sps": return PNCCTTX
    elif shader == "cloth_normal_spec.sps": return  PNCTX
    elif shader == "distance_map.sps": return  PNCTX
    elif shader == "cloth_spec_alpha.sps": return  PNCTX
    elif shader == "terrain_cb_w_4lyr_2tex_blend_pxm.sps": return PNCCTTTX
    elif shader == "terrain_cb_w_4lyr_cm_pxm.sps": return PNCTT3TX
    elif shader == "water_riverlod.sps": return  PNCT
    elif shader == "weapon_normal_spec_tnt.sps": return PBBNCTX
    elif shader == "trees_normal_spec.sps": return  PNCCTX
    elif shader == "terrain_cb_w_4lyr_cm_pxm_tnt.sps": return PNCTT3TX
    elif shader == "terrain_cb_w_4lyr_2tex.sps": return PNCCTTX
    elif shader == "normal_spec_cubemap_reflect.sps": return  PNCTX
    elif shader == "terrain_cb_w_4lyr_lod.sps": return PNCCT
    elif shader == "terrain_cb_w_4lyr_2tex_blend_lod.sps": return PNCCTT
    elif shader == "glass_displacement.sps": return  PNCTX
    elif shader == "trees_normal_diffspec.sps": return  PNCCTX
    elif shader == "cutout_um.sps": return PNCCT
    elif shader == "default_noedge.sps": return  PNCT
    elif shader == "glass_breakable.sps": return PNCTTX
    elif shader == "trees_normal.sps": return  PNCCTX
    elif shader == "normal_reflect_alpha.sps": return  PNCTX
    elif shader == "normal_tnt_alpha.sps": return  PNCTX
    elif shader == "reflect_alpha.sps": return PBBNCT
    elif shader == "ptfx_model.sps": return  PNCT
    elif shader == "decal_emissivenight_only.sps": return  PNCT
    elif shader == "normal_diffspec_detail.sps": return  PNCTX
    elif shader == "normal_diffspec_detail_dpm.sps": return PNCT4T5TX
    elif shader == "normal_spec_detail_dpm_vertdecal_tnt.sps": return PNCTT4T5TX
    elif shader == "normal_spec_um.sps": return  PNCCTX
    elif shader == "normal_um.sps": return  PNCCTX
    elif shader == "normal_um_tnt.sps": return PBBNCTTX
    elif shader == "normal_spec_wrinkle.sps": return PBBNCTX
    elif shader == "cutout_spec_tnt.sps": return  PNCT
    elif shader == "hash_7D3957DA": return  PNCT
    elif shader == "normal_spec_tnt_pxm.sps": return PNCTTTX
    elif shader == "parallax_specmap.sps": return  PNCTX
    elif shader == "spec_reflect_alpha.sps": return  PNCT
    elif shader == "normal_cubemap_reflect.sps": return  PNCTX
    elif shader == "vehicle_tire.sps": return PNCTTX
    elif shader == "weapon_normal_spec_detail_palette.sps": return PBBNCTX
    elif shader == "weapon_normal_spec_detail_tnt.sps": return PBBNCTX
    elif shader == "weapon_normal_spec_palette.sps": return PBBNCTX
    elif shader == "weapon_emissivestrong_alpha.sps": return PBBNCT
    elif shader == "weapon_normal_spec_alpha.sps": return PBBNCTX
    elif shader == "weapon_emissive_tnt.sps": return PBBNCT
    elif shader == "weapon_normal_spec_cutout_palette.sps": return PBBNCTX
    elif shader == "terrain_cb_w_4lyr_spec_int_pxm.sps": return PNCCT3TX
    elif shader == "cloth_default.sps": return  PNCT
    elif shader == "cloth_spec_cutout.sps": return  PNCTX
    elif shader == "normal_decal_pxm_tnt.sps": return PNCTTTX
    elif shader == "vehicle_lightsemissive.sps": return PBBNCTT
    elif shader == "vehicle_interior2.sps": return PBBNCT
    elif shader == "vehicle_badges.sps": return PBBNCTTX
    elif shader == "vehicle_mesh.sps": return PBBNCTTX
    elif shader == "vehicle_interior.sps": return PBBNCTX
    elif shader == "vehicle_licenseplate.sps": return PBBNCTTX
    elif shader == "vehicle_vehglass.sps": return PBBNCTTT
    elif shader == "vehicle_paint3.sps": return PBBNCTT
    elif shader == "vehicle_dash_emissive.sps": return PBBNCT
    elif shader == "vehicle_decal2.sps": return PBBNCTX
    elif shader == "vehicle_detail2.sps": return PBBNCTT
    elif shader == "vehicle_vehglass_inner.sps": return PBBNCTT

    print('Unknown shader: ', shader)

def write_model_node(objs, materials):
    
    m_node = Element("Item")
    rm_node = Element("RenderMask")
    rm_node.set("value", str(objs[0].mask))
    flags_node = Element("Flags")
    flags_node.set("value", "0")
    has_skin_node = Element("HasSkin")
    has_skin_node.set("value", "0")
    bone_index_node = Element("BoneIndex")
    bone_index_node.set("value", "0")
    unk1_node = Element("Unknown1")
    unk1_node.set("value", "0")
    
    geo_node = Element("Geometries")
    for obj in objs:
        model = obj.data
        
        i_node = Element("Item")
        
        shader_index = 0
        shader = None
        for idx in range(len(materials)):
            if(model.materials[0] == materials[idx]):
                shader_index = idx
                shader = materials[idx] 
        shd_index = Element("ShaderIndex")
        shd_index.set("value", str(shader_index))
        
        bound_box_min = obj.bound_box[0] 
        bound_box_max = obj.bound_box[6]
        
        bbmin_node = Element("BoundingBoxMin")
        bbmin_node.set("x", str(bound_box_min[0]))
        bbmin_node.set("y", str(bound_box_min[1]))
        bbmin_node.set("z", str(bound_box_min[2]))
        bbmin_node.set("w", "0")
        bbmax_node = Element("BoundingBoxMax")
        bbmax_node.set("x", str(bound_box_max[0]))
        bbmax_node.set("y", str(bound_box_max[1]))  
        bbmax_node.set("z", str(bound_box_max[2]))
        bbmax_node.set("w", "0")
        
        vb_node = Element("VertexBuffer")
        vbflags_node = Element("Flags")
        vbflags_node.set("value", "0")
        
        vblayout_node = Element("Layout")
        vblayout_node.set("type", "GTAV1")
        
        if(shader.sollumtype != "GTA"):
            return
        print('Processing shader', shader_index, shader.name)
        vlayout = get_vertex_layout(shader.name)

        data2_node = Element("Data2")
        vertex_str = get_vertex_string(model, vlayout)
        data2_node.text = vertex_str
        
        ib_node = Element("IndexBuffer")
        data_node = Element("Data")
        data_node.text = get_index_string(model)
        
        ib_node.append(data_node)
        
        for p in vlayout:
            p_node = Element(p)
            vblayout_node.append(p_node)
        
        vb_node.append(vbflags_node)
        vb_node.append(vblayout_node)
        vb_node.append(data2_node)
        
        i_node.append(shd_index)
        i_node.append(bbmin_node)
        i_node.append(bbmax_node)
        i_node.append(vb_node)
        i_node.append(ib_node)
        
        geo_node.append(i_node)
    
    m_node.append(rm_node)
    m_node.append(flags_node)
    m_node.append(has_skin_node)
    m_node.append(bone_index_node)
    m_node.append(unk1_node)
    m_node.append(geo_node)
    
    #print(prettify(m_node))
    
    return m_node
    
def write_drawablemodels_node(models, materials):
    
    high_models = []
    med_models = []
    low_models = []
    
    for obj in models:
        if(obj.level_of_detail == "High"):
            high_models.append(obj)
        if(obj.level_of_detail == "Medium"):
            med_models.append(obj)
        if(obj.level_of_detail == "Low"):
            low_models.append(obj)
    
    drawablemodels_high_node = None
    drawablemodels_med_node = None 
    drawablemodels_low_node = None 
    
    if(len(high_models) != 0):
        drawablemodels_high_node = Element("DrawableModelsHigh")
        drawablemodels_high_node.append(write_model_node(high_models, materials))
    if(len(med_models) != 0):
        drawablemodels_med_node = Element("DrawableModelsMedium")
        drawablemodels_med_node.append(write_model_node(med_models, materials))
    if(len(low_models) != 0):
        drawablemodels_low_node = Element("DrawableModelsLow")
        drawablemodels_low_node.append(write_model_node(low_models, materials))
    
    dm_nodes = []
    if(drawablemodels_high_node != None):
        dm_nodes.append(drawablemodels_high_node)
    if(drawablemodels_med_node != None):
        dm_nodes.append(drawablemodels_med_node)
    if(drawablemodels_low_node != None):
        dm_nodes.append(drawablemodels_low_node)
    
    return dm_nodes

def write_imageparam_node(node):
    
    paramname = node.name 
    
    #if the same parameter is imported multiple time naming gets prefixed with .00X 
    #if("." in paramname): 
    #    split = paramname.split(".")
       # paramname = split[0]
        
    iname = node.name 
    type = "Texture"
    tname = "None" #givemechecker? 
    
    #if(node.image != None):
        #tname = os.path.basename(node.image.filepath)
    tname = node.texture_name[:-4] #delete file extension
    
    i_node = Element("Item")
    i_node.set("name", iname)
    i_node.set("type", type)

    if (tname != None and len(tname) > 0):
        name_node = Element("Name")
        name_node.text = tname
    
        unk32_node = Element("Unk32")
        unk32_node.set("value", "128")
    
        i_node.append(name_node)
        i_node.append(unk32_node)
    
    return i_node 

#some parameters use y,z,w
def write_vectorparam_node(nodes):
    
    i_node = Element("Item")
    
    name = nodes[0].name[:-2] #remove _x
    type = "Vector"
    
    i_node.set("name", name)
    i_node.set("type", type)
    i_node.set("x", str(nodes[0].outputs[0].default_value))
    i_node.set("y", str(nodes[1].outputs[0].default_value))
    i_node.set("z", str(nodes[2].outputs[0].default_value))
    i_node.set("w", str(nodes[3].outputs[0].default_value))
    
    return i_node

def write_shader_node(mat):
    
    i_node = Element("Item")
    
    name_node = Element("Name")
    name_node.text = fix_shader_name(mat.name, True)
    
    filename_node = Element("FileName")
    filename_node.text = fix_shader_name(mat.name) 
    
    renderbucket_node = Element("RenderBucket")
    renderbucket_node.set("value", "0")
    
    params_node = Element("Parameters")
    
    mat_nodes = mat.node_tree.nodes
    for node in mat_nodes:
        if(isinstance(node, bpy.types.ShaderNodeTexImage)):
            imgp_node = write_imageparam_node(node)
            params_node.append(imgp_node) 
        if(isinstance(node, bpy.types.ShaderNodeValue)):
            if(node.name[-1] == "x"):
                x = node
                y = mat_nodes[node.name[:-1] + "y"]
                z = mat_nodes[node.name[:-1] + "z"]
                w = mat_nodes[node.name[:-1] + "w"]
                vnode = write_vectorparam_node([x, y, z, w])
                params_node.append(vnode)
    
    i_node.append(name_node)
    i_node.append(filename_node)
    i_node.append(renderbucket_node)
    i_node.append(params_node)
    
    return i_node
    

def write_shaders_node(materials):
    
    shaders_node = Element("Shaders")

    for mat in materials:
        shader_node = write_shader_node(mat)
        shaders_node.append(shader_node)
        
    #print(prettify(shader_node))    
    
    return shaders_node

def write_tditem_node(exportpath, mat):
    
    i_nodes = []
    
    mat_nodes = mat.node_tree.nodes
    for node in mat_nodes:
        if(isinstance(node, bpy.types.ShaderNodeTexImage)):
            
            if(node.embedded == False):
                i_nodes.append(None)
            else:
                if(node.image != None):
                    
                
                    foldername = "\\" + os.path.splitext(os.path.splitext(os.path.basename(exportpath))[0])[0]
                    
                    if(os.path.isdir(os.path.dirname(exportpath) + foldername) == False):
                        os.mkdir(os.path.dirname(exportpath) + foldername)
                    
                    txtpath = node.image.filepath
                    dstpath = os.path.dirname(exportpath) + foldername + "\\" + os.path.basename(node.image.filepath)
                    
                    shutil.copyfile(txtpath, dstpath)
                else:
                    print("Missing Embedded Texture, please supply texture! The texture will not be copied to the texture folder until entered!")

                #node.image.save_render(os.path.dirname(exportpath) + "\\untitled\\"+ os.path.basename(node.image.filepath), scene=None)
                
                i_node = Element("Item")
                
                name_node = Element("Name")
                name_node.text = os.path.splitext(node.texture_name)[0]
                i_node.append(name_node)
                
                unk32_node = Element("Unk32")
                unk32_node.set("value", "128")
                i_node.append(unk32_node)
                
                usage_node = Element("Usage")
                usage_node.text = node.usage
                i_node.append(usage_node)
                
                uflags_node = Element("UsageFlags")
                uflags_text = "" 
                 
                if(node.not_half == True):
                    uflags_text += "NOT_HALF, "
                if(node.hd_split == True):
                    uflags_text += "HD_SPLIT, "
                if(node.flag_full == True):
                    uflags_text += "FULL, "
                if(node.maps_half == True):
                    uflags_text += "MAPS_HALF, "
                if(node.x2 == True):
                    uflags_text += "X2, "
                if(node.x4 == True):
                    uflags_text += "X4, "
                if(node.y4 == True):
                    uflags_text += "Y4, "
                if(node.x8 == True):
                    uflags_text += "MAPS_HALF, "
                if(node.x16 == True):
                    uflags_text += "X16, "
                if(node.x32 == True):
                    uflags_text += "X32, "
                if(node.x64 == True):
                    uflags_text += "X64, "
                if(node.y64 == True):
                    uflags_text += "Y64, "
                if(node.x128 == True):
                    uflags_text += "X128, "
                if(node.x256 == True):
                    uflags_text += "X256, "
                if(node.x512 == True):
                    uflags_text += "X512, "
                if(node.y512 == True):
                    uflags_text += "Y512, "
                if(node.x1024 == True):
                    uflags_text += "X1024, "
                if(node.y1024 == True):
                    uflags_text += "Y1024, "
                if(node.x2048 == True):
                    uflags_text += "X2048, "
                if(node.y2048 == True):
                    uflags_text += "Y2048, "
                if(node.embeddedscriptrt == True):
                    uflags_text += "EMBEDDEDSCRIPTRT, "
                if(node.unk19 == True):
                    uflags_text += "UNK19, "
                if(node.unk20 == True):
                    uflags_text += "UNK20, "
                if(node.unk21 == True):
                    uflags_text += "UNK21, "
                if(node.unk24 == True):
                    uflags_text += "UNK24, "
                
                uflags_text = uflags_text[:-2] #remove , from str
                uflags_node.text = uflags_text
                
                i_node.append(uflags_node)

                eflags_node = Element("ExtraFlags")
                eflags_node.set("value", str(node.extra_flags))
                i_node.append(eflags_node)
                
                width_node = Element("Width")
                
                size = [0, 0]
                if(node.image != None):
                    size = node.image.size
                    
                width_node.set("value", str(size[0]))
                i_node.append(width_node)
                
                height_node = Element("Height")
                height_node.set("value", str(size[1]))
                i_node.append(height_node)
                
                miplevels_node = Element("MipLevels")
                miplevels_node.set("value", "8")
                i_node.append(miplevels_node)
                
                format_node = Element("Format")
                format_node.text = "D3DFMT_" + node.format_type
                i_node.append(format_node)
                
                filename_node = Element("FileName")
                filename_node.text = node.texture_name
                i_node.append(filename_node)
                
                i_nodes.append(i_node)
                    
    return i_nodes 

def write_texturedictionary_node(materials, exportpath):
    
    td_node = Element("TextureDictionary")
    
    all_nodes = []
    
    for mat in materials:
        i_nodes = write_tditem_node(exportpath, mat)
        
        for node in i_nodes:
            if(node != None):
                all_nodes.append(node)
    
    #removes duplicate embedded textures!
    for node in all_nodes: 
        t_name = node[0].text
        
        append = True 
        
        for t in td_node:
            if(t[0].text == t_name):
                append = False
                
        if(append == True):
            td_node.append(node)        
    
    return td_node

def write_shader_group_node(materials, filepath):
    
    shaderg_node = Element("ShaderGroup")
    unk30_node = Element("Unknown30")
    unk30_node.set("value", "0")
    
    td_node = write_texturedictionary_node(materials, filepath)
    
    shader_node = write_shaders_node(materials)
    
    shaderg_node.append(unk30_node)
    shaderg_node.append(td_node)
    shaderg_node.append(shader_node)
    
    return shaderg_node

def write_skeleton_node(obj):
    skeleton_node = Element("Skeleton")
    bones_node = Element("Bones")
    skeleton_node.append(bones_node)

    bones = obj.pose.bones

    ind = 0
    for pbone in bones:
        bone = pbone.bone
        bone["BONE_INDEX"] = ind
        ind = ind + 1

    for pbone in bones:
        bone = pbone.bone

        bone_node = Element("Item")

        bone_node_name = Element("Name")
        bone_node_name.text = bone.name
        bone_node.append(bone_node_name)

        bone_node_tag = Element("Tag")

        if "BONE_TAG" in bone:
            bone_node_tag.set("value", str(bone["BONE_TAG"]))

        bone_node.append(bone_node_tag)

        bone_node_tag = Element("Index")
        bone_node_tag.set("value", str(bone["BONE_INDEX"]))
        bone_node.append(bone_node_tag)

        bone_node_parent_index = Element("ParentIndex")
        bone_node_sibling_index = Element("SiblingIndex")

        if bone.parent != None:
            bone_node_parent_index.set("value", str(bone.parent["BONE_INDEX"]))

            sibling = bone.parent.children[0]["BONE_INDEX"]
            if sibling == bone["BONE_INDEX"]:
                if len(bone.parent.children) > 1:
                    sibling = bone.parent.children[1]["BONE_INDEX"]
                else:
                    sibling = -1

            bone_node_sibling_index.set("value", str(sibling))
        else:
            bone_node_parent_index.set("value", "-1")
            bone_node_sibling_index.set("value", "-1")

        bone_node.append(bone_node_parent_index)
        bone_node.append(bone_node_sibling_index)

        bone_node_flags = Element("Flags")
        bone_node_flags.text = ""
        bone_node.append(bone_node_flags)

        trans = bone.head
        bone_node_translation = Element("Translation")
        bone_node_translation.set("x", str(trans[0]))
        bone_node_translation.set("y", str(trans[1]))
        bone_node_translation.set("z", str(trans[2]))
        bone_node.append(bone_node_translation)

        #quat = bone.rotation_euler.to_quaternion()
        quat = bone.matrix_local.to_quaternion()
        bone_node_rotation = Element("Rotation")
        bone_node_rotation.set("x", str(quat[1]))
        bone_node_rotation.set("y", str(quat[2]))
        bone_node_rotation.set("z", str(quat[3]))
        bone_node_rotation.set("w", str(quat[0]))
        bone_node.append(bone_node_rotation)

        bone_node_scale = Element("Scale")
        bone_node_scale.set("x", str(pbone.scale[0]))
        bone_node_scale.set("y", str(pbone.scale[1]))
        bone_node_scale.set("z", str(pbone.scale[2]))
        bone_node.append(bone_node_scale)

        bone_node_transform_unk = Element("TransformUnk")
        bone_node_transform_unk.set("x", "0")
        bone_node_transform_unk.set("y", "4")
        bone_node_transform_unk.set("z", "-3")
        bone_node_transform_unk.set("w", "0")
        bone_node.append(bone_node_transform_unk)

        bones_node.append(bone_node)

    return skeleton_node

def get_bbs(objs):
    bounding_boxs = []
    for obj in objs:
        bounding_boxs.append(obj.bound_box)
        
    bounding_boxmin = []
    bounding_boxmax = []

    for b in bounding_boxs:
        bounding_boxmin.append(b[0])
        bounding_boxmax.append(b[6])
    
    min_xs = []
    min_ys = []
    min_zs = []
    for v in bounding_boxmin:
        min_xs.append(v[0])
        min_ys.append(v[1])
        min_zs.append(v[2])
        
    max_xs = []
    max_ys = []
    max_zs = []
    for v in bounding_boxmax:
        max_xs.append(v[0])
        max_ys.append(v[1])
        max_zs.append(v[2])
    
    bounding_box_min = []    
    bounding_box_min.append(min(min_xs))
    bounding_box_min.append(min(min_ys))
    bounding_box_min.append(min(min_zs))
    
    bounding_box_max = []    
    bounding_box_max.append(max(max_xs))
    bounding_box_max.append(max(max_ys))
    bounding_box_max.append(max(max_zs))
    
    return [bounding_box_min, bounding_box_max]

def add_vector_list(list1, list2):
    x = list1[0] + list2[0]
    y = list1[1] + list2[1]
    z = list1[2] + list2[2]     
    return [x, y, z]

def subtract_vector_list(list1, list2):
    x = list1[0] - list2[0]
    y = list1[1] - list2[1]
    z = list1[2] - list2[2]     
    return [x, y, z]

def multiple_vector_list(list, num):
    x = list[0] * num
    y = list[1] * num
    z = list[2] * num
    return [x, y, z]

def get_vector_list_length(list):
    
    sx = list[0]**2  
    sy = list[1]**2
    sz = list[2]**2
    length = (sx + sy + sz) ** 0.5
    return length 
    
    
def get_sphere_bb(objs, bbminmax):
    
    allverts = []
    for obj in objs:
        mesh = obj.data
        for vert in mesh.vertices:
            allverts.append(vert)
    bscen = [0, 0, 0]
    bsrad = 0
    
    av = add_vector_list(bbminmax[0], bbminmax[1])
    bscen = multiple_vector_list(av, 0.5)

    for v in allverts:
        bsrad = max(bsrad, get_vector_list_length(subtract_vector_list(v.co, bscen)))

    return [bscen, bsrad]   

#still presents a problem where in a scenario you had other ydrs you didnt want to export in the scene it would pick up 
#them materials also 
def get_used_materials():
    
    materials = []
    
    all_objects = bpy.data.objects
    
    for obj in all_objects:
        if(obj.sollumtype == "Geometry"):
            mat = obj.active_material
            if(mat != None):
                materials.append(mat)          

    return materials

def write_drawable(obj, filepath):
    
    children = get_obj_children(obj)
    bbminmax = get_bbs(children)
    bbsphere = get_sphere_bb(children, bbminmax)
    
    drawable_node = Element("Drawable")
    name_node = Element("Name")
    name_node.text = obj.name
    
    bsc_node = Element("BoundingSphereCenter")
    bsc_node.set("x", str(bbsphere[0][0]))
    bsc_node.set("y", str(bbsphere[0][1]))
    bsc_node.set("z", str(bbsphere[0][2]))
    
    bsr_node = Element("BoundingSphereRadius")
    bsr_node.set("value", str(bbsphere[1]))
    
    bbmin_node = Element("BoundingBoxMin")
    bbmin_node.set("x", str(bbminmax[0][0]))
    bbmin_node.set("y", str(bbminmax[0][1]))
    bbmin_node.set("z", str(bbminmax[0][2]))
    
    bbmax_node = Element("BoundingBoxMax")
    bbmax_node.set("x", str(bbminmax[1][0]))
    bbmax_node.set("y", str(bbminmax[1][1]))
    bbmax_node.set("z", str(bbminmax[1][2]))
    
    ldh_node = Element("LodDistHigh")
    ldh_node.set("value", str(obj.drawble_distance_high))
    ldm_node = Element("LodDistMed")
    ldm_node.set("value", str(obj.drawble_distance_medium))
    ldl_node = Element("LodDistLow")
    ldl_node.set("value", str(obj.drawble_distance_low))
    ldvl_node = Element("LodDistVlow")
    ldvl_node.set("value", str(obj.drawble_distance_vlow)) 
    flagsh_node = Element("FlagsHigh")
    flagsh_node.set("value", "0")
    flagsm_node = Element("FlagsMed")
    flagsm_node.set("value", "0")
    flagsl_node = Element("FlagsLow")
    flagsl_node.set("value", "0")
    flagsvl_node = Element("FlagsVlow")
    flagsvl_node.set("value", "0")
    Unk9a_node = Element("Unknown9A")
    Unk9a_node.set("value", "0")
    
    geometrys = []
    materials = get_used_materials()
    bounds = []
    
    for c in children:
        if(c.sollumtype == "Geometry"):
            geometrys.append(c)
            
    shadergroup_node = write_shader_group_node(materials, filepath)
    skeleton_node = write_skeleton_node(obj)
    drawablemodels_node = write_drawablemodels_node(geometrys, materials)
    bounds_node = None
    
    drawable_node.append(name_node)
    drawable_node.append(bsc_node)
    drawable_node.append(bsr_node)
    drawable_node.append(bbmin_node)
    drawable_node.append(bbmax_node)
    drawable_node.append(ldh_node)
    drawable_node.append(ldm_node)
    drawable_node.append(ldl_node)
    drawable_node.append(ldvl_node)
    drawable_node.append(flagsh_node)
    drawable_node.append(flagsm_node)
    drawable_node.append(flagsl_node)
    drawable_node.append(flagsvl_node)
    drawable_node.append(Unk9a_node)
    drawable_node.append(shadergroup_node)
    drawable_node.append(skeleton_node)
    for dm_node in drawablemodels_node:
        drawable_node.append(dm_node)
    if(bounds_node != None):
        drawable_node.append(bounds_node)
    
    #print(prettify(drawable_node))
    
    return drawable_node
    

def write_ydr_xml(context, filepath):
    
    root = None

    objects = bpy.context.scene.collection.objects

    if(len(objects) == 0):
        return "No objects in scene for Sollumz export"
    
    #select the object first?
    for obj in objects:
        if(obj.sollumtype == "Drawable"):
            root = write_drawable(obj, filepath)
            try: 
                print("*** Complete ***")
            except:
                print(str(Exception))
                return str(Exception)

    if(root == None):
        return "No Sollumz Drawable found to export"
    
    xmlstr = prettify(root)
    with open(filepath, "w") as f:
        f.write(xmlstr)
        return "Sollumz Drawable was succesfully exported to " + filepath
            
class ExportYDR(Operator, ExportHelper):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "exportxml.ydr"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Export Ydr Xml (.ydr.xml)"

    # ExportHelper mixin class uses this
    filename_ext = ".ydr.xml"

    def execute(self, context):
        start = datetime.now    ()
        
        #try:
        result = write_ydr_xml(context, self.filepath)
        self.report({'INFO'}, result)
        
        #except Exception:
        #    self.report({"ERROR"}, str(Exception) )
            
        finished = datetime.now()
        difference = (finished - start).total_seconds()
        print("Exporting : " + self.filepath)
        print("Export Time:")
        print("start time: " + str(start))
        print("end time: " + str(finished))
        print("difference: " + str(difference) + " seconds")
        return {'FINISHED'}

# Only needed if you want to add into a dynamic menu
def menu_func_export(self, context):
    self.layout.operator(ExportYDR.bl_idname, text="Ydr Xml Export (.ydr.xml)")

def register():
    bpy.utils.register_class(ExportYDR)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.utils.unregister_class(ExportYDR)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
