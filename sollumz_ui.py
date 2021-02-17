import bpy
from bpy.types import PropertyGroup, Panel, UIList, Operator
from bpy.props import CollectionProperty, PointerProperty, StringProperty, IntProperty, BoolProperty, FloatProperty

class SollumzMainPanel(Panel):
    bl_label = "Sollumz"
    bl_idname = "SOLLUMZ_PT_MAIN_PANEL"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout

        object = context.active_object
                
        if(object == None):
            layout.label(text = "No objects in scene")            
        else:
            mainbox = layout.box()
            
            textbox = mainbox.box()
            textbox.prop(object, "name", text = "Object Name")
            
            subbox = mainbox.box() 
            subbox.props_enum(object, "sollumtype")
            
            if(object.sollumtype == "Drawable"):
                box = mainbox.box()
                row = box.row()
                box.prop(object, "drawble_distance_high")
                box.prop(object, "drawble_distance_medium")
                row = box.row()
                box.prop(object, "drawble_distance_low")
                box.prop(object, "drawble_distance_vlow")
            if(object.sollumtype == "Geometry"):
                box = mainbox.box()
                box.prop(object, "level_of_detail")
                box.prop(object, "mask")   
        
        box = layout.box()
        box.label(text = "Tools") 
        
def param_name_to_title(pname):
    
    title = ""
    
    a = pname.split("_")
    b = a[0]
    glue = ' '
    c = ''.join(glue + x if x.isupper() else x for x in b).strip(glue).split(glue)
    d = ""
    for word in c:
        d += word
        d += " "
    title = d.title() #+ a[1].upper() dont add back the X, Y, Z, W
    
    return title

class SollumzMaterialPanel(Panel):
    bl_label = "Sollumz Material Panel"
    bl_idname = "Sollumz_PT_MAT_PANEL"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"
    
    shadername : bpy.props.StringProperty(default = "default.sps")
    
    def draw(self, context):
        layout = self.layout

        object = context.active_object        
        if(object == None):
            return
        mat = object.active_material  
        
        tbox = layout.box()
        tbox.label(text = "Tools")
        box = tbox.box()
        box.label(text = "Create Shader")
        row = box.row()  
        row.label(text = "Shader Type:")
        row.prop_menu_enum(object, "shadertype", text = object.shadertype)
        box.operator("sollum.createvshader").shadername = object.shadertype
        
        if(mat == None):
            return
        
        if(mat.sollumtype == "Blender"):
            box = tbox.box()
            row = box.row()
            row.label(text = "Convert To Shader")
            row.operator("sollum.converttov") 
        
        
        if(mat.sollumtype == "GTA"):
            
            box = layout.box()
            box.prop(mat, "name", text = "Shader")
            
            #layout.label(text = "Parameters")
            
            #box = layout.box()
            #box.label(text = "Parameters")
            
            mat_nodes = mat.node_tree.nodes
            
            image_nodes = []
            value_nodes = []
            
            for n in mat_nodes:
                if(isinstance(n, bpy.types.ShaderNodeTexImage)):
                    image_nodes.append(n)
                elif(isinstance(n, bpy.types.ShaderNodeValue)):
                    value_nodes.append(n)
                #else:
            
            for n in image_nodes:
                box = box.box()
                box.label(text = n.name + " Texture")
                
                row = box.row()
                if(n.image != None):
                    row.prop(n.image, "filepath")
                row.prop(n, "embedded")
                
                row = box.row()
                #row.prop(specnode, "type") #gims fault
                row.prop(n, "format_type")
                
                #row = box.row() #gims fault
                row.prop(n, "usage")
                
                uf_box = box.box()
                uf_box.label(text = "Usage Flags:")
                uf_row = uf_box.row()
                uf_row.prop(n, "not_half")
                uf_row.prop(n, "hd_split")
                uf_row.prop(n, "flag_full")
                uf_row.prop(n, "maps_half")
                uf_row = uf_box.row()
                uf_row.prop(n, "x2")
                uf_row.prop(n, "x4")
                uf_row.prop(n, "y4")
                uf_row.prop(n, "x8")
                uf_row = uf_box.row()
                uf_row.prop(n, "x16")
                uf_row.prop(n, "x32")
                uf_row.prop(n, "x64")
                uf_row.prop(n, "y64")
                uf_row = uf_box.row()
                uf_row.prop(n, "x128")
                uf_row.prop(n, "x256")
                uf_row.prop(n, "x512")
                uf_row.prop(n, "y512")
                uf_row = uf_box.row()
                uf_row.prop(n, "x1024")
                uf_row.prop(n, "y1024")
                uf_row.prop(n, "x2048")
                uf_row.prop(n, "y2048")
                uf_row = uf_box.row()
                uf_row.prop(n, "embeddedscriptrt")
                uf_row.prop(n, "unk19")
                uf_row.prop(n, "unk20")
                uf_row.prop(n, "unk21")
                uf_row = uf_box.row()
                uf_row.prop(n, "unk24")
                
                uf_box.prop(n, "extra_flags")
                
            prevname = ""
            #value_nodes.insert(1, value_nodes.pop(len(value_nodes) - 1)) #shift last item to second because params are messed up for some reason ? (fixed?)
            for n in value_nodes:
                if(n.name[:-2] not in prevname):
                    #new parameter
                    parambox = box.box()
                    parambox.label(text = param_name_to_title(n.name)) 
                      
                parambox.prop(n.outputs[0], "default_value", text = n.name[-1].upper())
                
                prevname = n.name 
            
        
class SOLLUMZ_UL_BoneFlags(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index): 
        custom_icon = 'FILE'

        if self.layout_type in {'DEFAULT', 'COMPACT'}: 
            layout.prop(item, 'name', text='', icon = custom_icon, emboss=False, translate=False)
        elif self.layout_type in {'GRID'}: 
            layout.alignment = 'CENTER' 
            layout.prop(item, 'name', text='', icon = custom_icon, emboss=False, translate=False)

class SOLLUMZ_OT_BoneFlags_NewItem(Operator): 
    bl_idname = "sollumz_flags.new_item" 
    bl_label = "Add a new item"
    def execute(self, context): 
        context.active_pose_bone.bone_properties.flags.add() 
        return {'FINISHED'}

class SOLLUMZ_OT_BoneFlags_DeleteItem(Operator): 
    bl_idname = "sollumz_flags.delete_item" 
    bl_label = "Deletes an item" 
    @classmethod 
    def poll(cls, context): 
        return context.active_pose_bone.bone_properties.flags

    def execute(self, context): 
        list = context.active_pose_bone.bone_properties.flags
        index = context.active_pose_bone.bone_properties.ul_flags_index
        list.remove(index) 
        context.active_pose_bone.bone_properties.ul_flags_index = min(max(0, index - 1), len(list) - 1) 
        return {'FINISHED'}

class SollumzBonePanel(Panel):
    bl_label = "Sollumz Bone Panel"
    bl_idname = "SOLLUMZ_PT_BONE_PANEL"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "bone"

    def draw(self, context):
        layout = self.layout

        bone = context.active_pose_bone
                
        if(bone == None):
            return

        layout.prop(bone, "name", text = "Bone Name")
        layout.prop(bone.bone_properties, "id", text = "BoneID")

        layout.label(text="Flags")
        layout.template_list("SOLLUMZ_UL_BoneFlags", "Flags", bone.bone_properties, "flags", bone.bone_properties, "ul_flags_index")
        row = layout.row() 
        row.operator('sollumz_flags.new_item', text='New')
        row.operator('sollumz_flags.delete_item', text='Delete')

classes = (
    SollumzMaterialPanel,
    SollumzMainPanel,
    SollumzBonePanel,
    SOLLUMZ_UL_BoneFlags,
    SOLLUMZ_OT_BoneFlags_NewItem,
    SOLLUMZ_OT_BoneFlags_DeleteItem,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)