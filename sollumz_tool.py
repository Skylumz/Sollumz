import bpy
from typing import NamedTuple, Optional
from .ytyp.tools import ArchetypeExtensionTool
from .ydr.gizmos.light_manipulators.culling_plane import LightCullingPlaneTool

class SollumzToolDef(NamedTuple):
    cls: type
    separator: bool = False
    group: bool = False
    after: Optional[str] = None

tools = (
    SollumzToolDef(ArchetypeExtensionTool, separator=True, group=True),
    SollumzToolDef(LightCullingPlaneTool, after=ArchetypeExtensionTool.bl_idname),
)

def register_tools():
    for tool in tools:
        bpy.utils.register_tool(tool.cls, separator=tool.separator, group=tool.group, after=tool.after)

def unregister_tools():
    for tool in tools:
        bpy.utils.unregister_tool(tool.cls)
