import sys
sys.path.append('./')
import bpy

from options import Options
from load_bvh_old import load_bvh
from scene_old import make_scene, add_material_for_character, add_rendering_parameters

if __name__ == '__main__':
    args = Options(sys.argv).parse()

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    character = load_bvh("/home/giulia.martinelli-2/Documents/Code/DATRetarget/results/gt_quat.bvh")
    scene = make_scene()
    add_material_for_character(character)
    bpy.ops.object.select_all(action='DESELECT')

    add_rendering_parameters(bpy.context.scene, args, scene[1])

    bpy.ops.render.render(animation=True, use_viewport=True)
