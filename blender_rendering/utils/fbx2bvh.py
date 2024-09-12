import bpy
import numpy as np
from os import listdir, path
from tqdm import tqdm

def fbx2bvh(data_path, file):
    sourcepath = data_path+"/"+file
    new_path = "/media/mmlab/Volume2/MixamoBVH"
    bvh_path = new_path+"/"+"Penguin/" + file.split(".fbx")[0]+".bvh"

    bpy.ops.import_scene.fbx(filepath=sourcepath)

    frame_start = 9999
    frame_end = -9999
    action = bpy.data.actions[-1]
    if  action.frame_range[1] > frame_end:
      frame_end = action.frame_range[1]
    if action.frame_range[0] < frame_start:
      frame_start = int(action.frame_range[0])

    frame_end = int(np.max([60, frame_end]))
    bpy.ops.export_anim.bvh(filepath=bvh_path,
                            frame_start=frame_start,
                            frame_end=frame_end, root_transform_only=True)
    bpy.data.actions.remove(bpy.data.actions[-1])
    # print(data_path+"/"+file+" processed.")

if __name__ == '__main__':
    data_path = "/media/mmlab/Volume2/MixamoFBX/"
    directories = sorted([f for f in listdir(data_path) if not f.startswith(".")])
    for d in directories:
      if d == 'Penguin':
        files = sorted([f for f in listdir(data_path+d) if f.endswith(".fbx")])
        for file in tqdm(files):
          fbx2bvh(path.join(data_path,d), file)
    # fbx2bvh(data_path,"T-Pose.fbx")
    print('ciao')
