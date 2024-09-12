
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
import numpy as np
from datasets.shape_feeder import Feeder
from src.forward_kinematics import FK
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss
from mesh_intersection.filter_faces import FilterFaces
from mesh import create    
import pyrender
import trimesh
from putils.BVH_FILE import read_bvh
import wandb
from putils.BVH import load,save
from Stopping import EarlyStopping
from prior import create_prior
from putils.Quaternions import Quaternions
from tqdm import tqdm
from prior import create_prior
from einops import rearrange
import json
from putils.Filtering import gaussian_smooth
from putils.IK import get_character_height
from evaluation_utils import de_normalize, concat_together, slice_to_equal_frame_len, to_format_tensor
import putils.Animation_deep as Animation
from putils.bvh_writer import BvhWriter
from putils.Filtering import gaussian_smooth

def build_bone_topology(topology):
    edges = []
    joint_num = len(topology)
    for i in range(1, joint_num):
        edges.append((topology[i], i))
    return edges

def topology(anim,simplify_map):
    _topology = anim.parents.copy()
    for i in range(_topology.shape[0]):
        if i >= 1:
            _topology[i] = simplify_map[_topology[i]]
    # return a np.array
    return _topology

def compute_collisions(vertices,faces,faces_idx,search_tree,collision_idxs_tpose,filter_faces,body_lst,
                       leftarm_lst,rightarm_lst,leftleg_lst,rightleg_lst,head_lst,hands_lst,
                       righthands_lst,lefthands_lst,interactive,query_names,batch_size,viewer = None,scene = None):
    
    verts = vertices
    triangles = verts.view([-1, 3])[faces_idx]*0.01

    with torch.no_grad():
        outputs = search_tree(triangles)[0]
        collision_idxs = outputs[outputs[:, 0] >= 0, :].unsqueeze(0)
        outputs_new = torch.zeros(size=(abs(collision_idxs.shape[1]-collision_idxs_tpose.shape[1]),collision_idxs.shape[2])).cuda()
        collision_idxs_tpose_reshaped = collision_idxs_tpose.expand(collision_idxs.shape[1], -1, -1)  # [1673, 2042, 2]
        collision_idxs_reshaped = collision_idxs.expand(collision_idxs_tpose.shape[1], -1, -1)  # [2042, 1673, 2]

        # Check for common elements along axis 1
        mask = torch.any(collision_idxs_tpose_reshaped == collision_idxs_reshaped.transpose(0, 1), dim=1)  # [1673, 2042]


        # Find indices of common elements
        row_indices, _ = torch.nonzero(~mask, as_tuple=True)
        row_indices = torch.unique(row_indices)

        collision_idxs = collision_idxs[0][row_indices].unsqueeze(0)

        collision_idxs = filter_faces(collision_idxs)

        # print("Collision to solve: ",collision_idxs.shape[1])
        
        index_toremove = []
        body_set = set(body_lst)
        leftarm_set = set(leftarm_lst)
        rightarm_set = set(rightarm_lst)
        leftleg_set = set(leftleg_lst)
        rightleg_set = set(rightleg_lst)
        head_set = set(head_lst)
        hands_set = set(hands_lst)
        righthands_set = set(righthands_lst)
        lefthands_set = set(lefthands_lst)

    index_toremove = []

    for index, element in enumerate(collision_idxs[0]):
        part1, part2 = int(element[0]), int(element[1])
        
        # Check for valid combinations and add indices to index_toremove
        condition_results = [
        (part1 in body_set and part2 in leftarm_set),(part1 in leftarm_set and part2 in body_set),
        (part1 in body_set and part2 in rightarm_set),(part1 in rightarm_set and part2 in body_set),
        (part1 in body_set and part2 in leftleg_set),(part1 in leftleg_set and part2 in body_set),
        (part1 in body_set and part2 in rightleg_set),(part1 in rightleg_set and part2 in body_set),
        (part1 in body_set and part2 in head_set),(part1 in head_set and part2 in body_set),
        (part1 in body_set and part2 in hands_set),(part1 in hands_set and part2 in body_set),
        (part1 in leftarm_set and part2 in rightarm_set),(part1 in rightarm_set and part2 in leftarm_set),
        (part1 in leftarm_set and part2 in leftleg_set),(part1 in leftleg_set and part2 in leftarm_set),
        (part1 in leftarm_set and part2 in rightleg_set),(part1 in rightleg_set and part2 in leftarm_set),
        (part1 in leftarm_set and part2 in head_set),(part1 in head_set and part2 in leftarm_set),
        (part1 in leftarm_set and part2 in righthands_set),(part1 in righthands_set and part2 in leftarm_set),
        (part1 in rightarm_set and part2 in leftleg_set),(part1 in leftleg_set and part2 in rightarm_set),
        (part1 in rightarm_set and part2 in rightleg_set),(part1 in rightleg_set and part2 in rightarm_set),
        (part1 in rightarm_set and part2 in head_set),(part1 in head_set and part2 in rightarm_set),
        (part1 in rightarm_set and part2 in lefthands_set),(part1 in lefthands_set and part2 in rightarm_set),
        (part1 in leftleg_set and part2 in rightleg_set),(part1 in rightleg_set and part2 in leftleg_set),
        (part1 in leftleg_set and part2 in head_set),(part1 in head_set and part2 in leftleg_set),
        (part1 in leftleg_set and part2 in hands_set),(part1 in hands_set and part2 in leftleg_set),
        (part1 in head_set and part2 in hands_set),(part1 in hands_set and part2 in head_set)]

        if any(condition_results):
            index_toremove.append(index)

    indices_to_keep = torch.ones(collision_idxs[0].shape, dtype=torch.bool)
    indices_to_keep[index_toremove] = 0
    row_indices, _ = torch.nonzero(~indices_to_keep, as_tuple=True)
    collision_idxs = collision_idxs[0][row_indices].unsqueeze(0)

    if interactive:
        with torch.no_grad():
            verts = vertices

        np_verts = verts.detach().cpu().numpy()

        np_collision_idxs = collision_idxs.detach().cpu().numpy()
        np_receivers = np_collision_idxs[:, :, 0]
        np_intruders = np_collision_idxs[:, :, 1]

        viewer.render_lock.acquire()

        for node in scene.get_nodes():
            if node.name is None:
                continue
            if any([query in node.name for query in query_names]):
                scene.remove_node(node)

        for bidx in range(batch_size):
            recv_faces_idxs = np_receivers[bidx][np_receivers[bidx] >= 0]
            intr_faces_idxs = np_intruders[bidx][np_intruders[bidx] >= 0]
            recv_faces = faces[recv_faces_idxs]
            intr_faces = faces[intr_faces_idxs]

            curr_verts = np_verts[bidx].copy()
            body_mesh = create_mesh(curr_verts, faces,
                                    color=(0.3, 0.3, 0.3, 0.99),
                                    wireframe=True)

            pose = np.eye(4)
            pose[0, 3] = bidx * 2
            scene.add(body_mesh,
                    name='body_mesh_{:03d}'.format(bidx),
                    pose=pose)

            if len(intr_faces) > 0:
                intr_mesh = create_mesh(curr_verts, intr_faces,
                                        color=(0.9, 0.0, 0.0, 1.0))
                scene.add(intr_mesh,
                        name='intr_mesh_{:03d}'.format(bidx),
                        pose=pose)

            if len(recv_faces) > 0:
                recv_mesh = create_mesh(curr_verts, recv_faces,
                                        color=(0.0, 0.9, 0.0, 1.0))
                scene.add(recv_mesh, name='recv_mesh_{:03d}'.format(bidx),
                        pose=pose)
                
        viewer.render_lock.release()

        
        if not viewer.is_active:
            return collision_idxs
        
        time.sleep(100 / 1000)

    return collision_idxs, triangles

def compute_loss(output,collision_idxs,pen_distance,triangles,pose_reg_weight,mse_loss,body,init_pose,coll_loss_weight,device):
    
    
    pen_loss = torch.tensor(0, device=device,
                                        dtype=torch.float32)

    if collision_idxs.ge(0).sum().item() > 0:
        pen_loss = pen_distance(triangles, collision_idxs) /body.faces.shape[0]

    pose_reg_loss = torch.tensor(0, device=device,
                                dtype=torch.float32)

    if pose_reg_weight > 0:
        pose_reg_loss = mse_loss(body.body_pose, init_pose)

                
    loss = coll_loss_weight * pen_loss + pose_reg_weight * pose_reg_loss #+ angle_prior_loss * angle_weight

    lpose_to_append = output.body_pose.clone().detach()

    # np_loss = loss.detach().cpu().squeeze().tolist()
    # if type(np_loss) != list:
    #     np_loss = [np_loss]
    # msg = '{:.5f} ' * len(np_loss)
    # print('Loss per model:', loss)
    # print('Pen Loss:', pen_loss)
    # print('Pose Loss:', pose_reg_loss)
    # # print('Angle Pose:', angle_prior_loss)
    # print('Number of collisions = ', collision_idxs.shape[1])

    return loss, lpose_to_append

def create_mesh(vertices, faces, color=(0.3, 0.3, 0.3, 1.0),
                        wireframe=True):

    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    rot = trimesh.transformations.rotation_matrix(np.radians(90),
                                                    [1, 0, 0])
    tri_mesh.apply_transform(rot)

    rot = trimesh.transformations.rotation_matrix(np.radians(90),
                                                    [0, 0, 1])
    
    tri_mesh.apply_transform(rot)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='BLEND',
        baseColorFactor=color)
    return pyrender.Mesh.from_trimesh(
        tri_mesh,
        material=material)

def main():

    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--interactive', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Display the mesh during the optimization' +
                        ' process')
    parser.add_argument('--delay', type=int, default=100,
                        help='The delay for the animation callback in ms')
    parser.add_argument('--point2plane', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use point to distance')
    parser.add_argument('--optimize_pose', default=True,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Enable optimization over the joint pose')
    parser.add_argument('--optimize_shape', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Enable optimization over the shape of the model')
    parser.add_argument('--sigma', default= 0.001, type=float,
                        help='The height of the cone used to calculate the' +
                        ' distance field loss')
    parser.add_argument('--lr', default= 0.1, type=float,
                        help='The learning rate for SGD')
    parser.add_argument('--coll_loss_weight', default=0.9, type=float,
                        help='The weight for the collision loss')
    parser.add_argument('--pose_reg_weight', default=0.1, type=float,
                        help='The weight for the pose regularizer')
    parser.add_argument('--angle_weight', default=0, type=float,
                        help='The weight for the pose regularizer')
    parser.add_argument('--max_collisions', default=1000, type=int,
                        help='The maximum number of bounding box collisions')
    parser.add_argument('--print_timings', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Print timings for all the operations')

    args, _ = parser.parse_known_args()

    interactive = args.interactive
    point2plane = args.point2plane
    lr = args.lr
    coll_loss_weight = args.coll_loss_weight
    pose_reg_weight = args.pose_reg_weight
    angle_weight = args.angle_weight
    max_collisions = args.max_collisions
    sigma = args.sigma

    batch_size = 1
    device = torch.device('cuda')
    mesh_path = "./datasets/mixamo/training_shape"
    character_name = "BigVegas"
    bvh_path = "./MixamoBVH"
    if '_m' in character_name:
        char_name = character_name.split('_m')[0]
    else:
        char_name = character_name

    t_pose_path = "./MoMaAnimation/Tpose/%s.npy" %(char_name)
    file_names = os.listdir(mesh_path)
    json_file_path = './datasets/filelist/Homeomorphic/mixamo.json'

    ## Load Tpose
    t_pose = torch.tensor(np.load(t_pose_path)).cuda().float()
    
    ## Load Mesh
    mesh_name = char_name + '.npz'
    fbx_data =  np.load(os.path.join(mesh_path, mesh_name),allow_pickle=True)
    np_rest_faces = fbx_data['rest_faces']
    num_face = np_rest_faces.shape[0]
    vertex_part_np = fbx_data['vertex_part']

    ## Start Segment of Collisions Filtering

    leftarm_bone_lst = np.array([15, 16, 17]).tolist()
    rightarm_bone_lst = np.array([19, 20, 21]).tolist()
    leftleg_bone_lst = np.array([6, 7, 8, 9]).tolist()
    rightleg_bone_lst = np.array([10, 11, 12, 13]).tolist()
    body_bone_lst = np.array([0, 1, 2, 3]).tolist()
    head_bone_lst = np.array([5]).tolist()
    
    face_part = []
    for i in range(num_face):
        face_part.append(vertex_part_np[np_rest_faces[i][0]])
    face_part = np.array(face_part)
    body_lst = []
    head_lst = []
    leftarm_lst = []
    rightarm_lst = []
    leftleg_lst = []
    rightleg_lst = []
    hands_lst = []
    righthands_lst = []
    lefthands_lst = []
    for i in range(num_face):
        if face_part[i] in body_bone_lst:
            body_lst.append(i)
        if face_part[i] in head_bone_lst:
            head_lst.append(i)
        if face_part[i] in leftarm_bone_lst:
            leftarm_lst.append(i)
        if face_part[i] in rightarm_bone_lst:
            rightarm_lst.append(i)
        if face_part[i] in leftarm_bone_lst[-1:]:
            lefthands_lst.append(i)
        if face_part[i] in rightarm_bone_lst[-1:]:
            righthands_lst.append(i)
        if face_part[i] in leftleg_bone_lst:
            leftleg_lst.append(i)
        if face_part[i] in rightleg_bone_lst:
            rightleg_lst.append(i)
        if (
            face_part[i] in leftarm_bone_lst[-1:]
            or face_part[i] in rightarm_bone_lst[-1:]
        ):
            hands_lst.append(i)


    parents_mapping = [(0,0),(1,0),(2,1),(3,2),(4,3),(5,4),(6,0),(7,6),(8,7),(9,8),(10,0),(11,10),(12,11),(13,12),(14,4),(15,14),
                       (16,15),(17,16),(18,4),(19,18),(20,19),(21,20)]
    

    ign_part_pairs=["1,14","1,15","3,14", "3,15", "2,14", "2,15","1,18","1,19","3,18", "3,19", "2,18", "2,19", "6,10","0,6","0,10"]
    
    filter_faces = FilterFaces(faces_segm = face_part, faces_parents=face_part,ign_part_pairs=ign_part_pairs).to(device=device)


    ## Define Bounding Volume Hierarchies and losse for collisions detection

    search_tree = BVH(max_collisions=max_collisions)

    pen_distance = \
        collisions_loss.DistanceFieldPenetrationLoss(sigma=sigma,
                                                     point2plane=point2plane,
                                                     penalize_outside=True,
                                                     vectorized=True)

    mse_loss = nn.MSELoss(reduction='sum').to(device=device)
   
    body = create(fbx_data,t_pose).to(device=device)

    with torch.no_grad():
        output = body()
        verts = output.vertices

    face_tensor = torch.tensor(output.faces, dtype=torch.long,
                        device=device).unsqueeze_(0).repeat([batch_size,
                                                            1, 1])


    bs, nv = verts.shape[:2]
    bs, nf = face_tensor.shape[:2]
    faces_idx = face_tensor + \
        (torch.arange(bs, dtype=torch.long).to(device) * nv)[:, None, None]


    with torch.no_grad():
        output = body()
        verts = output.vertices
        triangles = verts.view([-1, 3])[faces_idx]*0.01
        outputs = search_tree(triangles)[0]
        collision_idxs_tpose = outputs[outputs[:, 0] >= 0, :].unsqueeze(0)


    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    bvh_files = data[character_name]

    if '_m' in character_name:
        mapping_back = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,16,17,18,19,21,22,23,24]
    else:
        mapping_back = list(range(22))
   
    mapping = [0,6,7,8,9,10,11,12,13,1,2,3,4,5,14,15,16,17,18,19,20,21]
    
    print("Processing: ",character_name)

    for motion_index, animation in tqdm(enumerate(bvh_files)):

        start = time.time()

        file_name = animation.split('/')[-1]

        anim_ref, total_names, _ = read_bvh(os.path.join(bvh_path,character_name,file_name))

        tg_anim = anim_ref.copy()

        anim, simplified_name, frame_time = read_bvh(animation)


        total_names = simplified_name

        for i, name in enumerate(total_names):
            if ':' in name:
                name = name[name.find(':') + 1:]
                total_names[i] = name

        corps = []
        for name in simplified_name:
            if name == 'Head' or 'split' in name:
                continue
            j = total_names.index(name)
            corps.append(j)


        rotations = np.array(Quaternions.from_euler(np.radians(anim.rotations)).qs)[:,1:,:]


        positions = np.expand_dims(anim.positions[:,0,:], axis = 1)
        concat_motion = concat_together(rotations,positions)
        concat_motion = to_format_tensor(concat_motion)
        concat_motion = rearrange(slice_to_equal_frame_len(concat_motion), 'b q j w -> (b w) j q')
        rotations_ref , positions_ref = de_normalize(concat_motion)

        if anim.rotations.shape[0] != concat_motion.shape[0]:
            continue
        
        # anim_ref.rotations = Quaternions(rotations_ref)       

        optimizer = torch.optim.LBFGS([body.body_pose], lr=lr,max_iter=30)
        random_seed = 42 
        torch.manual_seed(random_seed)

        if anim_ref.rotations.shape[0] < 120:
            max_length = anim_ref.rotations.shape[0]
        else:
            max_length = 120

        max_length = anim.rotations.shape[0]

        prev_pose = body.prev_pose(anim,0,character_name)

        correct_animation = torch.zeros(size=(260,22,4)).cuda()

        max_length = 260


        for i in tqdm(range(60)):    #range(anim.rotations.shape[0]
    
            if interactive:
            # Plot the initial mesh
                with torch.no_grad():
                    output = body()
                    verts = output.vertices

                np_verts = verts.detach().cpu().numpy()

                scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0],
                                    ambient_light=(1.0, 1.0, 1.0))
                for bidx in range(np_verts.shape[0]):
                    curr_verts = np_verts[bidx].copy()
                    body_mesh = create_mesh(curr_verts, body.faces,
                                            color=(0.3, 0.3, 0.3, 0.99),
                                            wireframe=True)

                    pose = np.eye(4)
                    pose[0, 3] = bidx * 2
                    scene.add(body_mesh,
                            name='body_mesh_{:03d}'.format(bidx),
                            pose=pose)
        
                viewer = pyrender.Viewer(scene, use_raymond_lighting=True,
                                        viewport_size=(1200, 800),
                                        cull_faces=False,
                                        run_in_thread=True, record = True)

                query_names = ['recv_mesh', 'intr_mesh', 'body_mesh']
            else:
                viewer = None
                scene = None
                query_names = None

            body.set_pose(prev_pose)
            
            init_pose = body.prev_pose(anim,i,character_name).clone().detach()

            losses = []
            poses = []
            collisions = []
            ver = []
            def closure():
                output = body()
                vertices = output.vertices
                faces = output.faces
                collision_idxs, triangles = compute_collisions(vertices,faces,faces_idx,search_tree,collision_idxs_tpose,filter_faces,body_lst,
                        leftarm_lst,rightarm_lst,leftleg_lst,rightleg_lst,head_lst,hands_lst,
                        righthands_lst,lefthands_lst,interactive,query_names,batch_size,viewer,scene)
                loss, pose = compute_loss(output,collision_idxs,pen_distance,triangles,pose_reg_weight,mse_loss,body,init_pose,coll_loss_weight,device)
                losses.append(loss.item())
                poses.append(pose)
                optimizer.zero_grad()
                loss.backward(create_graph = False)
                return loss

            optimizer.step(closure=closure)
            best_index = losses.index(min(losses))
            prev_pose = poses[best_index].detach()
            correct_animation[i,:,:] = prev_pose

            if interactive:
                    viewer.close_external()


        correct_animation = correct_animation.detach().cpu().numpy()

        mapping_back = list(range(22))

        correct_animation_new = np.zeros(shape=correct_animation.shape)
        correct_animation_new[:,mapping_back,:] = correct_animation[:,mapping,:]

        position = anim.positions[:, 0, :]
        offsets = anim.offsets
        rotations = correct_animation_new
        rotations, position = gaussian_smooth(rot=rotations, pos=position)

        simplify_map = {}
        inverse_simplify_map = {}
        for simple_idx, complete_idx in enumerate(range(anim.offsets.shape[0])): #anim_ref
            simplify_map[complete_idx] = simple_idx
            inverse_simplify_map[simple_idx] = complete_idx
        # TODO why set -1 here ???
        inverse_simplify_map[0] = -1
        _topology = topology(tg_anim,simplify_map)
        edges = build_bone_topology(_topology)

        


        corps_pred = [0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21]
        

        rotations_final = Quaternions.from_euler(np.radians(tg_anim.rotations)).qs
        rotations_final[:max_length,corps,:] = rotations

        index = []
        for e in edges:
            index.append(e[0])

        finish = time.time() - start

        # print("Time",finish)

        # rotations_final = rotations_final[:max_length, index, :]

        offsets = anim.offsets

        writer = BvhWriter(edges,total_names,offsets)

        save_path = ""

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # writer.write(rotations_final, position, 'quaternion',save_path + "/%s_%s.bvh" %(file_name.split('.bvh')[0],character_name))
        writer.write(rotations_final, position, 'quaternion',save_path + "/%s.bvh" %(file_name.split('.bvh')[0]))


if __name__ == "__main__":

    wandb.init(project="optimization_collsion", name="GanGnam Style Mousey")
    main()

