import numpy as np
import torch
from einops import rearrange

def concat_together(rot, root_pos):
    """
    concatenate the rotation, root_position together as the dynamic input of the
    neural network
    :param rot: rotation matrix with shape [frame, simple_joint_num - 1, 4]
    :param root_pos: with shape [frame, 1, 3], pad a 0 in dim=2, to make the position with shape
    [frame, 1, 4]
    :return: tensor with shape [frame, simple_joint_num, 4]
    """
    frame_num = root_pos.size(0)
    # pad 0 make root_pos with shape [frame, 1, 4]
    if rot.shape[-1] == 4:
        pad_root_pos = torch.zeros([frame_num, 1, 4], dtype=torch.float)
    else:
        pad_root_pos = torch.zeros([frame_num, 1, 6], dtype=torch.float)

    pad_root_pos[:, :, 0:3] = root_pos
    # concatenate all together
    if pad_root_pos.device == rot.device:
        result = torch.cat([pad_root_pos, rot], dim=1)
    else:
        pad_root_pos = pad_root_pos.bfloat16().cuda()
        result = torch.cat([pad_root_pos, rot], dim=1)
    return result

def get_bvh(character_name, bvh_file_name):
    bvh_path = '/home/giuliamartinelli/Code/R2ET/datasets/mixamo/tpose/{}/{}.bvh'.format(character_name, bvh_file_name)
    # bvh_path = '/media/mmlab/Volume2/Mixamo/Test/{}/{}.bvh'.format(character_name, bvh_file_name)
    return bvh_path

def build_bone_topology(topology):
    edges = []
    joint_num = len(topology)
    for i in range(1, joint_num):
        edges.append((topology[i], i))
    return edges

def build_edge_topology(topology, offset):
    # get all edges (pa, child, offset)
    edges = []
    joint_num = len(topology)
    for i in range(1, joint_num):
        edges.append((topology[i], i, offset[i]))
    return edges

def build_joint_topology(edges, origin_names):
    parent = []
    offset = []
    names = []
    edge2joint = []
    joint_from_edge = []  # -1 means virtual joint
    joint_cnt = 0
    out_degree = [0] * (len(edges) + 10)
    for edge in edges:
        out_degree[edge[0]] += 1

    # add root joint
    joint_from_edge.append(-1)
    parent.append(0)
    offset.append(np.array([0, 0, 0]))
    names.append(origin_names[0])
    joint_cnt += 1

    def make_topology(edge_idx, pa):
        nonlocal edges, parent, offset, names, edge2joint, joint_from_edge, joint_cnt
        edge = edges[edge_idx]
        if out_degree[edge[0]] > 1:
            parent.append(pa)
            offset.append(np.array([0, 0, 0]))
            names.append(origin_names[edge[1]] + '_virtual')
            edge2joint.append(-1)
            pa = joint_cnt
            joint_cnt += 1

        parent.append(pa)
        offset.append(edge[2])
        names.append(origin_names[edge[1]])
        edge2joint.append(edge_idx)
        pa = joint_cnt
        joint_cnt += 1

        for idx, e in enumerate(edges):
            if e[0] == edge[1]:
                make_topology(idx, pa)

    for idx, e in enumerate(edges):
        if e[0] == 0:
            make_topology(idx, 0)

    return parent, offset, names, edge2joint


def _concat_together(rot, root_pos):
    """
    concatenate the rotation, root_position together as the dynamic input of the
    neural network
    :param rot: rotation matrix with shape [frame, simple_joint_num - 1, 4]
    :param root_pos: with shape [frame, 1, 3], pad a 0 in dim=2, to make the position with shape
    [frame, 1, 4]
    :return: tensor with shape [frame, simple_joint_num, 4]
    """
    frame_num = root_pos.size(0)
    # pad 0 make root_pos with shape [frame, 1, 4]
    if rot.shape[-1] == 4:
        pad_root_pos = torch.zeros([frame_num, 1, 4], dtype=torch.float)
    else:
        pad_root_pos = torch.zeros([frame_num, 1, 6], dtype=torch.float)

    pad_root_pos[:, :, 0:3] = root_pos
    # concatenate all together
    if pad_root_pos.device == rot.device:
        result = torch.cat([pad_root_pos, rot], dim=1)
    else:
        pad_root_pos = pad_root_pos.bfloat16().cuda()
        result = torch.cat([pad_root_pos, rot], dim=1)
    return result

def de_normalize(raw: torch.tensor):
    device = raw.device
    rot_part = raw[:, :, 1:, :]
    pos_part = raw[:, 0:-1, 0:1, :]
    # denormalize
    # rot_part = (rot_part * rot_var) + rot_mean
    # pos_part = (pos_part * pos_var) + pos_mean
    return rot_part, pos_part

def convert_to_bvh_write_format(raw: torch.tensor): #, character_idx
    """
    :param raw: in shape [B, 4, J, frame], since this function only called during inference stage,
    the B is always equal to 1
    :param character_idx: an int number
    :return: tensor with shape
    """
    # denormalize first
    # [1, 4, J-1, frame], [1, 3, 1, frame]
    denorm_rot, denorm_root_pos = de_normalize(raw)
    denorm_rot = rearrange(denorm_rot, 'b q j w -> q j (b w)')
    denorm_rot = denorm_rot[np.newaxis, ...]
    denorm_root_pos = rearrange(denorm_root_pos, 'b q j w -> q j (b w)')
    denorm_root_pos = denorm_root_pos[np.newaxis, ...]
    # make rotation from [1, 4, J-1, frame] to [frame, J-1, 4]
    rotation = denorm_rot.squeeze(0).permute(2, 1, 0)
    # make root position from [1, 3, 1, frame] to [frame, 1, 3]
    root_pos = denorm_root_pos.squeeze(0).permute(2, 1, 0)
    # into [frame, (simple_joint_num - 1) * 4]
    rotation = rotation.reshape(rotation.size(0), -1)
    # into [frame, 3]
    root_pos = root_pos.reshape(root_pos.size(0), -1)
    # into [frame, (simple_joint_num - 1) * 4 + 3]
    result = torch.cat([rotation, root_pos], dim=1)
    # into [(simple_joint_num - 1) * 4 + 3, frame]
    result = result.permute(1, 0)
    return result
