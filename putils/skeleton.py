import numpy as np
import torch

def concat_together(rot, root_pos):
    """
    concatenate the rotation, root_position together as the dynamic input of the
    neural network
    :param rot: rotation matrix with shape [frame, simple_joint_num - 1, 4]
    :param root_pos: with shape [frame, 1, 3], pad a 0 in dim=2, to make the position with shape
    [frame, 1, 4]
    :return: tensor with shape [frame, simple_joint_num, 4]
    """
    rot = rot[:,1:,:]
    # pad 0 make root_pos with shape [frame, 1, 4]
    rotation = rot.reshape(rot.size(0), -1)
    # into [frame, 3]
    root_pos = root_pos.reshape(root_pos.size(0), -1)
    # into [frame, (simple_joint_num - 1) * 4 + 3]
    result = torch.cat([rotation, root_pos], dim=1)
    # into [(simple_joint_num - 1) * 4 + 3, frame]
    result = result.permute(1, 0)
    return result

def topology(anim,_names):
    corps = []
    for i, name in enumerate(_names):
        if ':' in name:
            name = name[name.find(':') + 1:]
            _names[i] = name
    for name in _names:
        j = _names.index(name)
        corps.append(j)
    simplify_map = {}
    for simple_idx, complete_idx in enumerate(corps):
        simplify_map[complete_idx] = simple_idx
    # return a np.array
    return corps

def build_bone_topology(topology):
    edges = []
    joint_num = len(topology)
    for i in range(1, joint_num):
        edges.append((topology[i], i))
    return edges

def get_bvh(character_name, bvh_file_name):
    bvh_path = '/home/giuliamartinelli/Code/R2ET/datasets/mixamo/tpose/{}/{}.bvh'.format(character_name, bvh_file_name)
    # bvh_path = '/media/mmlab/Volume2/Mixamo/Test/{}/{}.bvh'.format(character_name, bvh_file_name)
    return bvh_path

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
