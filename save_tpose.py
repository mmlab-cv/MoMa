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
from utils.BVH import save
from Stopping import EarlyStopping
from prior import create_prior
from putils.Quaternions import Quaternions


def main():
    data_path = "./datasets/mixamo/training_numpy"
    stats_path = "./datasets/mixamo/stats"
    bvh_path = "./MixamoBVH"
    character_name = "Yaku"
    if '_m' in character_name:
        char_name = character_name.split('_m')[0]
    else:
        char_name = character_name
    shape_path = "./datasets/mixamo/training_shape/" + char_name + '.npz'
    max_length = 60
    data_feeder = Feeder(data_path,stats_path,shape_path,char_name,max_length)

    data_loader = torch.utils.data.DataLoader(
            dataset=data_feeder, batch_size=1, num_workers=8, shuffle=True
        )

    for batch_idx, (
            index,
            seq,
            skel,
            aeReg,
            mask,
            height,
            shape,
            quat,
            # bvh,
        ) in enumerate(data_loader):
            seq = seq.float().cuda()
            skel = skel.float().cuda()
            aeReg = aeReg.float().cuda()
            mask = mask.float().cuda()
            height = height.float().cuda()
            quat = quat.float().cuda()
            shape = shape.float().cuda()


    local_std = data_feeder.local_std
    local_mean = data_feeder.local_mean


    t_poseB = torch.reshape(skel[:, 0, :], [1, 22, 3])
    t_poseB = t_poseB * torch.from_numpy(local_std).cuda(
        t_poseB.device
    ) + torch.from_numpy(local_mean).cuda(t_poseB.device)
    t_poseB = t_poseB.detach().cpu().numpy()
    # t_poseB = t_poseB.float()


    np.save("./Tpose/%s.npy" %(character_name),t_poseB)


if __name__ == "__main__":
    main()