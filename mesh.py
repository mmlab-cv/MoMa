import torch.nn as nn
from src.linear_blend_skin import linear_blend_skinning
import numpy as np
import torch
from typing import Optional, Dict, Union
from common import (MESHOutput, Tensor, Array)
from einops import rearrange
import sys
import os

from putils.Quaternions import Quaternions

class MESH(nn.Module):
    
    def __init__(
            self, fbx_data, tpose, parents=None
    ):
        super().__init__()

        self.tpose = tpose

        vertices_np = fbx_data['rest_vertices']
        sk_weights_np = fbx_data['skinning_weights']
        self.faces = fbx_data['rest_faces']

        self.rest_vertices = (
        torch.from_numpy(vertices_np).cuda().type(torch.float)
        )

        self.sk_weights = (
            torch.from_numpy(sk_weights_np).cuda().type(torch.float)
        )

        
        self.parents = np.array(
            [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20]
        )

        default_body_pose = torch.zeros(
                    [1, 22, 4], dtype=torch.float32)

        self.body_pose = nn.Parameter(default_body_pose,
                                         requires_grad=True)


        
    def forward(
        self
    ) -> MESHOutput:
               
        body_pose = self.body_pose
        

        if torch.any(body_pose.data) == False:
            self.vertices = self.rest_vertices.reshape(1,self.rest_vertices.shape[0],self.rest_vertices.shape[1])

        else:

            self.vertices = linear_blend_skinning(
                self.parents, body_pose, self.tpose[0], self.rest_vertices, self.sk_weights
            )



        output = MESHOutput(vertices=self.vertices,
                            faces = self.faces,
                            body_pose=body_pose)

        return output
    
    @torch.no_grad()
    def set_params(self,anim,index) -> None:
        quat_tpose = (anim.rotations[index,:,:]).reshape(-1,22,3)
        self.body_pose.data = torch.tensor(Quaternions.from_euler(np.radians(quat_tpose)).qs,dtype=torch.float32).cuda()

    def set_pose(self,pose) -> None:
        self.body_pose.data = torch.tensor(pose,dtype=torch.float32).cuda()
    @torch.no_grad()
    def reset_params(self) -> None:
        self.body_pose.data = torch.zeros([1, 22, 4], dtype=torch.float32).cuda()

    @torch.no_grad()
    def prev_pose(self,anim,index,char_name) -> None:
        if '_m' in char_name:
            mapping_back = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,16,17,18,19,21,22,23,24]
        else:
            mapping_back = list(range(22))
        mapping = [0,6,7,8,9,10,11,12,13,1,2,3,4,5,14,15,16,17,18,19,20,21]
        quat_tpose = np.zeros(shape=(1,22,3))
        # joints = [0,1,2,3,4,6,47,48,49,50,52,53,54,55,7,8,9,10,27,28,29,30]
        # joints = [0,1,2,3,4,5,71,72,73,74,77,78,79,80,10,11,12,13,35,36,37,38]
        quat_tpose[:,mapping,:] = (anim.rotations[index,mapping_back,:]).reshape(-1,22,3)
        return torch.tensor(Quaternions.from_euler(np.radians(quat_tpose)).qs,dtype=torch.float32).cuda()




def create(
   fbx_data,tpose
):
    return MESH(fbx_data,tpose)