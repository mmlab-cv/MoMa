import sys
import os
sys.path.append("/home/giulia.martinelli-2/Documents/Code/DATRetarget")
import torch
import numpy as np
from utils.Quaternions import Quaternions
from utils.FKinematics import ForwardKinematics
from utils.BVH_FILE import read_bvh, save_bvh
from dataset.skeleton import build_bone_topology, get_bvh


skeleton = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot',
               'RightToeBase', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm',
               'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']
skeleton_m = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot',
               'RightToeBase', 'Spine', 'Spine1', 'Spine1_split','Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftShoulder_split', 'LeftArm',
               'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightShoulder_split','RightArm', 'RightForeArm', 'RightHand']
skeleton_penguin = ["hip","lThigh","lShin","lFoot","lToe","rThigh","rShin","rFoot","rToe","Stomach","chest","NeckL","NeckM","NeckU","head","lShoulder",
                    "lArm","lForeArm","lHand","rShoulder","rArm","rForeArm","rHand","Rump","Tail","Tail_end"]
skeleton_mewtwo = ["Waist","Upperleg_L","Lowerleg_L","Foot_L","Upperleg_R","Lowerleg_R","Foot_R","Tummy","Chest","Neck","Head",
                   "Upperarm_L","Forearm_L","Hand_L","Upperarm_R","Forearm_R","Hand_R","Tail","Tail_001","Tail_002","Tail_003",
                   "Tail_004","Tail_005","Tail_006","Tail_007"]

skeleton_cmu =['Hips','LeftUpLeg','LeftLeg','LeftFoot','LeftToeBase','RightUpLeg','RightLeg','RightFoot','RightToeBase','Spine','Spine1','Neck1',
'Head','LeftArm','LeftForeArm','LeftHand','LeftHandIndex1','RightArm','RightForeArm','RightHand','RightHandIndex1',]

skeleton_real = ['Hip', 'RightUpLeg','RightLeg','RightFoot','LeftUpLeg','LeftLeg','LeftFoot','Spine','Spine3','Neck','LeftArm','LeftForeArm','LeftHand','RightArm','RightForeArm','RightHand']
skeleton_real = ['Hip','RightHip','RightKnee','RightAnkle','LeftHip','LeftKnee','LeftAnkle','Spine','Thorax','Neck','LeftShoulder','LeftElbow','LeftWrist','RightShoulder','RightElbow',
'RightWrist']

skeleton_smpl = ['Hips','LeftHip','LeftKnee','LeftFoot','RightHip','RightKnee','RightFoot','Waist','Spine','Chest','Neck','LeftInnerShoulder','LeftShoulder',
'LeftElbow','RightInnerShoulder','RightShoulder','RightElbow']
# skeleton_cmu =['Hips','LHipJoint','LeftUpLeg','LeftLeg','LeftFoot','LeftToeBase','RHipJoint','RightUpLeg','RightLeg','RightFoot','RightToeBase','LowerBack','Spine','Spine1','Neck',
# 'Head','LeftShoulder','LeftArm','LeftForeArm','LeftHand','RightShoulder','RightArm','RightForeArm','RightHand']

ee_name_aj = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']

# ee_name_cmu = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']

ee_name_cmu = ['LeftToeBase','RightToeBase','Head','LeftHandIndex1','RightHandIndex1']

ee_name_smpl= ['LeftFoot','RightFoot','Neck','LeftElbow','RightElbow']

ee_name_real = ['LeftAnkle','RightAnkle','Neck','LeftWrist','RightWrist']

ee_name_penguin = ["lToe","rToe","head","lHand","rHand"]
ee_name_mewtwo = ["Foot_L","Foot_R","Head","Hand_L","Hand_R"]

ee_names = [ee_name_aj,ee_name_penguin,ee_name_mewtwo,ee_name_smpl]


class BvhData(object):
    def __init__(self, character_name, motion_file_name, data_path = None, FLAGS=None):
        if data_path == None:
            data_path = FLAGS.dataset_path
        else:
            data_path = "/home/giuliamartinelli/Dataset/MoMaAnimation"
        # if FLAGS.retargeting == 'iso' or FLAGS.retargeting == 'homeo':
        #     self.skeleton_type = 0
        # else:
        # if data_path == '/home/giuliamartinelli/Dataset/MoMaAnimation/MixamoBVH':
        #     self.skeleton_type = 0
        # else:
        #     self.skeleton_type = 3
        # if character_name == "Penguin":
        #     self.skeleton_type = 1
        # elif character_name == "Mewtwo":
        #     self.skeleton_type = 2
        # elif FLAGS.dataset == 'CMU':
        #     self.skeleton_type == 3
        # else:
        #     self.skeleton_type = 0
        
        file_path = os.path.join(data_path, character_name, motion_file_name)
        self.anim, self._names, self.frame_time = read_bvh(file_path)
        self.complete_joint_num = self.anim.shape[1]
        self.edges = []
        self.edge_mat = []  # neighboring matrix
        self.edge_num = 0
        self._topology = None
        self.ee_length = []
        # eliminate the ":" in the JOINT name
        # for i, name in enumerate(self._names):
        #     if ':' in name:
        #         name = name[name.find(':') + 1:]
        #         self._names[i] = name
        #     elif 'm_avg_' in name:
        #         name = name[name.find('m_avg_') + 6:]
        #         self._names[i] = name

        # # self.set_new_root(1)

        # if '_m' in character_name:
        #     self.simplified_name = skeleton_m
        # else:
        #     if character_name == 'Penguin':
        #         self.simplified_name = skeleton_penguin
        #     elif character_name == 'Mewtwo':
        #         self.simplified_name = skeleton_mewtwo
        #         self._names = ['Root','Waist','Tummy','Chest','Neck','Head','Head_end','Neck_001','Neck_002','Neck_002_end',
        #                        'Upperarm_L','Forearm_L','Hand_L','Bone','Bone','Bone','Bone','Bone','Bone','Bone','Bone','Bone',
        #                        'Upperarm_R','Forearm_R','Hand_R','Bone','Bone','Bone','Bone','Bone','Bone','Bone','Bone','Bone',
        #                        'Upperleg_L','Lowerleg_L','Foot_L','Foot','Foot','Foot','Foot','Upperleg_R','Lowerleg_R','Foot_R',
        #                        'Foot','Foot','Foot','Foot',"Tail","Tail_001","Tail_002","Tail_003","Tail_004","Tail_005","Tail_006","Tail_007"]
        #     elif FLAGS.dataset == 'CMU':
        #         self.simplified_name = skeleton_cmu
        #     elif data_path != '/home/giuliamartinelli/Dataset/MoMaAnimation/MixamoBVH':
        #         self.simplified_name = skeleton_smpl
        #     else:
        #         self.simplified_name = skeleton
        # # self.corps store the index in complete skeleton
        # self.corps = []
        # for name in self.simplified_name:
        #     j = self._names.index(name)
        #     self.corps.append(j)
        # self.simplify_joint_num = len(self.corps)

        # ee_id is the end_effector's index list in the simplified skeleton
        # self.ee_id = []
        # for ee_name in ee_names[self.skeleton_type]:
        #     self.ee_id.append(self.simplified_name.index(ee_name))

        # 2 dicts map the index between simple & complete skeletons
        # self.simplify_map = {}
        # self.inverse_simplify_map = {}
        # for simple_idx, complete_idx in enumerate(self.corps):
        #     self.simplify_map[complete_idx] = simple_idx
        #     self.inverse_simplify_map[simple_idx] = complete_idx
        # # TODO why set -1 here ???
        # self.inverse_simplify_map[0] = -1
        self.edges = build_bone_topology(self.topology)
        return

    def scale(self, alpha):
        self.anim.offsets *= alpha
        global_position = self.anim.positions[:, 0, :]
        global_position[1:, :] *= alpha
        global_position[1:, :] += (1 - alpha) * global_position[0, :]

    def rotate(self, theta, axis):
        q = Quaternions(np.hstack((np.cos(theta/2), np.sin(theta/2) * axis)))
        position = self.anim.positions[:, 0, :].copy()
        rotation = self.anim.rotations[:, 0, :]
        position[1:, ...] -= position[0:-1, ...]
        q_position = Quaternions(np.hstack((np.zeros((position.shape[0], 1)), position)))
        q_rotation = Quaternions.from_euler(np.radians(rotation))
        q_rotation = q * q_rotation
        q_position = q * q_position * (-q)
        self.anim.rotations[:, 0, :] = np.degrees(q_rotation.euler())
        position = q_position.imaginaries
        for i in range(1, position.shape[0]):
            position[i] += position[i-1]
        self.anim.positions[:, 0, :] = position

    # @property
    # def topology(self):
    #     if self._topology is None:
    #         self._topology = self.anim.parents[self.corps].copy()
    #         for i in range(self._topology.shape[0]):
    #             if i >= 1:
    #                 self._topology[i] = self.simplify_map[self._topology[i]]
    #     # return a np.array
    #     return self._topology
    
    @property
    def topology(self):
        if self._topology is None:
            self._topology = self.anim.parents.copy()
        return self._topology

    # def get_ee_id(self):
    #     return self.ee_id

    def to_reconstruction_tensor(self):
        rotations = self.get_rotation()
        root_pos = self.get_root_position()
        # reshape the rotation into [frame, (simple_joint_num - 1) * 4]
        rotations = rotations.reshape(rotations.shape[0], -1)
        # reshape the position into [frame, 3]
        root_pos = root_pos.reshape(root_pos.shape[0], -1)
        result = np.concatenate([rotations, root_pos], axis=1)
        # convert to tensor type
        res = torch.tensor(result, dtype=torch.float)
        # in shape [(simple_joint_num - 1) * 4 + 3, frame]
        res = res.permute(1, 0)
        res = res.reshape((-1, res.shape[-1]))
        return res

    def get_all_positions(self):
        """
        Get the position of the root joint
        :return: A numpy Array in shape [frame, n joints, 3]
        """
        position_all = self.anim.positions#[:,self.corps,:]
        return position_all

    def get_root_position(self):
        """
        Get the position of the root joint
        :return: A numpy Array in shape [frame, 1, 3]
        """
        # position in shape[frame, 1, 3]
        position = self.anim.positions[:, 0:1, :]
        return position

    def get_root_rotation(self):
        """
        Get the rotation of the root joint
        :return: A numpy Array in shape [frame, 1, 4]
        """
        # rotation in shape[frame, 1, 4]
        rotation = self.anim.rotations#[:, self.corps, :]
        rotation = rotation[:, 0:1, :]
        # convert to quaternion format
        rotation = Quaternions.from_euler(np.radians(rotation)).qs
        return rotation

    def get_rotation(self):
        """
        Get the rotation of each joint's parent except the root joint
        :return: A numpy Array in shape [frame, simple_joint_num - 1, 4]]
        """
        # rotation in shape[frame, simple_joint_num, 3]
        rotations = self.anim.rotations#[:, self.corps, :]
        # transform euler degree into radians then into quaternion
        # in shape [frame, simple_joint_num, 4]
        rotations = Quaternions.from_euler(np.radians(rotations)).qs
        index = []
        for e in self.edges:
            index.append(e[0])
        # now rotation is in shape[frame, simple_joint_num - 1, 4]
        rotations = rotations[:, index, :]
        return rotations

    @property
    def offset(self) -> np.ndarray:
        # in shape[simple_joint_num, 3]
        return self.anim.offsets#[self.corps]

    # @property
    # def names(self):
    #     return self.simplified_name

    def get_height(self):
        offset = self.offset
        topo = self.topology
        res = 0

        p = self.ee_id[0]
        while p != 0:
            res += np.dot(offset[p], offset[p]) ** 0.5
            p = topo[p]

        p = self.ee_id[2]
        while p != 0:
            res += np.dot(offset[p], offset[p]) ** 0.5
            p = topo[p]
        return res

    def write(self, file_path):
        save_bvh(file_path, self.anim, self.names, self.frame_time, order='xyz')
        return

    def set_new_root(self, new_root):
        euler = torch.tensor(self.anim.rotations[:, 0, :], dtype=torch.float)
        transform = ForwardKinematics.transform_from_euler(euler, 'xyz')
        offset = torch.tensor(self.anim.offsets[new_root], dtype=torch.float)
        new_pos = torch.matmul(transform, offset)
        new_pos = new_pos.numpy() + self.anim.positions[:, 0, :]
        self.anim.offsets[0] = -self.anim.offsets[new_root]
        self.anim.offsets[new_root] = np.zeros((3, ))
        self.anim.positions[:, new_root, :] = new_pos
        rot0 = Quaternions.from_euler(np.radians(self.anim.rotations[:, 0, :]), order='xyz')
        rot1 = Quaternions.from_euler(np.radians(self.anim.rotations[:, new_root, :]), order='xyz')
        new_rot1 = rot0 * rot1
        new_rot0 = (-rot1)
        new_rot0 = np.degrees(new_rot0.euler())
        new_rot1 = np.degrees(new_rot1.euler())
        self.anim.rotations[:, 0, :] = new_rot0
        self.anim.rotations[:, new_root, :] = new_rot1

        new_seq = []
        vis = [0] * self.anim.rotations.shape[1]
        new_idx = [-1] * len(vis)
        new_parent = [0] * len(vis)

        def relabel(x):
            nonlocal new_seq, vis, new_idx, new_parent
            new_idx[x] = len(new_seq)
            new_seq.append(x)
            vis[x] = 1
            for y in range(len(vis)):
                if not vis[y] and (self.anim.parents[x] == y or self.anim.parents[y] == x):
                    relabel(y)
                    new_parent[new_idx[y]] = new_idx[x]
        relabel(new_root)
        self.anim.rotations = self.anim.rotations[:, new_seq, :]
        self.anim.offsets = self.anim.offsets[new_seq]
        names = self._names.copy()
        for i, j in enumerate(new_seq):
            self._names[i] = names[j]
        self.anim.parents = np.array(new_parent, dtype=int)