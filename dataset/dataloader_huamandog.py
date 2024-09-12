"""
The purpose of our task is retarget the skeleton A to skeleton B
So the dataset will load both A & B's .bvh motion data
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from dataset import get_bvh_file_names
from dataset.bvh_parser import BvhData
from einops import rearrange
import random
from geometry import quat_to_d6
import itertools
from sklearn.preprocessing import normalize
from tqdm import tqdm


class HumanDogDataModule(pl.LightningDataModule):
    
    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS = FLAGS

        if self.FLAGS.mode == 'train':
            self.train_dataset = HumanDogDataset(self.FLAGS, mode = 'train')
            self.val_dataset = HumanDogDataset(self.FLAGS, mode = 'validation')
        else:
            self.demo_dataset = HumanDogDataset(self.FLAGS, mode = 'demo')

    def prepare_data(self):
        print('Files have been loaded')

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset, 
            shuffle=False, 
            pin_memory=True,
            num_workers=self.FLAGS.num_workers, 
            batch_size=self.FLAGS.batch_size,
            drop_last=True,
            persistent_workers=True
        )

        return train_loader

    def val_dataloader(self):

        val_loader = DataLoader(
            self.val_dataset, 
            shuffle=False, 
            pin_memory=True,
            num_workers=self.FLAGS.num_workers, 
            batch_size=self.FLAGS.batch_size,
            drop_last=True,
            persistent_workers=True
        )

        return val_loader

    def test_dataloader(self):

        test_loader = DataLoader(
            self.val_dataset, 
            shuffle=False, 
            pin_memory=True,
            num_workers=self.FLAGS.num_workers, 
            batch_size=1, 
            drop_last=True,
            persistent_workers=True
        )

        return test_loader


class HumanDogDataset(Dataset):
    def __init__(self, FLAGS, mode = None):
        super(HumanDogDataset, self).__init__()
        self.FLAGS = FLAGS
        self.mode = mode
        
        self.mapping_dog = [0,13,14,15,16,17,18,1,2,3,4,5,6,7,8,9,10,11,12,19,20]
        self.mapping_super = [0,1,2,3,5,6,7,9,10,11,12,14,15,16,17,18,19,20,21,22,23]#[0,1,2,22,5,6,23,9,10,11,12,14,15,16,17,18,19,20,21,24,25]

        
        

        if self.mode == 'train':
            

            self.character_names = ["Human","Dog"]
            
            self.character_idx = dict()

            for i in range(len(self.character_names)):
                self.character_idx[self.character_names[i]] = i/(len(self.character_names)-1)

            
            std_bvh_data = []
            self.edges = []
            self.names = []
            self.offsets = []
            # self.ee_ids = []
            self.topologies = []

            # contains numpy arrays in shape[1, (J - 1), 4]
            # contains numpy arrays in shape [1, 1, 3]
            self.pos_means = []
            self.pos_vars = []
            self.index_tot = []
            self.id_tot = []
            # Load All motions
            self.character_data_rot = []
            self.character_data_pos = []

            # Save topologies of every character
            self.joint_nums = []
            for i in tqdm(range(len(self.character_names))):
                char = self.character_names[i]
                std_bvh_data.append(BvhData(self.character_names[i], motion_file_name=char+'.bvh', FLAGS=FLAGS))
                self.topologies.append(std_bvh_data[i].topology)
                self.edges.append(std_bvh_data[i].edges)
                # self.ee_ids.append(std_bvh_data[i].get_ee_id())
                # self.names.append(std_bvh_data[i].names)
                # the offset now in shape [simple_joint_num, 3]
                offset = torch.from_numpy(std_bvh_data[i].offset).float()
            
                
                self.offset = offset.cuda()
                
                self.offsets.append(self.offset)
                self.joint_nums.append(offset.shape[0])

                self.id = self.character_idx[self.character_names[i]]

                
                bvh_file_list = get_bvh_file_names(mode, character_name=self.character_names[i])
                
                # # Select 60 random motions for each character
                # bvh_file_list = random.sample(bvh_file_list, 60)

            
                
                for j in tqdm(range(len(bvh_file_list))):
                    
                    bvh_name = bvh_file_list[j]+'.bvh'
                    bvh_data = BvhData(self.character_names[i], motion_file_name=bvh_name,FLAGS = self.FLAGS)
                    # [frame, simple_joint_num - 1, 4]
                    rotation = bvh_data.get_rotation()
                    # [frame, 1, 3]
                    root_position = bvh_data.get_root_position()
                    rotation, root_position = self._normalize(rotation, root_position)
                    concat_tensor = self._concat_together(rotation, root_position)
                    rotation_super = torch.zeros(size=(concat_tensor.shape[0],self.FLAGS.n_joints,4))
                    if concat_tensor.shape[1] == 22:
                        rotation_super[:,:22,:] = concat_tensor
                    else:
                        rotation_super[:,self.mapping_super,:] = concat_tensor[:,self.mapping_dog,:]
                    final_output_rot = self._to_format_tensor(rotation_super)
                    final_output_rot = self._slice_to_equal_frame_len(final_output_rot)
                    self.character_data_rot.append(final_output_rot)
                    
                    self.index =  [i] * final_output_rot.shape[0]

                    self.character_id = [self.id] * final_output_rot.shape[0]

                    self.index_tot.append(self.index)

                    self.id_tot.append(self.character_id)
                    
            self.indexes = torch.tensor(list(itertools.chain(*self.index_tot))).unsqueeze(1)

            self.character_ids = list(itertools.chain(*self.id_tot))

            self.character_data_rot = torch.cat(self.character_data_rot)

            # self.character_ids_norm = normalize([self.character_ids], norm="l1")[0]
            
            

            self.character_data_rot = rearrange(self.character_data_rot,'b q j w -> b w j q')
            
        elif self.mode == 'validation':
            
            self.character_names = ["Human","Dog"]
            
            self.character_idx = dict()

            for i in range(len(self.character_names)):
                self.character_idx[self.character_names[i]] = i/(len(self.character_names)-1)
        
            
            std_bvh_data = []
            self.val_edges = []
            # self.names = []
            self.val_offsets = []
            # self.ee_ids = []
            self.topologies = []

            # contains numpy arrays in shape[1, (J - 1), 4]
            # contains numpy arrays in shape [1, 1, 3]
            self.pos_means = []
            self.pos_vars = []
            self.index_tot_val = []
            self.id_tot_val = []
            # Load All motions
            self.character_data_rot_val = []
            self.character_data_pos = []

            # Save topologies of every character
            self.joint_nums = []
            for i in tqdm(range(len(self.character_names))):
                char = self.character_names[i]
                std_bvh_data.append(BvhData(self.character_names[i], motion_file_name=char+'.bvh', FLAGS=FLAGS))
                self.topologies.append(std_bvh_data[i].topology)
                self.val_edges.append(std_bvh_data[i].edges)
                # self.ee_ids.append(std_bvh_data[i].get_ee_id())
                # self.names.append(std_bvh_data[i].names)
                # the offset now in shape [simple_joint_num, 3]
                offset = torch.from_numpy(std_bvh_data[i].offset).float()
                self.offset = offset.cuda()
                self.val_offsets.append(self.offset)
                self.joint_nums.append(offset.shape[0])

                self.id = self.character_idx[self.character_names[i]]

                
                bvh_file_list = get_bvh_file_names(mode, character_name=self.character_names[i])

                
                for j in tqdm(range(len(bvh_file_list))):
                    
                    bvh_name = bvh_file_list[j]+'.bvh'
                    bvh_data = BvhData(self.character_names[i], motion_file_name=bvh_name,FLAGS = self.FLAGS)
                    # [frame, simple_joint_num - 1, 4]
                    rotation = bvh_data.get_rotation()
                    # [frame, 1, 3]
                    root_position = bvh_data.get_root_position()
                    rotation, root_position = self._normalize(rotation, root_position)
                    concat_tensor = self._concat_together(rotation, root_position)
                    rotation_super = torch.zeros(size=(concat_tensor.shape[0],self.FLAGS.n_joints,4))
                    if concat_tensor.shape[1] == 22:
                        rotation_super[:,:22,:] = concat_tensor
                    else:
                        rotation_super[:,self.mapping_super,:] = concat_tensor[:,self.mapping_dog,:]
                    final_output_rot = self._to_format_tensor(rotation_super)
                    final_output_rot = self._slice_to_equal_frame_len(final_output_rot)
                    self.character_data_rot_val.append(final_output_rot)
                    
                    self.index =  [i] * final_output_rot.shape[0]

                    self.character_id = [self.id] * final_output_rot.shape[0]

                    self.index_tot_val.append(self.index)

                    self.id_tot_val.append(self.character_id)

                    
            self.indexes_val = torch.tensor(list(itertools.chain(*self.id_tot_val))).unsqueeze(1)

            self.character_ids_val = list(itertools.chain(*self.id_tot_val))

            # self.character_ids_norm_val = normalize([self.character_ids_val], norm="l1")[0]

            self.character_data_rot_val = torch.cat(self.character_data_rot_val)
            
            self.character_data_rot_val= rearrange(self.character_data_rot_val,'b q j w -> b w j q')
            
            
    def __len__(self):
        if self.mode == 'train':
            return self.character_data_rot.shape[0]
        else:
            return self.character_data_rot_val.shape[0]

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.mode == 'train':
            sample['msk_rot'] = self.character_data_rot[idx]
            sample['index'] = self.indexes[idx]
            sample['id'] = self.character_ids[idx]
        else:
            sample['msk_rot'] = self.character_data_rot_val[idx]
            sample['index'] = self.indexes_val[idx]
            sample['id'] = self.character_ids_val[idx]

        return sample

    def _normalize(self, rot, root_pos): #character_idx: int
        """
        :param rot: [frame, simple_joint_num - 1, 4]
        :param root_pos:  [frame, 1, 3]
        :param character_idx: idx for get mean and var for different character
        """
        rot = self._convert_to_tensor(rot)
        root_pos = self._convert_to_tensor(root_pos)

        return rot, root_pos

    def de_normalize(self, raw: torch.tensor):
        device = raw.device
        rot_part = raw[:, :, 1:, :]
        pos_part = raw[:, 0:-1, 0:1, :]
        # denormalize
        # rot_part = (rot_part * rot_var) + rot_mean
        # pos_part = (pos_part * pos_var) + pos_mean
        return rot_part, pos_part

    def _concat_together(self, rot, root_pos):
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
            pad_root_pos = pad_root_pos.cuda()
            result = torch.cat([pad_root_pos, rot], dim=1)
        return result

    def _convert_to_tensor(self, np_array):
        result = torch.from_numpy(np_array).to(torch.float)
        return result

    def _to_format_tensor(self, t):
        """
        :return: Tensor with shape [4, simple_joint_num, frame]
        """
        result = t.permute(2, 1, 0)
        return result

    def _slice_to_equal_frame_len(self, input_tensor):
        """
        ONLY USED DURING TRAINING STAGE
        :param input_tensor: tensor in shape [7, simple_joint_num, frame]
        :return:tensor in shape [B, 7, simple_joint_num, args.window_size]
        Where B depends on frame and args.window_size
        """
        win_size = self.FLAGS.window_size
        total_frame = input_tensor.size(2)
        win_num = total_frame // win_size
        if win_num == 0:
            raise Exception("The total frame is less than window_size!")
        result_list = []
        for i in range(win_num):
            tmp_frame_idx = range(i*win_size, (i+1)*win_size)
            tmp_tensor = input_tensor[:, :, tmp_frame_idx]
            # expand dim to [1, 7, simple_joint_num, args.window_size]
            tmp_tensor.unsqueeze_(0)
            result_list.append(tmp_tensor)
        return torch.cat(result_list, dim=0)

    def convert_to_bvh_write_format(self, raw: torch.tensor): #, character_idx
        """
        :param raw: in shape [B, 4, J, frame], since this function only called during inference stage,
        the B is always equal to 1
        :param character_idx: an int number
        :return: tensor with shape
        """
        # denormalize first
        # [1, 4, J-1, frame], [1, 3, 1, frame]
        denorm_rot, denorm_root_pos = self.de_normalize(raw)
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