import numpy as np
from bvh_parser import BvhData
from data_flags import FLAGS
from absl import app
from absl import flags
import pickle
from tqdm import tqdm

def get_bvh_file_names(FLAGS,character_name):
    if '_m' in character_name:
        character_name = character_name.split('_m')[0]
    if FLAGS.mode == 'train':
        file = open(f'/home/giulia.martinelli-2/Documents/Code/DATRetarget/dataset/filelist/train_{character_name.lower()}.txt', 'r')
    elif FLAGS.mode == 'validation':
        file = open(f'/home/giulia.martinelli-2/Documents/Code/DATRetarget/dataset/filelist/test_{character_name.lower()}.txt', 'r')
    files_list = file.readlines()
    files_list = [f[:-1][:-4] for f in files_list]
    return files_list

def _concat_together(rot, root_pos):
    """
    concatenate the rotation, root_position together as the dynamic input of the
    neural network
    :param rot: rotation matrix with shape [frame, simple_joint_num - 1, 4]
    :param root_pos: with shape [frame, 1, 3], pad a 0 in dim=2, to make the position with shape
    [frame, 1, 4]
    :return: tensor with shape [frame, simple_joint_num, 4]
    """
    frame_num = root_pos.shape[0]    # pad 0 make root_pos with shape [frame, 1, 4]
    if rot.shape[-1] == 4:
        pad_root_pos = np.zeros([frame_num, 1, 4], dtype=np.float)
    else:
        pad_root_pos = np.zeros([frame_num, 1, 6], dtype=np.float)

    pad_root_pos[:, :, 0:3] = root_pos
    # concatenate all together
    result = np.concatenate([pad_root_pos, rot], axis=1)
    return result


def _to_format_tensor(t):
    """
    :return: Tensor with shape [4, simple_joint_num, frame]
    """
    result = t.transpose(2, 1, 0)
    return result

def _slice_to_equal_frame_len(input_tensor):
    """
    ONLY USED DURING TRAINING STAGE
    :param input_tensor: tensor in shape [7, simple_joint_num, frame]
    :return:tensor in shape [B, 7, simple_joint_num, args.window_size]
    Where B depends on frame and args.window_size
    """
    win_size = FLAGS.window_size
    total_frame = input_tensor.shape[2]
    win_num = total_frame // win_size
    if win_num == 0:
        raise Exception("The total frame is less than window_size!")
    result_list = []
    for i in range(win_num):
        tmp_frame_idx = range(i*win_size, (i+1)*win_size)
        tmp_tensor = input_tensor[:, :, tmp_frame_idx]
        # expand dim to [1, 7, simple_joint_num, args.window_size]
        tmp_tensor = np.expand_dims(tmp_tensor, axis=0)
        result_list.append(tmp_tensor)
    return np.concatenate(result_list, axis=0)


def main(argv):  

    character_names_train= ["Aj","BigVegas","James","James_m","Parasite","Parasite_m","Joe","Joe_m","Liam","Liam_m","LolaB","LolaB_m","Maria","Maria_m",
                       "Olivia","Olivia_m","Paladin","Paladin_m","Pumpkinhulk","Pumpkinhulk_m","Remy","Remy_m","SportyGranny","Timmy","Timmy_m","Yaku","Yaku_m",
                       "Mousey","Knight_m","Goblin","Goblin_m","Abe","Abe_m","Knight"]
    
    character_names_testing = ["Ortiz","SportyGranny","Mousey_m","Vampire_m"]

    character_names =  ["Ortiz_m","Claire","Michelle","Racer_m","Michelle_m","Racer","Mremireh_m","Mremireh","Knight_m","Vampire"]
    
    skeleton_m = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot',
               'RightToeBase', 'Spine', 'Spine1', 'Spine1_split','Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftShoulder_split', 'LeftArm',
               'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightShoulder_split','RightArm', 'RightForeArm', 'RightHand']

    std_bvh_data = []
    edges = []
    names = []
    offsets = []
    ee_ids = []
    topologies = []
    bvh_names = []
    char_names = []
    
    mapping = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,16,17,18,19,21,22,23,24]
    insert_name_index = [11,15,20]
    insert_names = ['Spine1_split','LeftShoulder_split','RightShoulder_split']

    # contains numpy arrays in shape[1, (J - 1), 4]
    # contains numpy arrays in shape [1, 1, 3]
    pos_means = []
    pos_vars = []

    # Save topologies of every character
    joint_nums = []
    for i in tqdm(range(len(character_names))):
        if '_m' in character_names[i]:
            char = character_names[i].split('_m')[0]
            type_skel = 'homeo'
        else:
            char = character_names[i]
            type_skel = 'iso'
        std_bvh_data.append(BvhData(character_names[i], motion_file_name=char+'.bvh', FLAGS=FLAGS))
        

        bvh_file_list = get_bvh_file_names(FLAGS, character_name=character_names[i])

        # Load All motions
        character_data_rot = []
        character_data_pos = []

        for j in tqdm(range(len(bvh_file_list))):
            topologies_new = np.zeros(shape=(25))

            if type_skel == 'iso':
                topologies_new[:22] = std_bvh_data[i].topology
                edges_new = std_bvh_data[i].edges + [(0,0)] * 3
                offset_new = std_bvh_data[i].offset
                joint_nums.append(22)
                names_new = std_bvh_data[i].names[:insert_name_index[0]] + [insert_names[0]] + std_bvh_data[i].names[insert_name_index[0]:insert_name_index[1]] + [insert_names[1]] + std_bvh_data[i].names[insert_name_index[1]:insert_name_index[2]] + [insert_names[2]] + std_bvh_data[i].names[insert_name_index[2]:]
            else:
                topologies_new = std_bvh_data[i].topology
                edges_new = std_bvh_data[i].edges
                offset_new = std_bvh_data[i].offset.reshape(1,25,3)
                joint_nums.append(25)
                names_new = std_bvh_data[i].names
            topologies.append(topologies_new)
            edges.append(edges_new)
            # ee_ids.append(std_bvh_data[i].get_ee_id())
            names.append(names_new)
            # the offset now in shape [simple_joint_num, 3]
            offsets.append(offset_new)
            
            bvh_name = bvh_file_list[j]+'.bvh'
            bvh_names.append(bvh_file_list[j])
            char_names.append(character_names[i])
            bvh_data = BvhData(character_names[i], motion_file_name=bvh_name,FLAGS = FLAGS)
            # [frame, simple_joint_num - 1, 4]
            rotation = bvh_data.get_rotation()
            # [frame, 1, 3]
            root_position = bvh_data.get_root_position()
            concat_tensor = _concat_together(rotation, root_position)
            rotation_super = np.zeros(shape=(concat_tensor.shape[0],25,4))
            if concat_tensor.shape[1] == 25:
                rotation_super = concat_tensor
            else:
                rotation_super[:,mapping,:] = concat_tensor
            final_output_rot = _to_format_tensor(rotation_super)
            final_output_rot = _slice_to_equal_frame_len(final_output_rot)
            character_data_rot.append(final_output_rot)

    data = {}
    data['name'] = char_names
    data['motion_name'] = char_names
    data['motion'] = character_data_rot
    data['tolopogy'] = topologies
    data['offset'] = offsets
    data['joints_num'] = joint_nums
    data['edges'] = edges
    data['joints_names'] = names

    with open('/media/mmlab/Volume2/Mixamo/val2_superskel.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # character_data_rot = np.cat(character_data_rot)

if __name__ == '__main__':
    app.run(main)