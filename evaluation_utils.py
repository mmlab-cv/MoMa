import numpy as np

def de_normalize(raw):
    rot_part = raw[:, 1:, :]
    pos_part = raw[:, 0:1, :]
    return rot_part, pos_part

def get_folders_with_to(path,dis=None):
    folders_with_to = []
    
    # Get the list of all items (files and folders) in the specified path
    items = os.listdir(path)
    
    # Filter only directories containing '_to_' in their names
    folders_with_to = [item for item in items if os.path.isdir(os.path.join(path, item)) and '_to_' + dis in item  and dis + '_to_' + dis not in item]
    
    return folders_with_to

def concat_together(rot, root_pos):
    """
    concatenate the rotation, root_position together as the dynamic input of the
    neural network
    :param rot: rotation matrix with shape [frame, simple_joint_num - 1, 4]
    :param root_pos: with shape [frame, 1, 3], pad a 0 in dim=2, to make the position with shape
    [frame, 1, 4]
    :return: tensor with shape [frame, simple_joint_num, 4]
    """
    frame_num = root_pos.shape[0]
    # pad 0 make root_pos with shape [frame, 1, 4]
    if rot.shape[-1] == 4:
        pad_root_pos = np.zeros([frame_num, 1, 4], dtype=float)
    else:
        pad_root_pos = np.zeros([frame_num, 1, 6], dtype=float)

    pad_root_pos[:, :, 0:3] = root_pos
    # concatenate all together
    result = np.concatenate([pad_root_pos, rot], axis=1)
    return result

def to_format_tensor(t):
    """
    :return: Tensor with shape [4, simple_joint_num, frame]
    """
    result = t.transpose(2, 1, 0)
    return result
def slice_to_equal_frame_len(input_tensor):
    """
    ONLY USED DURING TRAINING STAGE
    :param input_tensor: tensor in shape [7, simple_joint_num, frame]
    :return:tensor in shape [B, 7, simple_joint_num, args.window_size]
    Where B depends on frame and args.window_size
    """
    win_size = 8
    total_frame = input_tensor.shape[2]
    win_num = total_frame // win_size
    if win_num == 0:
        raise Exception("The total frame is less than window_size!")
    result_list = []
    for i in range(win_num):
        tmp_frame_idx = range(i*win_size, (i+1)*win_size)
        tmp_tensor = input_tensor[:, :, tmp_frame_idx]
        # expand dim to [1, 7, simple_joint_num, args.window_size]
        tmp_tensor = np.expand_dims(tmp_tensor,axis=0)
        result_list.append(tmp_tensor)
    return np.concatenate(result_list, axis=0)