import os
import numpy as np
import rawpy
import torch
from unetTorch import Unet


def write_back_dng(src_path, dest_path, raw_data):
    """
    replace dng data
    """
    width = raw_data.shape[0]
    height = raw_data.shape[1]
    falsie = os.path.getsize(src_path)
    data_len = width * height * 2
    header_len = 8

    with open(src_path, "rb") as f_in:
        data_all = f_in.read(falsie)
        dng_format = data_all[5] + data_all[6] + data_all[7]

    with open(src_path, "rb") as f_in:
        header = f_in.read(header_len)
        if dng_format != 0:
            _ = f_in.read(data_len)
            meta = f_in.read(falsie - header_len - data_len)
        else:
            meta = f_in.read(falsie - header_len - data_len)
            _ = f_in.read(data_len)

        data = raw_data.tobytes()

    with open(dest_path, "wb") as f_out:
        f_out.write(header)
        if dng_format != 0:
            f_out.write(data)
            f_out.write(meta)
        else:
            f_out.write(meta)
            f_out.write(data)

    if os.path.getsize(src_path) != os.path.getsize(dest_path):
        print("replace raw data failed, file size mismatch!")
    else:
        print("replace raw data finished")


def inv_normalization(input_data, black_level, white_level):
    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)
    return output_data


def write_image(input_data, height, width):
    output_data = np.zeros((height, width), dtype=np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[channel_y:height:2, channel_x:width:2] = input_data[0:, :, :, 2 * channel_y + channel_x]
    return output_data


def read_image(input_path):
    raw = rawpy.imread(input_path)
    raw_data = raw.raw_image_visible
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    return raw_data_expand_c, height, width


def normalization(input_data, black_level, white_level):
    output_data = (input_data.astype(float) - black_level) / (white_level - black_level)
    return output_data


def expand_data(data_tensor):
    pad = torch.nn.ReplicationPad2d((0, 248, 0, 184))
    padding_data = pad(data_tensor)
    return padding_data


def split_data(data_tensor):
    split_h1_list = []
    split_h2_list = []
    split_h3_list = []
    split_h = torch.split(data_tensor, 640, dim=2)
    split_h1 = split_h[0]
    split_h2 = split_h[1]
    split_h3 = split_h[2]
    split_w1 = torch.split(split_h1, 640, dim=3)
    split_w2 = torch.split(split_h2, 640, dim=3)
    split_w3 = torch.split(split_h3, 640, dim=3)
    for ind in split_w1:
        split_h1_list.append(ind)
    for ind in split_w2:
        split_h2_list.append(ind)
    for ind in split_w3:
        split_h3_list.append(ind)
    return split_h1_list, split_h2_list, split_h3_list


def cat_list_tensor(input_list):
    return torch.cat(input_list, dim=3)


def cat_tensor(tensor1, tensor2, tensor3):
    return torch.cat([tensor1, tensor2, tensor3], dim=2)


raw_data_expand_c, height, width = read_image("./data/testset/noisy9.dng")
raw_data_expand_c_normal = normalization(raw_data_expand_c, 1024, 16383)
raw_data_expand_c_normal = torch.from_numpy(np.transpose(
    raw_data_expand_c_normal.reshape(-1, height // 2, width // 2, 4), (0, 3, 1, 2))).float()

padding_data = expand_data(raw_data_expand_c_normal)
tensor_list1, tensor_list2, tensor_list3 = split_data(padding_data)
device = torch.device('cpu')
net = Unet().to(device)

net.load_state_dict(torch.load("./models/epoch_450.pth"))
net.eval()
result_list1 = [net(x) for x in tensor_list1]
result_list2 = [net(x) for x in tensor_list2]
result_list3 = [net(x) for x in tensor_list3]

cat_tensor1 = cat_list_tensor(result_list1)
cat_tensor2 = cat_list_tensor(result_list2)
cat_tensor3 = cat_list_tensor(result_list3)
re_tensor = cat_tensor(cat_tensor1, cat_tensor2, cat_tensor3)
re_img = re_tensor[:, :, 0:1736, 0:2312]
re_img = re_img.detach().numpy().transpose(0, 2, 3, 1)
result_img = inv_normalization(re_img, 1024, 16383)
result_write_data = write_image(result_img, height, width)
write_back_dng("./data/testset/noisy9.dng", "./data/result/denoise9.dng", result_write_data)
