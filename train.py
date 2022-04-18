import os
import rawpy
import torch
import numpy as np
from unetTorch import Unet
import torch.optim as optim
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--black_level', type=int, default=1024)
parser.add_argument('--white_level', type=int, default=16383)
parser.add_argument('--noisy', type=str, default="noisy")
parser.add_argument('--gt', type=str, default="ground truth")
parser.add_argument('--epochs', type=int, default=500)
args = parser.parse_args()


# data:
def get_data(input_path):
    train_data_path = []
    data_root = os.getcwd()
    data_path = os.path.join(data_root, 'dataset', input_path)
    for file in os.listdir(data_path):
        if file.endswith('.dng'):
            file_path = os.path.join(data_path, file)
            train_data_path.append(file_path)
        else:
            continue
    return train_data_path


def normalization(input_data, black_level, white_level):
    output_data = (input_data.astype(float) - black_level) / (white_level - black_level)
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


def random_flip(input_patch, gt_patch):
    if np.random.randint(3, size=1)[0] == 0:  # random flip
        input_patch = np.flip(input_patch, axis=1)
        gt_patch = np.flip(gt_patch, axis=1)
    elif np.random.randint(3, size=1)[0] == 1:
        input_patch = np.flip(input_patch, axis=2)
        gt_patch = np.flip(gt_patch, axis=2)
    else:
        input_patch = input_patch
        gt_patch = gt_patch
    return input_patch, gt_patch


def to_tensor(input_patch, black_level, white_level, patch_size):
    raw_data_expand_c_normal = normalization(input_patch, black_level, white_level)
    raw_data_expand_c_normal = torch.from_numpy(np.transpose(
        raw_data_expand_c_normal.reshape(-1, patch_size, patch_size, 4), (0, 3, 1, 2))).float()
    return raw_data_expand_c_normal


# loss
def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()


g_loss = np.zeros((200, 1))
device = torch.device('cuda')
learning_rate = 1e-4
model = Unet()
model._initialize_weights()
model = model.to(device)
save_freq = 50
opt = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.1, last_epoch=- 1)
patch_size = [512, 640, 768, 896, 1024]
for epoch in range(args.epochs):
    noisy_path = get_data(args.noisy)  # get list of noisy
    gt_path = get_data(args.gt)  # get list of gt
    for ind in range(len(noisy_path)):
        noisy_, height, width = read_image(noisy_path[ind])  # get noisy_patch
        gt_, _, _ = read_image(gt_path[ind])  # get gt_patch
        ps = random.choice(patch_size)
        xx = np.random.randint(0, width // 2 - ps)
        yy = np.random.randint(0, height // 2 - ps)
        noisy_patch = noisy_[yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_[yy:yy + ps, xx:xx + ps, :]
        flip_noisy_patch, flip_gt_patch = random_flip(noisy_patch, gt_patch)  # get random_flip noisy_path gt_noisy
        noisy_img = to_tensor(flip_noisy_patch, args.black_level, args.white_level, patch_size=ps)
        gt_img = to_tensor(flip_gt_patch, args.black_level, args.white_level, patch_size=ps)
        noisy_img = noisy_img.to(device)
        gt_img = gt_img.to(device)
        out_img = model(noisy_img)
        loss = reduce_mean(out_img, gt_img)
        opt.zero_grad()
        loss.backward()
        opt.step()
        g_loss[ind] = loss.data.cpu()
    scheduler.step()
    mean_loss = np.mean(g_loss)
    print('epoch:{}, mean_loss:{}'.format(epoch, mean_loss))
    if epoch % save_freq == 0:
        model_state = model.state_dict()
        save_path = os.path.join(os.getcwd(), 'checkpoints')
        torch.save(model_state, os.path.join(save_path, 'epoch' + '_' + str(epoch) + '.pth'))
