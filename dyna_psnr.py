import cv2
import numpy as np
from PIL import Image
import argparse
import torch
import os
import torchvision.transforms as transforms
from models import dyna_wdsr
from models.dyna_wdsr import update_argparser

size_h = 10
size_w = 20
lr_index = 1
split_times = 0
cp_list = []

parser = argparse.ArgumentParser()

parser.add_argument("--size_w",type=int,default=20, help="number of patches on wide")
parser.add_argument("--size_h",type=int,default=10, help="number of patches on height")
parser.add_argument("--scale", type=int, help="SR type", default=2, choices=[2,3,4])
parser.add_argument("--source_path",type=str,default="/home/lee/data/")
parser.add_argument("--tt",type=str,default="vlog_15")
parser.add_argument("--length", type=int, help="video length", default=15)
parser.add_argument("--dataset",type=str,default="vsd4k")
parser.add_argument("--psnr_threshold",type=int,default=40, help="psnr_threshold")

args, _ = parser.parse_known_args()
size_w = args.size_w
size_h = args.size_h
num_frames = args.length*30
def load_model(checkpoint_path):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  args = update_argparser(parser)
  model, criterion, optimizer, lr_scheduler, metrics = dyna_wdsr.get_model_spec(args)
  checkpoint = torch.load(checkpoint_path, map_location=device)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()

  return model

def frame_check(frame):
    for i in frame:
        if i.__contains__('.png'):
            pass
        else:
            frame.remove(i)
    return frame

def gen_data(lr_path,hr_path,lr_folder_path,hr_folder_path):
    lr_frame = os.listdir(lr_path)
    lr_frame = frame_check(lr_frame)
    lr_frame.sort(key=lambda x: int(x[:-6]))
    lr_frame = lr_frame[0:num_frames]
    lr_frame_path = []
    for frame_name in lr_frame:
        lr_frame_path.append(lr_path + '/' + frame_name)
    hr_frame = os.listdir(hr_path)
    hr_frame = frame_check(hr_frame)
    hr_frame.sort(key=lambda x: int(x[:-4]))
    hr_frame = hr_frame[0:num_frames]
    hr_frame_path = []
    index = 0
    for frame_name in hr_frame:
        hr_frame_path.append(hr_path + '/' + frame_name)
    for i,j in zip(lr_frame_path,hr_frame_path):
        img_patches(i,j,size_w,size_h,lr_folder_path,hr_folder_path,args.psnr_threshold)
        index = index + 1
        print('image {} processed'.format(index))
        if index % 100 == 0:
            print('processing: {}/{}'.format(index,num_frames))

def img_patches(lr_path,hr_path,size_w,size_h,lr_folder_path,hr_folder_path,psnr_threshold):
    """
    :param path: frame path
    :param size_w: patch size wide
    :param size_h: patch size height
    """
    lr_folder_path = lr_folder_path
    hr_folder_path = hr_folder_path
    lr_image = Image.open(lr_path)
    hr_image = Image.open(hr_path)
    l_w,l_h = lr_image.size
    w,h = hr_image.size
    len_max_l_h = int(l_h/size_h) * 2
    len_max_l_w = int(l_w/size_w) * 2
    len_max_h_h = int(h/size_h) * 2
    len_max_h_w = int(w/size_w) * 2
    count = 0
    if not os.path.exists(lr_folder_path):
        os.makedirs(lr_folder_path)
    if not os.path.exists(hr_folder_path):
        os.makedirs(hr_folder_path)
    for i in range(0,int(size_h/2)):
        for j in range(0,int(size_w/2)):
            count = count+1
            lr_box = (j*len_max_l_w,i*len_max_l_h,(j+1)*len_max_l_w,(i+1)*len_max_l_h,)
            lr_patch = lr_image.crop(lr_box)
            hr_box = (j * len_max_h_w, i * len_max_h_h, (j + 1) * len_max_h_w, (i + 1) * len_max_h_h,)
            hr_patch = hr_image.crop(hr_box)
            exam_complexity(lr_patch,hr_patch,1,lr_folder_path,hr_folder_path,psnr_threshold)
def split_image(image):
    width, height = image.size
    split_width = width // 2
    split_height = height // 2
    split_images = []
    for i in range(2):
        for j in range(2):
            left = j * split_width
            upper = i * split_height
            right = left + split_width
            lower = upper + split_height
            box = (left, upper, right, lower)
            split_images.append(box)
    return split_images

def exam_complexity(lr_patch,hr_patch,depth,lr_folder_path,hr_folder_path,psnr_threshold):
    split_depth = depth
    complexity = calculate_texture_complexity(lr_patch,hr_patch)
    # global cp_list
    # cp_list.append(complexity)
    if complexity <= psnr_threshold and split_depth <= 1 :
        # global split_times
        # split_times += 1
        split_lr_img = split_image(lr_patch)
        split_hr_img = split_image(hr_patch)
        for i,j in zip(split_lr_img,split_hr_img):
            small_lr_patches = lr_patch.crop(i)
            small_hr_patches = hr_patch.crop(j)
            global lr_index
            small_lr_patches.save(lr_folder_path + '/' + str(lr_index) + '.png')
            small_hr_patches.save(hr_folder_path + '/' + str(lr_index) + '.png')
            lr_index += 1
    else:
        lr_patch.save(lr_folder_path + '/' + str(lr_index) + '.png')
        hr_patch.save(hr_folder_path + '/' + str(lr_index) + '.png')
        lr_index += 1


def calculate_texture_complexity(lr_patch,hr_patch):
    lr_patch = transforms.functional.to_tensor(lr_patch)
    hr_patch = transforms.functional.to_tensor(hr_patch)
    output = model(lr_patch)
    return get_patch_psnr(output,hr_patch)


def get_patch_psnr(sr, hr, shave=4):
    sr = sr.to(hr.dtype)
    sr = (sr * 255).round().clamp(0, 255) / 255
    diff = sr - hr
    if shave:
        diff = diff[..., shave:-shave, shave:-shave]
    mse = diff.pow(2).mean([-3, -2, -1])
    psnr = -10 * mse.log10()
    return psnr

if __name__ == "__main__":
    cur_path = args.source_path
    cur_path = os.path.join(cur_path, args.tt)
    load_path = './checkpoint_pretrained/epoch_30_X'+str(args.scale)+'.pth'
    model = load_model(load_path)
    lr_path = cur_path + '/DIV2K_train_LR_bicubic/' + 'X' + str(args.scale)
    hr_path = cur_path + '/DIV2K_train_HR'
    lr_path_folder = cur_path + '/DIV2K_train_LR_bicubic_chunk0/'+'X' + str(args.scale)
    hr_path_folder = cur_path + '/DIV2K_train_HR_chunk0/'+'X' + str(args.scale)
    gen_data(lr_path,hr_path,lr_path_folder,hr_path_folder)

    cur_path = args.source_path
    cur_path = os.path.join(cur_path, args.tt)
    lr_path = cur_path + '/DIV2K_train_LR_bicubic/' + 'X' + str(args.type)
    hr_path = cur_path + '/DIV2K_train_HR'

    image_path = './images/00001x2.png'
    image_hr_path = './images/00001_hr.png'
    folder_path_hr = './img_hr_folder'
    folder_path_lr = './img_lr_folder'
    img_patches(image_path,image_hr_path,size_w,size_h,folder_path_lr,folder_path_hr,args.psnr_threshold)




