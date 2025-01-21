## inference dependence
from inference_track_pred import inference
import argparse
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import logging
import os
import random 
import skimage.transform as st
from skvideo.io import vwrite
import gdown
import torch.nn as nn
import torchvision
import collections
import pickle
from torch.nn import functional as F
from torchvision.datasets.utils import download_url
import imageio
from scipy.spatial.transform import Rotation 
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from collections import OrderedDict
from PIL import Image
import cv2 
import skimage.transform as st
# from models import DiT_models
from single_script import DiT_models as DiT_models_track
from matplotlib import cm
import matplotlib.pyplot as plt

## co-tracker dependence
import imageio.v3 as iio
import imageio


###################################################################################################
#                                                                                                 #
#                                              co-tracker                                         #
#                                                                                                 #
###################################################################################################

def cotracker(path, x_origin, y_origin):

    init_path = "/gs/fs/tga-artt551/chen/report/ground-truth/init_test.jpg"
    goal_path = "/gs/fs/tga-artt551/chen/report/ground-truth/goal_test.jpg"

    frames = np.asarray(iio.imread(path, plugin="FFMPEG"))  # plugin="pyav"
    print(frames.shape)
    # Only take 8 frames
    frames = frames[:32, ...]
    # Get the first frame and the last frame in 8 frames
    first_frame, last_frame = frames[0], frames[-1]
    imageio.imwrite(init_path, first_frame)
    imageio.imwrite(goal_path, last_frame)

    device = 'cuda'

    grid_size = 20
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

    # Run Offline CoTracker:
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1

    
    # visualization
    from cotracker.utils.visualizer import Visualizer
    vis = Visualizer(save_dir="/gs/fs/tga-artt551/chen/report/ground-truth", pad_value=120, linewidth=3)
    vis.visualize(video, pred_tracks, pred_visibility)

    return pred_tracks, init_path, goal_path


###################################################################################################
#                                                                                                 #
#                                             inference                                           #
#                                                                                                 #
###################################################################################################

def get_result_from_inference(init_frame_path, goal_frame_path, new_W, new_H):
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_points", type=int, default=25)
    parser.add_argument("--visualize", type=bool, default=True,help="whether to visualize predicted tracks")
    parser.add_argument("--model", type=str, choices=list(DiT_models_track.keys()), default="DiT-L/2-NoPosEmb")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--init", type=str, default=init_frame_path,
                        help="path to initial image")
    parser.add_argument("--goal", type=str, default=goal_frame_path,
                        help="path to goal image")
    parser.add_argument("--ckpt", type=str, default='/gs/fs/tga-artt551/chen/trained-model-parameters/track_pred_L_2_new_best.pt',
                        help="path to trained checkpoint")
    parser.add_argument("--new_W", type=int, default=new_W, help="new_video_W")
    parser.add_argument("--new_H", type=int, default=new_H, help="new_video_H")

    args = parser.parse_args()
    return inference(args)


def get_delta(res_inf_time_step, res_cot_time_step, x):
    # res_inf_time_step.shape = [1, 400, 2]; res_cot_time_step.shape = [1, 400, 2]

    num_points = res_inf_time_step.shape[2]
    num_points_correct = 0

    # Iterate out all the points in inference result, and compare them with the corresponding points in cotracker result
    for i in range(num_points):
        if np.linalg.norm(res_inf_time_step[0, i, :] - res_cot_time_step[0, i, :]) < x:

            num_points_correct += 1
        #if i % 100 == 0:
            #print(f'inf: {res_inf_time_step[0, i, :]}')
            #print(f'cot: {res_cot_time_step[0, i, :]}')
    return num_points_correct / num_points


def score(res_inf, res_cot, H=8, N=10):
    # res_inf.shape = [1, 8, 400, 2]; raw_res_cot = [1, 8, 400, 2]

    score_total = 0
    for h in range(H):
        for x in range(1, N + 1):
            score_total += get_delta(res_inf[:, h, ...], res_cot[:, h, ...], x)

    return score_total / (H * N)




if __name__ == "__main__":
    
    video_path = '/gs/fs/tga-artt551/chen/my-dataset/robot.mp4'

    H, W, _ = iio.imread(video_path, plugin="FFMPEG")[0].shape

    result_from_cotracker, init_path, goal_path = cotracker(video_path,x_origin=W, y_origin=H)
    result_from_cotracker = result_from_cotracker.to('cpu').numpy()

    # Get the result from inference
    result_from_inference = inference_result = get_result_from_inference(init_path, goal_path, new_H=H, new_W=W)

    """
    # Check the points
    points1 = result_from_cotracker[0, -1, :]
    points2 = result_from_inference[0, -1, :]

    plt.scatter(points1[:, 0], points1[:, 1], color='red', label='Group 1')
    plt.scatter(points2[:, 0], points2[:, 1], color='blue', label='Group 2')

    plt.title("Two Groups of Points")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()

    plt.show()
    plt.savefig("output.png") 
    """

    score = score(result_from_inference, result_from_cotracker)
    print(score)

