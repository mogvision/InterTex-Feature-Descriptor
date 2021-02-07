
import os
import sys
import cv2
from threading import Thread
from os.path import isfile, join
import numpy as np
from tempfile import TemporaryFile
from os import listdir

import string
import argparse
import subprocess

# FFD cv2Wrapper
import prcv

from util.util import Matcher_NNDR2

if __name__ == '__main__':
    curr_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='Feature Detection by FFD')

    parser.add_argument(
        '--NUM_show_matches', type=int, default=100,
        help='Number of matches shown in the output')

    parser.add_argument(
        '--input_pairs', type=str, default='image',
        help='Path to the images')

    parser.add_argument(
        '--max_keypoints', type=int, default=2500,
        help='Maximum number of keypoints detected by FFD'
             ' (\'-1\' keeps all keypoints)')

    parser.add_argument(
        '--num_level', type=int, default=3,
        help='Number of decomposition levels')

    parser.add_argument(
        '--contrast_threshold', type=float, default=0.03,
        help='FFD\'s contrast threshold')

    parser.add_argument(
        '--curvature_ratio', type=float, default=10.,
        help='FFD\'s curvature ratio')

    parser.add_argument(
        '--Upright', type=bool, default=True,
        help='InterTex: Upright?')

    parser.add_argument(
        '--time_cost', type=int, default=0,
        help='Report running time over 25 runs'
        ' (\'-1\' doesn\' report time')

    opt = parser.parse_args()
    print(opt)

    FFD_InterTex = prcv.prcv_InterTexFFD(opt.num_level, opt.contrast_threshold, opt.curvature_ratio) 


    KPTS_DES = []
    IMGs = []

    image_formats = [".jpg", ".png", ".ppm", ".pgm"]
    for image_name in os.listdir(opt.input_pairs):
        ext = os.path.splitext(image_name)[1]
        if ext.lower() in image_formats:
            image_dir  = os.path.join(curr_dir, opt.input_pairs, image_name)
            im = cv2.imread(image_dir)
            IMGs.append(im)

            keypoints, des = FFD_InterTex.detectAndCompute(im, opt.Upright, opt.max_keypoints, opt.time_cost)

            KPTS_DES.append([keypoints, des])

    Matcher_NNDR2(IMGs, KPTS_DES)


