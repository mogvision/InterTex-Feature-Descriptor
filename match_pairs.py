
import os
import sys
import time
import cv2
from threading import Thread
from os.path import isfile, join
import numpy as np
from tempfile import TemporaryFile
from os import listdir

import string
import argparse
import subprocess
import time

from util.util import Matcher_NNDR, readkp

if __name__ == '__main__':
    curr_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='Feature Extraction by InterTex')

    parser.add_argument(
        '--input_pairs', type=str, default='image',
        help='Path to the images')

    parser.add_argument(
        '--max_keypoints', type=int, default=1e4,
        help='Maximum number of keypoints detected by FFD (feature detector)'
             ' (\'-1\' keeps all keypoints)')

    parser.add_argument(
        '--num_level', type=int, default=3,
        help='Number of decomposition levels')

    parser.add_argument(
        '--contrast_threshold', type=float, default=0.05,
        help='FFD\'s contrast threshold')

    parser.add_argument(
        '--curvature_ratio', type=float, default=10.,
        help='FFD\'s curvature ratio')

    parser.add_argument(
        '--upright', type=bool, default=True,
        help='Upright')

    opt = parser.parse_args()
    print(opt)


    KPTS_DES = []
    IMGs = []

    image_formats = [".jpg", ".png", ".ppm", ".pgm"]
    for image_name in os.listdir(opt.input_pairs):
        ext = os.path.splitext(image_name)[1]
        if ext.lower() in image_formats:
            image_dir  = os.path.join(curr_dir, opt.input_pairs, image_name)
            store_dir  = os.path.join(curr_dir, opt.input_pairs)
            IMGs.append(image_dir)
            process = subprocess.Popen('./InterTexFFD '+ str(os.path.join(curr_dir, opt.input_pairs, image_name)) + ' ' \
                + str(store_dir) + ' ' \
                + str(opt.num_level) + ' ' \
                + str(opt.max_keypoints) + ' ' \
                + str(opt.contrast_threshold) + ' ' \
                + str(opt.curvature_ratio) + ' '\
                + str(1*opt.upright), 
                shell=True,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            keypoints, des, num_kp = readkp(os.path.join(store_dir, 'InterTex_FFD_'+image_name+'.txt'))
            KPTS_DES.append([keypoints, des])

    Matcher_NNDR(IMGs, KPTS_DES)

