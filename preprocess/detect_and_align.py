#!/usr/bin/env python3
# coding: utf-8

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress Tensorflow verbose prints
warnings.simplefilter(action='ignore', category=FutureWarning)

from mtcnn import MTCNN
import multiprocessing
from tqdm import tqdm
import numpy as np
import argparse
import pathlib
import sys
import cv2

process_count = 16
image_size = (96, 112)

# shifts the rectangle coords so eyes are in the center of it (not being used for now)
def shift(keypoints, x1, x2):
    delta = (x2 + x1 - keypoints['right_eye']
             [0] - keypoints['left_eye'][0]) // 2
    x1 = (x1 - delta) if x1 > delta else 0
    x2 = (x2 - delta) if x2 > delta else 0
    return x1, x2

# simple alignment by eyes
def align(image, keypoints):
    def rotate(image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    angle = np.degrees(np.arctan(abs((keypoints['left_eye'][1] - keypoints['right_eye'][1]) / (
        keypoints['left_eye'][0] - keypoints['right_eye'][0]))))
    if keypoints['left_eye'][1] > keypoints['right_eye'][1]:
        angle *= -1
    image = rotate(image, angle)
    return image

# returns the biggest face keypoints and 2 points or None
def detect_face(detector, image):
    faces = detector.detect_faces(image)
    if len(faces) == 0:
        return None
    elif len(faces) > 1:
        # choose the biggest one
        widths = list(map(lambda x: x['box'][2], faces))
        biggest_face_index = widths.index(max(widths))
        face = faces[biggest_face_index]
    else:
        face = faces[0]

    x1, y1, w, h = face['box'][0:4]
    # sometimes detector returns negative numbers
    x1, y1 = [value if value > 0 else 0 for value in [x1, y1]]

    # shift face
    x1, x2 = shift(face['keypoints'], x1, x1 + w)
    return face['keypoints'], (x1, y1), (x2, y1 + h)


def process(subdirectory, save_dir, dataset_path):
    detector = MTCNN()
    # in case subdirectory already exists
    if not (save_dir / subdirectory).exists():
        (save_dir / subdirectory).mkdir()

    images_directory = dataset_path / subdirectory
    for image_name in os.listdir(images_directory):
        image_path = images_directory / image_name
        image = cv2.imread(str(image_path))
        result = detect_face(detector, image)
        if result == None:
            continue
        keypoints, (x1, y1), (x2, y2) = result
        image = align(image, keypoints)
        image = image[y1:y2, x1:x2]
        image = cv2.resize(image, image_size, cv2.INTER_AREA)
        cv2.imwrite(str(save_dir / subdirectory / image_name), image)


def parse_arguments():
    parser = argparse.ArgumentParser('Detect, align and save faces')
    parser.add_argument('--source_dir', help='directory with original images')
    parser.add_argument('--save_dir', help='directory to save images with aligned faces')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    source_dir = pathlib.Path(args.source_dir)
    save_dir = pathlib.Path(args.save_dir)
    print("Processing {} dataset".format(source_dir))
    print("Aligned faces will be saves to {}".format(save_dir))

    if not source_dir.exists():
        exit("Path {} doesn't exist!".format(source_dir))
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # each subdirectory in lwf dataset in a specific person photo collection
    subdirectories = os.listdir(source_dir)

    print("Detect and align faces:\n")
    pool = multiprocessing.Pool(processes=process_count)
    pbar = tqdm(total=len(subdirectories))

    def update(*a):
        pbar.update()
    async_requests = [pool.apply_async(process, args=(
        subdirectory, save_dir, source_dir), callback=update) for subdirectory in subdirectories]
    pool.close()
    pool.join()
