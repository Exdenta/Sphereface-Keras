#!/usr/bin/env python3
# coding: utf-8

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['GLOG_minloglevel'] = '2'  # suppress Caffe verbose prints

from datetime import datetime
from scipy import spatial
from pathlib import Path
import multiprocessing
from tqdm import tqdm
import numpy as np
import caffe
import time
import math
import cv2
import sys

current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
project_path = current_dir.parent

process_count = 6
flip_image = False # will add embeddings of the flipped image also
model_path = project_path / 'models' / 'sphereface_20_caffe' / 'sphereface_deploy.prototxt'
weights_path = project_path / 'models' / 'sphereface_20_caffe' / 'sphereface_model.caffemodel'
dataset_path = project_path / 'test' / 'data'
list_path = dataset_path / 'pairs.txt'
folder_path = dataset_path / 'lfw_112X96'

caffe.set_mode_cpu()

def form_features(pair):
    def get_embeddings(net, image):
        image_ = (image - 127.5) / 128
        image_ = np.asarray(image_)
        image_ = np.transpose(image_, (2, 0, 1))
        image_ = image_[None, :]
        net.blobs['data'].data[...] = image_
        out = net.forward()
        return out['fc5'][0]
    
    # init model
    net = caffe.Net(str(model_path), str(weights_path), caffe.TEST)

    # featureL
    imageL = cv2.imread(str(pair['fileL']))
    embsL = get_embeddings(net, imageL)
    if flip_image:
        imageL_flipped = cv2.flip(imageL, 1)
        embsL_flipped = get_embeddings(net, imageL_flipped)
        embsL = np.concatenate((embsL, embsL_flipped))
    pair['featureL'] = embsL.copy()

    # featureR
    imageR = cv2.imread(str(pair['fileR']))
    embsR = get_embeddings(net, imageR)
    if flip_image:
        imageR_flipped = cv2.flip(imageR, 1)
        embsR_flipped = get_embeddings(net, imageR_flipped)
        embsR = np.concatenate((embsR, embsR_flipped))
    pair['featureR'] = embsR.copy()

    # cosine distance
    pair['score'] = 1 - spatial.distance.cosine(pair['featureL'], pair['featureR'])
    
    return pair


def create_pairs():
    """ Creates list of pairs from pairs.txt and saves them to pairs_*.npy
        Each pair is a map object with fields:
            fileL    - path to the first image
            fileR    - path to the second image
            flag     - "1" if first and second images are of the same person, "-1" otherwise
            fold     - value from 1 to 10, denotes section of the lfw dataset
            featureL - feature vector of the first image (model output, 512 features)
            featureR - feature vector of the first image (model output, 512 features)
            score    - cosine distance between featureL and featureR
    """

    # check paths do exist
    paths = [list_path, folder_path]
    for path in paths:
        if not path.exists():
            exit("Path {} doesn't exist!".format(path))

    # get features
    pairs = []
    with open(str(list_path), 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if i < 1:
            continue
        pair = {}
        l = line.split()
        if len(l) == 3:
            # recreating relative path + file name
            pair['fileL'] = folder_path / l[0] / (l[0] +
                                                  "_" + l[1].rjust(4, '0') + ".jpg")
            pair['fileR'] = folder_path / l[0] / (l[0] +
                                                  "_" + l[2].rjust(4, '0') + ".jpg")
            pair['flag'] = 1
        elif len(l) == 4:
            pair['fileL'] = folder_path / l[0] / (l[0] +
                                                  "_" + l[1].rjust(4, '0') + ".jpg")
            pair['fileR'] = folder_path / l[2] / (l[2] +
                                                  "_" + l[3].rjust(4, '0') + ".jpg")
            pair['flag'] = -1
        pair['fold'] = math.ceil(i / 600)
        if pair['fileL'].exists() and pair['fileR'].exists():
            pairs.append(pair)

    print("Process images:")
    pool = multiprocessing.Pool(processes=process_count)
    pbar = tqdm(total=len(pairs))

    def update(*a):
        pbar.update()

    async_requests = [pool.apply_async(form_features, args=(pair,), callback=update) for pair in pairs]
    pool.close()
    pool.join()
    pairs = [request.get() for request in async_requests]

    return pairs


def get_accuracy(pairs, threshold):
    TP = sum(map(lambda x: x['score'] > threshold and x['flag'] == 1, pairs))
    TN = sum(map(lambda x: x['score'] < threshold and x['flag'] != 1, pairs))
    return (TP + TN) / len(pairs)


if __name__ == "__main__":
    # process images from pairs.txt
    pairs = create_pairs()

    # save for future testing
    pairs_path = current_dir / 'data' / 'pairs.npy'
    np.save(str(pairs_path), pairs)
    
    # load saved file
    pairs = np.load(str(pairs_path), allow_pickle=True)

    print("\nAccuracy:")
    ACCs = np.zeros(10)
    Thrs = np.zeros(10)
    for j in np.arange(0, 10):
        validation_pairs = list(filter(lambda x: x['fold'] != j + 1, pairs))
        test_pairs = list(filter(lambda x: x['fold'] == j + 1, pairs))

        thrNum = 1000
        accuracys = np.zeros(2*thrNum)
        thresholds = np.arange(-thrNum, thrNum) / thrNum
        for i in np.arange(0, 2*thrNum):
            accuracys[i] = get_accuracy(validation_pairs, thresholds[i])

        threshold = np.mean(thresholds[np.where(accuracys == max(accuracys))])
        ACCs[j] = get_accuracy(test_pairs, threshold)
        Thrs[j] = threshold
        print("fold: ", j + 1, ", accuracy: ", ACCs[j] * 100, ", threshold: ", threshold)

    print('----------------')
    print('AVE: ', np.mean(ACCs)*100)
    print('Threshold: ', np.mean(Thrs))
