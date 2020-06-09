#!/usr/bin/env python3
# coding: utf-8

from scipy import spatial
from pathlib import Path
import multiprocessing
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import random
import math
import cv2
import os


def process(person_index, data_list):
    # find all elements from folder 'i', they all belong to the same person
    person_list = [x for x in data_list if x[1] == str(person_index)]
    combinations = list(itertools.combinations(person_list, 2))
    pairs = []
    for combination in combinations:
        pair = {}
        pair['fileL'] = combination[0][0]
        pair['fileR'] = combination[1][0]
        pair['flag'] = 1  # same person
        pairs.append(pair)

    # add the same amount of combinations for 2 different persons
    others_list = [x for x in data_list if x[1] != str(person_index)]
    for _ in combinations:
        pair = {}
        pair['fileL'] = random.choice(person_list)[0]
        pair['fileR'] = random.choice(others_list)[0]
        pair['flag'] = 0
        pairs.append(pair)
    return pairs


def form_train_data(dataset_path):
    # train dataset paths
    casia_dataset_folder = dataset_path / 'CASIA-WebFace-112x96'
    casia_list_path = dataset_path / 'CASIA-WebFace-112x96.txt'

    print("Form CASIA-Webface image list")
    casia_dataset_subfolders = os.listdir(casia_dataset_folder)

    # exclude the identities appearing in LFW dataset
    [casia_dataset_subfolders.remove(x)
        for x in ['0166921', '1056413', '1193098']]

    # create the list for training
    with casia_list_path.open(mode='w') as f:
        for i, subfolder in enumerate(casia_dataset_subfolders):
            for image_name in os.listdir(casia_dataset_folder / subfolder):
                image_path = os.path.join(subfolder, image_name)
                f.write("{} {}\n".format(image_path, i))

    print("Load CASIA-Webface image list")
    casia_list = []
    with casia_list_path.open(mode='r') as f:
        casia_list = list(map(lambda x: x.split(), f.readlines()))

    # count a number of unique persons
    person_count = len(set(map(lambda x: x[1], casia_list)))
    pairs = []

    print("Form train dataset")
    pool = multiprocessing.Pool(processes=6)
    pbar = tqdm(total=person_count)

    def update(*a):
        pbar.update()

    async_requests = [pool.apply_async(process, args=(
        i, casia_list,), callback=update) for i in range(0, person_count)]
    pool.close()
    pool.join()
    [pairs.extend(request.get()) for request in async_requests]

    # pairs[0]: {'fileL': '0000045\\001.jpg', 'fileR': '0000045\\002.jpg', 'flag': 1}
    return pairs


def form_test_data(dataset_path):
    lfw_dataset_folder = dataset_path / 'lfw_112x96'
    lfw_list_path = dataset_path / 'lfw_pairs.txt'

    lfw_list = []
    with lfw_list_path.open(mode='r') as f:
        lfw_list = list(map(lambda x: x.split(), f.readlines()))

    pairs = []
    for i, list_tuple in enumerate(tqdm(lfw_list)):
        if i < 1:
            continue
        pair = {}
        if len(list_tuple) == 3:
            # recreating relative path + file name
            pair['fileL'] = os.path.join(
                list_tuple[0], (list_tuple[0] + "_" + list_tuple[1].rjust(4, '0') + ".jpg"))
            pair['fileR'] = os.path.join(
                list_tuple[0], (list_tuple[0] + "_" + list_tuple[2].rjust(4, '0') + ".jpg"))
            pair['flag'] = 1
        elif len(list_tuple) == 4:
            pair['fileL'] = os.path.join(
                list_tuple[0], (list_tuple[0] + "_" + list_tuple[1].rjust(4, '0') + ".jpg"))
            pair['fileR'] = os.path.join(
                list_tuple[2], (list_tuple[2] + "_" + list_tuple[3].rjust(4, '0') + ".jpg"))
            pair['flag'] = 0

        if os.path.exists(os.path.join(str(lfw_dataset_folder), pair['fileL'])) and os.path.exists(os.path.join(str(lfw_dataset_folder), pair['fileR'])):
            pairs.append(pair)
    return pairs


def load_data(dataset_path):
    # training data
    casia_pairs_path = dataset_path / 'casia_pairs.npz'
    err = False

    if casia_pairs_path.exists():
        print("Loading train data")
        casia_pairs = dict(np.load(str(casia_pairs_path), allow_pickle=True))
        casia_pairs = casia_pairs['arr_0']
    else:
        print("{} does not exist!".format(str(casia_pairs_path)))
        err = True
        casia_pairs = []

    # test data
    lfw_pairs_path = dataset_path / 'lfw_pairs.npz'
    if lfw_pairs_path.exists():
        print("Loading test data")
        lfw_pairs = dict(np.load(str(lfw_pairs_path), allow_pickle=True))
        lfw_pairs = lfw_pairs['arr_0']
    else:
        print("{} does not exist!".format(str(lfw_pairs_path)))
        err = True
        lfw_pairs = []

    if err:
        print(
            "You need to generate data first. Run generate_data(...) from form_train_data.")

    return casia_pairs, lfw_pairs


def generate_data(project_dir):

    dataset_path = project_dir / 'datasets'

    # training data
    casia_pairs_path = dataset_path / 'casia_pairs.npz'
    print("\nForming train data")
    casia_pairs = form_train_data(dataset_path)
    np.savez_compressed(str(casia_pairs_path), casia_pairs)  # ~ 300 Mb

    # test data
    lfw_pairs_path = dataset_path / 'lfw_pairs.npz'
    print("\nForming test data")
    lfw_pairs = form_test_data(dataset_path)
    np.savez_compressed(str(lfw_pairs_path), lfw_pairs)


if __name__ == "__main__":
    project_dir = Path(__file__).parent.parent
    generate_data(project_dir)
