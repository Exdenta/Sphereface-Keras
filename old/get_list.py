#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path
from tqdm import tqdm
import os


if __name__ == "__main__":
    dataset = Path.cwd() / 'datasets' / 'CASIA-WebFace-112x96'
    subfolders = os.listdir(dataset)

    # exclude the identities appearing in LFW dataset
    [subfolders.remove(x) for x in ['0166921', '1056413', '1193098']]

    # create the list for training
    list_path = Path.cwd() / 'datasets' / 'CASIA-WebFace-112x96.txt'
    with list_path.open(mode='w') as f:
        for i, folder in enumerate(tqdm(subfolders)):
            for image in os.listdir(dataset / folder):
                path = dataset / folder / image
                f.write("{} {}\n".format(path, i))
