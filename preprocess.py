#!/usr/bin/env python3
# coding: utf-8

from alignment import FaceAligner
from detection import FaceDetector
import multiprocessing
from tqdm import tqdm
import pathlib
import cv2
import os


def process_directory(subdirectory, save_dir, source_dir):
    # in case directories do not exist
    if not source_dir.exists():
        exit("Path {} doesn't exist!".format(source_dir))
    if not (save_dir / subdirectory).exists():
        (save_dir / subdirectory).mkdir(parents=True)

    face_detector = FaceDetector()
    face_aligner = FaceAligner()

    images_directory = source_dir / subdirectory
    save_images_directory = save_dir / subdirectory
    for image_name in os.listdir(images_directory):
        image_path = source_dir / subdirectory / image_name
        image = cv2.imread(str(image_path))
        det = face_detector.process(image)
        if det == None:
            continue
        keypoints, _, _ = det
        image = face_aligner.process_image(image, keypoints)
        det = face_detector.process(image)
        if det == None:
            continue
        keypoints, (x1, y1), (x2, y2) = det
        image = image[y1:y2, x1:x2]
        image = cv2.resize(image, image_size, cv2.INTER_AREA)
        if not save_images_directory.exists():
            save_images_directory.mkdir(parents=True)
        cv2.imwrite(str(save_images_directory / image_name), image)


def process_dataset(source_dir, save_dir, image_size=(96, 112), process_count=6):
    print("Detect and align faces:\n")
    subdirectories = os.listdir(source_dir)
    pool = multiprocessing.Pool(processes=process_count)
    pbar = tqdm(total=len(subdirectories))

    def update(*a):
        pbar.update()
    async_requests = [pool.apply_async(process_directory, args=(
        subdirectory, save_dir, source_dir), callback=update) for subdirectory in subdirectories]
    pool.close()
    pool.join()


datasets_path = pathlib.Path.cwd() / 'datasets'
casia_path = datasets_path / 'CASIA-WebFace'
lwf_path = datasets_path / 'lfw'
processed_casia_path = datasets_path / 'CASIA-WebFace-112x96'
processed_lwf_path = datasets_path / 'lfw-112x96'
image_size = (96, 112)

print("process test data")
process_dataset(lwf_path, processed_lwf_path,
                image_size=(96, 112), process_count=5)
print("process train data")
process_dataset(casia_path, processed_casia_path,
                image_size=(96, 112), process_count=5)
