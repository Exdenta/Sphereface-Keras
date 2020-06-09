#!/usr/bin/env python3
# coding: utf-8

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress Tensorflow verbose prints

from mtcnn import MTCNN

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def process(self, image):
        ''' Detects 1 face (the biggest on image)
            returns facial keypoints and rectangle coordinates
        '''
        faces = self.detector.detect_faces(image)
        if len(faces) == 0:
            return None

        # choose the biggest face on image
        elif len(faces) > 1:
            width_list = list(map(lambda x: x['box'][2], faces))
            biggest_face_index = width_list.index(max(width_list))
            face = faces[biggest_face_index]
        else:
            face = faces[0]

        x1, y1, w, h = face['box'][0:4]
        # to prevent negative numbers
        x1, y1 = [value if value > 0 else 0 for value in [x1, y1]]
        return face['keypoints'], (x1, y1), (x1 + w, y1 + h)
