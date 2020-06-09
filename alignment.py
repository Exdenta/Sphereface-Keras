#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import cv2


class FaceAligner:
    def process_image(self, image, keypoints):
        M = self.__get_rotation_matrix(keypoints)
        height, width = image.shape[:2]
        aligned = cv2.warpAffine(
            image, M, (width, height), flags=cv2.INTER_CUBIC)
        return aligned

    def __angle_between_2_points(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        tan = (y2 - y1) / (x2 - x1)
        return np.degrees(np.arctan(tan))

    def __get_rotation_matrix(self, keypoints):
        p1 = keypoints['left_eye']
        p2 = keypoints['right_eye']
        angle = self.__angle_between_2_points(p1, p2)
        x1, y1 = p1
        x2, y2 = p2
        xc = (x1 + x2) // 2
        yc = (y1 + y2) // 2
        M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
        return M
