#!/usr/bin/env python

"""
    Background Modeling code
    Usage:
        train_bg_model with images
        get_fg_mask
"""

import os
import cv2

__author__ = 'shixudongleo'
__date__ = '2015/05/02'


class BackgroundSubtractor:
    def __init__(self):
        path, _ = os.path.split(os.path.realpath(__file__))
        self.bg_dir = os.path.join(path, 'bg_images')
        self.bg_model = cv2.BackgroundSubtractorMOG()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    def get_fg_mask(self, img):
        fg_mask = self.bg_model.apply(img)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)

        return fg_mask

    def train_bg_model(self, img):
        self.bg_model.apply(img)

    def build_bg_model(self):
        bg_imgs = [os.path.join(self.bg_dir, file)
                   for file in os.listdir(self.bg_dir) if file.endswith('.png')]

        for bg_img in bg_imgs:
            img = cv2.imread(bg_img)
            self.train_bg_model(img)

        return self
