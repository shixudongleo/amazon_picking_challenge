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
        self.bg_model = cv2.BackgroundSubtractorMOG()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    def get_fg_mask(self, img):
        fg_mask = self.bg_model.apply(img)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)

        return fg_mask

    def train_bg_model(self, img):
        self.bg_model.apply(img)

    def build_bg_model(self, bg_dir='bg_images'):
        bg_imgs = [os.path.join(bg_dir, file)
                   for file in os.listdir(bg_dir) if file.endswith('.png')]

        for bg_img in bg_imgs:
            img = cv2.imread(bg_img)
            self.train_bg_model(img)

        return self


from argparse import ArgumentParser

if __name__ == '__main__':
    """ usage: python background_subtractor.py -i test.png  """

    ap = ArgumentParser()
    ap.add_argument('-i', '--image', required=True,
                    help='the path to the test image')
    args = vars(ap.parse_args())
    test_img = args['image']
    test_img = cv2.imread(test_img)

    bg_model = BackgroundSubtractor()
    bg_model = bg_model.build_bg_model()

    fg_mask = bg_model.get_fg_mask(test_img)

    # display foreground image
    cv2.imshow('foreground', fg_mask)
    cv2.waitKey(0)
