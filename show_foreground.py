#!/usr/bin/env python

"""
    Foreground Segmentation
    Usage:
        give a picture with foreground, show foreground
"""
import os
import cv2
import numpy as np
from fg_segmentation.background_subtractor import BackgroundSubtractor
from argparse import ArgumentParser

__author__ = 'shixudongleo'
__date__ = '2015/05/15'

if __name__ == '__main__':
    """ usage: python background_subtractor.py -i test.png  """

    ap = ArgumentParser()
    ap.add_argument('-d', '--directory', required=True,
                    help='the path to the test image')
    args = vars(ap.parse_args())

    test_dir = args['directory']
    test_imgs = [os.path.join(test_dir, file)
                 for file in os.listdir(test_dir) if file.endswith('.png')]

    bg_model = BackgroundSubtractor()
    bg_model = bg_model.build_bg_model()

    for file in test_imgs:
        test_img = cv2.imread(file)
        fg_mask = bg_model.get_fg_mask(test_img)
        fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2RGB)

        display = np.hstack([test_img, fg_mask])
        # display foreground image
        cv2.imshow('foreground', display)
        cv2.waitKey(0)
