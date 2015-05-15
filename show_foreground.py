#!/usr/bin/env python

"""
    Foreground Segmentation
    Usage:
        give a picture with foreground, show foreground
"""

import cv2
from fg_segmentation.background_subtractor import BackgroundSubtractor
from argparse import ArgumentParser

__author__ = 'shixudongleo'
__date__ = '2015/05/15'

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
