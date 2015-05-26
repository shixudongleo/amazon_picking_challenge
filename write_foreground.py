#!/usr/bin/env python

"""
    Foreground Segmentation
    Usage:
        give the background folder, object folder and output folder
"""
import os
import cv2
from fg_segmentation.background_subtractor import BackgroundSubtractor
from argparse import ArgumentParser

__author__ = 'shixudongleo'
__date__ = '2015/05/15'


def write_mask(bg_model, in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    files = [file for file in os.listdir(in_dir) if file.endswith('.png')]

    for file in files:
        test_img = cv2.imread(os.path.join(in_dir, file))
        fg_mask = bg_model.get_fg_mask(test_img)

        cv2.imwrite(os.path.join(out_dir, file), fg_mask)


if __name__ == '__main__':
    """ usage: python background_subtractor.py -i test.png  """

    ap = ArgumentParser()
    ap.add_argument('-b', '--background', required=False, nargs='?',
                    default='fg_segmentation/bg_images',
                    help='path to the background dir')
    ap.add_argument('-i', '--in_dir', required=False, nargs='?',
                    default='color_hist_detector/objects_training_data',
                    help='the path to the objects root folder')
    ap.add_argument('-o', '--out_dir', required=False, nargs='?',
                    default='color_hist_detector/objects_training_data_fg',
                    help='the path to the foreground mask root folder')
    args = vars(ap.parse_args())

    bg_dir = args['background']
    bg_model = BackgroundSubtractor(bg_dir)
    bg_model = bg_model.build_bg_model()

    objects_dir = args['in_dir']
    fg_dir = args['out_dir']

    obj_dirs = [dir for dir in os.listdir(objects_dir)
                if os.path.isdir(os.path.join(objects_dir, dir))]

    for dir in obj_dirs:
        in_dir = os.path.join(objects_dir, dir)
        out_dir = os.path.join(fg_dir, dir)

        write_mask(bg_model, in_dir, out_dir)
