#!/usr/bin/env python

"""
    Object detection code
    Input: rgb image and depth image
    Output: object class name, bounding box, depth of the object
 """

from fg_segmentation.background_subtractor import BackgroundSubtractor


__author__ = 'shixudongleo'
__date__ = '2015/05/02'


def detect(rgb_img, depth_img, bg_model):
    result = {}
    result['label'] = ''
    result['bbox'] = (0, 0, 0, 0)
    result['depth'] = 0
    return result

if __name__ == '__main__':
    # build  bg model
    bg_model = BackgroundSubtractor().build_bg_model()

    # read test_rgb.png and test_depth.txt
    rgb_file = 'test_rgb.png'
    test_rgb = None
    depth_file = 'test_depth.txt'
    test_depth = None

    # using detect function to return class label, bbox, depth
    detect_result = detect(test_rgb, test_depth)

    # draw the class label

    # draw the bounding box

    # draw the depth distance
