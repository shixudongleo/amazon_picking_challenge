#!/usr/bin/env python

"""
    Object detection code
    Input: rgb image and depth image
    Output: object class name, bounding box, depth of the object
 """

import cv2
import numpy as np
from fg_segmentation.background_subtractor import BackgroundSubtractor
from color_hist_detector.color_hist_classifier import Color_Hist_Classifier
from color_hist_detector.rgbhistogram import RGBHistogram, HSVHistogram


__author__ = 'shixudongleo'
__date__ = '2015/05/02'


def get_center_depth(depth_img, contour):
    M = cv2.moments(contour)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])

    padding = 3
    sub_mat = depth_img[center_y-padding: center_y+padding,
                    center_x-padding: center_x+padding]

    depth = np.median(sub_mat)

    return depth

def detect_objects_by_contour(img, min_size=100, max_size=1000):
    # contour detection
    # return contour and mask tutples
    (contours, _) = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

    height = img.shape[0]
    width = img.shape[1]

    bbox_mask_list = []
    for i in range(len(contours)):
        contour = contours[i]

        if len(contour) < min_size or len(contour) > max_size:
            continue

        mask = np.zeros((height, width), dtype='uint8')
        cv2.drawContours(mask, contours, i, [255, 255, 255], -1)

        (x, y, w, h) = cv2.boundingRect(contour)
        bbox = (y, x, h, w)
        bbox_mask_list.append((contour, bbox, mask))

    return bbox_mask_list


def detect(rgb_img, depth_img, bg_model, color_clf):
    fg_mask = bg_model.get_fg_mask(rgb_img)
    bbox_mask_list = detect_objects_by_contour(fg_mask, min_size=30)

    objects = []
    for (contour, bbox,  mask) in bbox_mask_list:
        label = color_clf.predict_image_with_fg_mask(rgb_img, mask)

        object = {}
        object['label'] = label
        object['bbox'] = bbox
        object['depth'] = get_center_depth(depth_img, contour)
        objects.append(object)

    return objects

from argparse import ArgumentParser

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-i', '--image', required=True,
                    help='path to image file')
    ap.add_argument('-d', '--depth', required=True,
                    help='path to depth image file')
    args = vars(ap.parse_args())

    # build  bg model
    bg_images = '/Users/shixudongleo/Projects/RoboticsVision/vision_4_amazon_challenge/fg_segmentation/bg_images'
    bg_model = BackgroundSubtractor().build_bg_model(bg_dir=bg_images)

    model_file = '/Users/shixudongleo/Projects/RoboticsVision/vision_4_amazon_challenge/color_hist_detector/rgb_hist_model.txt'
    feature_extractor = RGBHistogram()
    color_clf = Color_Hist_Classifier(feature_extractor)
    color_clf = color_clf.load_model(model_file=model_file)

    # read test_rgb.png and test_depth.txt
    rgb_file = args['image']
    test_rgb = cv2.imread(rgb_file)
    depth_file = args['depth']
    test_depth = np.load(depth_file)

    # using detect function to return class label, bbox, depth
    detect_objects = detect(test_rgb, test_depth, bg_model, color_clf)

    for obj in detect_objects:
        name = obj['label']
        (y, x, h, w) = obj['bbox']
        depth = obj['depth']
        # draw the class label
        cv2.putText(test_rgb, "#{}".format(name), (x, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # draw the bounding box
        cv2.rectangle(test_rgb, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # draw depth
        cv2.putText(test_rgb, "#{}".format(depth), (x, y-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('test', test_rgb)
        cv2.waitKey(0)
