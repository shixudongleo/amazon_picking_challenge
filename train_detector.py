#!/usr/bin/env python

"""
    Training Object Classifier
    Usage:
        put data in desired folder and run the script
"""

from color_hist_detector.rgbhistogram import RGBHistogram, HSVHistogram
from color_hist_detector.color_hist_classifier import Color_Hist_Classifier

__author__ = 'shixudongleo'
__date__ = '2015/05/15'

if __name__ == '__main__':
    feature_extractor = RGBHistogram()
    color_clf = Color_Hist_Classifier(feature_extractor)

    if color_clf.is_model_trained():
        # load model
        color_clf.load_model()
    else:
        # train model
        X, y = color_clf.prepare_data()
        color_clf.train(X, y)
        color_clf.save_model()
