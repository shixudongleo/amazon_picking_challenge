#!/usr/bin/env python

"""
    Test Object Classifier Accuracy
    Usage:
        run the script
"""
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from fg_segmentation.background_subtractor import BackgroundSubtractor
from color_hist_detector.rgbhistogram import RGBHistogram, HSVHistogram
from color_hist_detector.color_hist_classifier import Color_Hist_Classifier

__author__ = 'shixudongleo'
__date__ = '2015/05/15'

if __name__ == '__main__':
    # test accuracy of classifier
    feature_extractor = RGBHistogram()
    # feature_extractor = HSVHistogram()
    color_clf = Color_Hist_Classifier(feature_extractor)

    X, y = color_clf.prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    color_clf.train(X_train, y_train)

    y_pred = color_clf.predict(X_test)
    print classification_report(y_test, y_pred)
    print confusion_matrix(y_test, y_pred)
