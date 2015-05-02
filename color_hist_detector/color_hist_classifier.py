#!/usr/bin/env python

"""
    Color Histogram for Object Detection
    train classifier and predict label
"""

import os
import cv2
import cPickle
import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV


__author__ = 'shixudongleo'
__date__ = '2015/05/02'


class Color_Hist_Classifier:
    def __init__(self, feature_extractor):
        self.param_grid = {'C': [10**x for x in range(2, 5)],
                           'gamma': [10**x for x in range(-4, 0)]}
        self.clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto',
                                    probability=True), self.param_grid)

        self.name_label_table = {'crayola_64_ct': 1,
                                 'elmers_washable_no_run_school_glue': 2,
                                 'expo_dry_erase_board_eraser': 3,
                                 'feline_greenies_dental_treats': 4,
                                 'first_years_take_and_toss_straw_cups': 5,
                                 'highland_6539_self_stick_notes': 6,
                                 'kong_air_dog_squeakair_tennis_ball': 7,
                                 'kong_duck_dog_toy': 8,
                                 'kong_sitting_frog_dog_toy': 9,
                                 'kygen_squeakin_eggs_plush_puppies': 10,
                                 'mommys_helper_outlet_plugs': 11,
                                 'munchkin_white_hot_duck_bath_toy': 12,
                                 'oreo_mega_stuf': 13}

        self.label_name_table = {1: 'crayola_64_ct',
                                 2: 'elmers_washable_no_run_school_glue',
                                 3: 'expo_dry_erase_board_eraser',
                                 4: 'feline_greenies_dental_treats',
                                 5: 'first_years_take_and_toss_straw_cups',
                                 6: 'highland_6539_self_stick_notes',
                                 7: 'kong_air_dog_squeakair_tennis_ball',
                                 8: 'kong_duck_dog_toy',
                                 9: 'kong_sitting_frog_dog_toy',
                                 10: 'kygen_squeakin_eggs_plush_puppies',
                                 11: 'mommys_helper_outlet_plugs',
                                 12: 'munchkin_white_hot_duck_bath_toy',
                                 13: 'oreo_mega_stuf'}
        self.descriptor = feature_extractor

    def train(self, train_X, train_y):
        self.clf.fit(train_X, train_y)

    def predict_image_with_fg_mask(self, img, fg_mask):
        f = self.descriptor.describe_with_mask(img, fg_mask)
        pred = self.clf.predict(f)
        pred = pred[0]

        return self.label_name_table[pred]

    def predict_image(self, img, bg_model):
        fg_mask = bg_model.get_fg_mask(img)

        return self.predict_with_fg_mask(img, fg_mask)

    def predict(self, test_X):
        pred = self.clf.predict(test_X)

        return pred
    def prepare_data(self, train_data_dir, bg_model):
        X = []
        y = []

        obj_dirs = [os.path.join(train_data_dir, dir)
                    for dir in os.listdir(train_data_dir)]
        obj_dirs = [dir for dir in obj_dirs if os.path.isdir(dir)]

        dir_label_tuples = []
        for obj_dir in obj_dirs:
            name = obj_dir.split('/').pop()
            label = self.name_label_table[name]

            dir_label_tuples.append((obj_dir, label))

        for (img_dir, label) in dir_label_tuples:
            img_files = [os.path.join(img_dir, file)
                         for file in os.listdir(img_dir)
                         if file.endswith('.png')]

            for filename in img_files:
                img = cv2.imread(filename)
                fg_mask = bg_model.get_fg_mask(img)

                feature = self.descriptor.describe_with_mask(img, fg_mask)
                X.append(feature)
                y.append(label)

        X = np.asarray(X)
        y = np.asarray(y)

        return X, y

    def save_model(self, model_file='rgb_hist_model.txt'):
        f = open(model_file, 'w')
        cPickle.dump(self.clf, f)
        f.close()

    def load_model(self, model_file='rgb_hist_model.txt'):
        clf = cPickle.loads(open(model_file).read())
        self.clf = clf

        return self

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from fg_segmentation.background_subtractor import BackgroundSubtractor
    from rgbhistogram import RGBHistogram, HSVHistogram

    bg_images = '/Users/shixudongleo/Projects/RoboticsVision/vision_4_amazon_challenge/fg_segmentation/bg_images'
    bg_model = BackgroundSubtractor().build_bg_model(bg_dir=bg_images)

    feature_extractor = RGBHistogram()
    color_clf = Color_Hist_Classifier(feature_extractor)

    model_file = 'rgb_hist_model.txt'

    if os.path.exists(model_file):
        # load model
        color_clf.load_model(model_file=model_file)
    else:
        # train model
        train_data = './objects_training_data'
        X, y = color_clf.prepare_data(train_data, bg_model)
        color_clf.train(X, y)
        color_clf.save_model(model_file=model_file)


#    # test accuracy of classifier
#    from sklearn.cross_validation import train_test_split
#    from sklearn.metrics import classification_report
#    from sklearn.metrics import confusion_matrix
#
#    feature_extractor = RGBHistogram()
#    color_clf = Color_Hist_Classifier(feature_extractor)
#
#    data = './objects_training_data'
#    X, y = color_clf.prepare_data(data, bg_model)
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#    color_clf.train(X_train, y_train)
#
#    y_pred = color_clf.predict(X_test)
#    print classification_report(y_test, y_pred)
#    print confusion_matrix(y_test, y_pred, labels=range(13))
