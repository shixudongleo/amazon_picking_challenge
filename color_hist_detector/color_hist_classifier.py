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
        path, _ = os.path.split(os.path.realpath(__file__))
        self.model_file = os.path.join(path, 'rgb_hist_model.txt')
        self.train_dir  = os.path.join(path, 'objects_training_data')
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
                                 'oreo_mega_stuf': 13,
                                 'mead_index_cards': 14}

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
                                 13: 'oreo_mega_stuf',
                                 14: 'mead_index_cards'}
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

    def prepare_data(self, bg_model):
        X = []
        y = []

        obj_dirs = [os.path.join(self.train_dir, dir)
                    for dir in os.listdir(self.train_dir)]
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

    def save_model(self):
        f = open(self.model_file, 'w')
        cPickle.dump(self.clf, f)
        f.close()

    def load_model(self):
        clf = cPickle.loads(open(self.model_file).read())
        self.clf = clf

    def is_model_trained(self):
        if os.path.exists(self.model_file):
            return True
        else:
            return False
