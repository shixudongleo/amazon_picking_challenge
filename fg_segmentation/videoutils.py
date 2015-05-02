import cv2
import numpy as np


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


class BackgroundSubtractorWithShadow:
    def __init__(self):
        self.bg_model = cv2.BackgroundSubtractorMOG2()
        self.shadow = 250

    def set_shadow_value(self, gray_value):
        self.shadow = gray_value

    def get_fg_mask(self, img):
        fg_mask = self.bg_model.apply(img)

        # retain only high confidence mask
        fg_mask = cv2.inRange(fg_mask, self.shadow, 256)
        return fg_mask

    def train_bg_model(self, img):
        self.bg_model.apply(img)

def detect_objects_by_contour(img, min_size=100, max_size=1000):
    # contour detection
    # return contour and mask tutples
    (contours, _) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height= img.shape[0]
    width = img.shape[1]

    contour_mask_list = []
    for i in range(len(contours)):
        contour = contours[i]

        if len(contour) < min_size or len(contour) > max_size:
            continue

        mask = np.zeros((height, width), dtype='uint8')
        cv2.drawContours(mask, contours, i, [255, 255, 255], -1)

        (x, y, w, h) = cv2.boundingRect(contour)
        bbox = (y, x, h, w)
        contour_mask_list.append((bbox, mask))

    return contour_mask_list


import os


if __name__ == '__main__':
    data = '/Users/shixudongleo/Downloads/fore_back_ground/'
    bg_data = os.path.join(data, 'back_img')
    fg_data = os.path.join(data, 'obj_img')

    bg_model = BackgroundSubtractor()
    # bg_model = BackgroundSubtractorWithShadow()


    bg_imgs = [os.path.join(bg_data, file) for file in os.listdir(bg_data)]
    fg_imgs = [os.path.join(fg_data, file) for file in os.listdir(fg_data)]

    for bg_img in bg_imgs:
        img = cv2.imread(bg_img)
        bg_model.train_bg_model(img)

    for fg_img in fg_imgs:
        img = cv2.imread(fg_img)
        fg_mask = bg_model.get_fg_mask(img)

        cv2.imshow('fg', fg_mask)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        bbox_fgs = detect_objects_by_contour(fg_mask, min_size=30)
        for bbox, fg in bbox_fgs:
            cv2.imshow('fg_each', fg)
            cv2.waitKey(1000)

            # draw bounding box
            (y, x, h, w) = bbox
            color_fg = cv2.cvtColor(fg, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(color_fg, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.imshow('bbox', color_fg)
            cv2.waitKey(1000)
