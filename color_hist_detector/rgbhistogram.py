import cv2


class RGBHistogram:
    def __init__(self, bins=[8, 8, 8]):
        # store the number of bins the histogram will use
        self.bins = bins

    def describe(self, image):
        # compute a 3D histogram in the RGB colorspace,
        # then normalize the histogram so that images
        # with the same content, but either scaled larger
        # or smaller will have (roughly) the same histogram
        hist = cv2.calcHist([image], [0, 1, 2],
                            None, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist)

        # return out 3D histogram as a flattened array
        return hist.flatten()

    def describe_with_mask(self, image, mask_img):
        # compute a 3D histogram in the RGB colorspace,
        # then normalize the histogram so that images
        # with the same content, but either scaled larger
        # or smaller will have (roughly) the same histogram
        hist = cv2.calcHist([image], [0, 1, 2],
                            mask_img, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist)

        # return out 3D histogram as a flattened array
        return hist.flatten()

    def get_dim(self):
        dim = 1
        for bin in self.bins:
            dim = dim * bin
        return dim


class HSVHistogram:
    def __init__(self, bins=[10, 10]):
        # store the number of bins the histogram will use
        self.bins = bins

    def describe(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1],
                            None, self.bins, [0, 256, 0, 256])
        hist = cv2.normalize(hist)

        return hist.flatten()

    def describe_with_mask(self, image, mask_img):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1],
                            mask_img, self.bins, [0, 256, 0, 256])
        hist = cv2.normalize(hist)

        return hist.flatten()

    def get_dim(self):
        dim = 1
        for bin in self.bins:
            dim = dim * bin
        return dim
