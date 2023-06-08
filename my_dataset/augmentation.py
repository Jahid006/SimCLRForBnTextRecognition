import random
import cv2
import numpy as np
import skimage
from . import warp_augment


# todo: incorporate straug
# import straug


# Credit goes to:
# https://theai.codes/computer-vision/a-list-of-the-most-useful-opencv-filters/
# https://gist.github.com/Prasad9/28f6a2df8e8d463c6ddd040f4f6a028a?permalink_comment_id=2933012#gistcomment-2933012


def format(function):
    def inner(*args, **kwargs):
        kwargs["image"] = kwargs["image"] / 255.0
        output = function(*args, **kwargs)
        return output * 255.0

    return inner


class NoiseAugment:
    def __init__(self, probability=0.2):
        self.probability = probability

    def __call__(self, image: np.ndarray, probability=None):
        probability = self.probability if probability is None else probability
        method_list = [
            func
            for func in dir(self)
            if callable(getattr(self, func)) and not func.startswith("__")
        ]
        if np.random.rand() < probability:
            return getattr(self, random.choice(method_list))(image=image)

        return image

    def skip_noise(self, image=None):
        return image

    def otsu_binarization(self, image=None):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        otsu_threshold, image_result = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return image_result

    def inverse_binarization(self, image=None):
        image = self.otsu_binarization(image=image)
        return 255 - image

    def perspective(self, image=None):
        return warp_augment.perspective(image)

    @format
    def gaussian(self, image=None):
        return skimage.util.random_noise(image, mode="gaussian", clip=True)

    @format
    def localvar(self, image=None):
        return skimage.util.random_noise(image, mode="localvar")

    @format
    def poisson(self, image=None):
        return skimage.util.random_noise(image, mode="poisson", clip=True)

    @format
    def salt(self, image=None):
        return skimage.util.random_noise(image, mode="salt")

    @format
    def pepper(self, image=None):
        return skimage.util.random_noise(image, mode="pepper")

    @format
    def s_p(self, image=None):
        return skimage.util.random_noise(image, mode="s&p")

    @format
    def speckle(self, image=None):
        return skimage.util.random_noise(image, mode="speckle", clip=True)
    
    # def distort(self, image=None):
    #     segment = np.random.randint(1, 4)
    #     return warp_augment.distort(image, segment)

    # def stretch(self, image=None):
    #     segment = np.random.randint(1, 4)
    #     return warp_augment.stretch(image, segment)
