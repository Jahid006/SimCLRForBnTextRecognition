import random
import numpy as np
import skimage
import albumentations as A


# https://theai.codes/computer-vision/a-list-of-the-most-useful-opencv-filters/
# https://gist.github.com/Prasad9/28f6a2df8e8d463c6ddd040f4f6a028a?permalink_comment_id=2933012#gistcomment-2933012
# You could use straug and Albamunations too


def normalizer(function):
    def inner(*args, **kwargs):
        kwargs["image"] = kwargs["image"] / 255.0
        output = function(*args, **kwargs)
        return output * 255.0
    return inner


class Augmentation(object):
    def __init__(self, proablility=0.2):
        self.probability = proablility
        self.level1_augmentation = [
            func
            for func in dir(self)
            if (
                callable(getattr(self, func))
                and not func.startswith("__")
                and func.startswith("l1")
            )
        ]

        self.level2_augmentation = [
            func
            for func in dir(self)
            if (
                callable(getattr(self, func))
                and not func.startswith("__")
                and func.startswith("l2")
            )
        ]

    def __call__(self, image: np.ndarray):
        try:
            function = getattr(self, random.choice(self.level2_augmentation))
            if np.random.rand() < self.probability:
                image = function(image=image)
            function = getattr(self, random.choice(self.level1_augmentation))
            if np.random.rand() < self.probability:
                image = function(image=image)
        except Exception as e:
            print(f"Exception: {e} @{function}")

        return image
    
    def l2_piecewise_affine(self, image=None):
        return A.PiecewiseAffine(p=1.0)(image=image)["image"]

    def l2_rotate(self, image=None):
        return A.Rotate(7, p=1.0)(image=image)["image"]

    def l2_grid_distortion(self, image=None):
        return A.GridDistortion(p=1.0)(image=image)["image"]

    def l2_optical_distortion(self, image=None):
        return A.OpticalDistortion(p=1.0)(image=image)["image"]

    def l1_inverse(self, image=None):
        return 255 - image

    def l1_skip_noise(self, image=None):
        return image

    @normalizer
    def l1_gaussian(self, image=None):
        return skimage.util.random_noise(image, mode="gaussian", clip=True)

    @normalizer
    def l1_localvar(self, image=None):
        return skimage.util.random_noise(image, mode="localvar")

    @normalizer
    def l1_poisson(self, image=None):
        return skimage.util.random_noise(image, mode="poisson", clip=True)

    @normalizer
    def l1_salt(self, image=None):
        return skimage.util.random_noise(image, mode="salt")

    @normalizer
    def l1_pepper(self, image=None):
        return skimage.util.random_noise(image, mode="pepper")

    @normalizer
    def l1_s_p(self, image=None):
        return skimage.util.random_noise(image, mode="s&p")

    @normalizer
    def l1_speckle(self, image=None):
        return skimage.util.random_noise(image, mode="speckle", clip=True)
