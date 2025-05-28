import numpy as np

def data_augmentation(image, mode):
    augmentations = {
        0: lambda img: img,                                 # original
        1: lambda img: np.flipud(img),                      # flip up and down
        2: lambda img: np.rot90(img),                       # rotate counterclockwise 90 degrees
        3: lambda img: np.flipud(np.rot90(img)),            # rotate 90 degrees and flip up and down
        4: lambda img: np.rot90(img, k=2),                  # rotate 180 degrees
        5: lambda img: np.flipud(np.rot90(img, k=2)),       # rotate 180 degrees and flip
        6: lambda img: np.rot90(img, k=3),                  # rotate 270 degrees
        7: lambda img: np.flipud(np.rot90(img, k=3)),      # rotate 270 degrees and flip
    }
    return augmentations[mode](image)

class Degradation(object):
    def __init__(self, args):
        self.args = args

    def _add_gaussian_noise(self, clean_patch, sigma):
        noise = np.random.randn(*clean_patch.shape)
        return np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)

    def _degrade_by_type(self, clean_patch, degrade_type):
        sigma_values = {0: 15, 1: 25, 2: 50}
        sigma = sigma_values.get(degrade_type, 15)  # Default to sigma=15 if type is not found
        return self._add_gaussian_noise(clean_patch, sigma)

    def degrade(self, clean_patch, degrade_type=None):
        return self._degrade_by_type(clean_patch, degrade_type)

def add_gaussian_noise( clean_patch, sigma):
    noise = np.random.randn(*clean_patch.shape)
    return np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)