from torchvision.transforms.functional import (
    rotate,
    affine,
    adjust_brightness,
    adjust_contrast,
    adjust_gamma,
    adjust_saturation,
    gaussian_blur,
)

import random


def Blur(img, v):
    return gaussian_blur(img, [5, 5], v)


def Brightness(img, v):
    return adjust_brightness(img, v)


def Contrast(img, v):
    return adjust_brightness(img, v)


def Gamma(img, v):
    return adjust_gamma(img, v)


def Identity(img, v):
    return img


def Rotate(img, v):
    return rotate(img, v)


def ShearX(img, v):
    return affine(img, angle=0, scale=1, translate=(0, 0), shear=(v, 0))


def ShearY(img, v):
    return affine(img, angle=0, scale=1, translate=(0, 0), shear=(0, v))


def TranslateX(img, v):
    v = v * img.shape[-2]
    return affine(img, angle=0, scale=1, translate=(v, 0), shear=(0, 0))


def TranslateY(img, v):
    v = v * img.shape[-1]
    return affine(img, angle=0, scale=1, translate=(0, v), shear=(0, 0))


def torch_augmentation_list():
    l = [
        (Blur, 0.5, 2),
        (Brightness, 0.1, 2),
        (Contrast, 0.1, 2),
        (Gamma, 0.1, 2.0),
        (Identity, 0, 1),
        (Rotate, -30, 30),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3),
    ]
    return l


if __name__ == "__main__":
    import time
    import torch
    import numpy as np

    test_img = torch.zeros([30, 13, 64, 64]).cuda()
    test_img += torch.rand([30, 13, 64, 64]).cuda()
    print(torch.max(test_img))
    print(torch.min(test_img))
    for op, min_val, max_val in torch_augmentation_list():
        val = min_val + float(max_val - min_val) * random.random()
        start = time.time()
        for i in range(100):
            img = op(test_img, val)
        end = time.time()
        print(f"{op} required: {end - start}s")
