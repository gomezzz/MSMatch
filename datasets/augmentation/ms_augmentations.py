from torchvision import transforms

import albumentations as A

from imgaug import augmenters as iaaa


def AutoContrast(img, v):
    return iaaa.pillike.Autocontrast(cutoff=0)(image=img)["image"]


def Brightness(img, v):
    return iaaa.pillike.EnhanceBrightness(factor=v)(image=img)["image"]


def Color(img, v):
    return iaaa.pillike.EnhanceColor(factor=v)(image=img)["image"]


def Contrast(img, v):
    return iaaa.pillike.EnhanceContrast(factor=v)(image=img)["image"]


def Equalize(img, v):
    return A.equalize(img)


def Identity(img, v):
    return img


def Posterize(img, v):
    return A.posterize(img, v)


def Rotate(img, v):
    return A.rotate(img, v)


def Sharpness(img, v):
    v = v / 2  # In PIL code 0.1 to 1.9
    return A.IAASharpen(alpha=v, always_apply=True)(image=img)["image"]


def ShearX(img, v):
    return A.IAAAffine(shear=(v, 0), always_apply=True)(image=img)["image"]


def ShearY(img, v):
    return A.IAAAffine(shear=(0, v), always_apply=True)(image=img)["image"]


def Solarize(img, v):
    return A.solarize(img, v)


def TranslateX(img, v):
    return A.IAAAffine(translate_percent=(v, 0), always_apply=True)(image=img)["image"]


def TranslateY(img, v):
    return A.IAAAffine(translate_percent=(0, v), always_apply=True)(image=img)["image"]


def ms_augmentation_list():
    l = [
        # The below four don't work with multispectral images
        # (AutoContrast, 0, 1),
        # (Brightness, 0.05, 0.95),
        # (Color, 0.05, 0.95),
        # (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3),
    ]
    return l
