"""
Image preprocessing pipelines for each model type.

Each model has specific preprocessing requirements. This module
centralizes transform definitions so they can be reused in training
and inference consistently.
"""


def get_histopathology_transforms():
    """
    Preprocessing for histopathology patches.

    - Resize to 224×224 (ResNet input size)
    - Normalize with ImageNet statistics
    """
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def get_chest_xray_transforms(augment=False):
    """
    Preprocessing for chest X-ray images.

    - Resize to 224×224
    - Convert grayscale to 3-channel RGB
    - Optional augmentation for training
    - Normalize with ImageNet statistics
    """
    import torchvision.transforms as T

    if augment:
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
