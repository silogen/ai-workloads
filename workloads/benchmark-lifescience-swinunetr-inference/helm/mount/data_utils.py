import logging

from monai import transforms

LOGGER = logging.getLogger(__name__)

IMAGE_DATA = "image"
SPACING_MODE = "bilinear"
DEVICE = "cuda"


def get_transforms(args):
    """Returns the transforms to be applied to the input data."""
    inference_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=[IMAGE_DATA]),
            transforms.EnsureChannelFirstd(keys=[IMAGE_DATA], channel_dim="no_channel"),
            transforms.Spacingd(
                keys=[IMAGE_DATA], pixdim=(args.space_x, args.space_y, args.space_z), mode=SPACING_MODE
            ),
            transforms.ScaleIntensityRanged(
                keys=[IMAGE_DATA], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
        ]
    )
    return inference_transforms


def get_post_transforms_inverter(forward_transforms_obj):
    """Transform to revert all transforms previously applied."""
    return transforms.Invertd(
        keys=IMAGE_DATA,
        transform=forward_transforms_obj,
        orig_keys=IMAGE_DATA,
        orig_meta_keys=f"{IMAGE_DATA}_meta_dict",
        nearest_interp=True,
        to_tensor=True,
        device=DEVICE,
    )
