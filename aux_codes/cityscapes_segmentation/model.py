from ffnet import resnet
from ffnet.ffnet_blocks import create_ffnet


def segmentation_ffnet40S_BBB_mobile_pre_down(weights_path="ffnet40S/ffnet40S_BBB_cityscapes_state_dict_quarts.pth"):
    return create_ffnet(
        ffnet_head_type="B_mobile",
        task="segmentation_B",
        num_classes=19,
        model_name="ffnnet40S_BBB_mobile_pre_down",
        backbone=resnet.Resnet40S,
        pre_downsampling=True,
        pretrained_weights_path=weights_path,
        strict_loading=True,
    )
