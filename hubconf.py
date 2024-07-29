import timm
import torch

_DINOv2_SMALL_URL = "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/dinov2_small_finetuned.pth"

dinov2_small_name = "vit_small_patch14_dinov2.lvd142m"


def load_model(*, finetuned: bool = True, model_type, **kwargs):

    model = timm.create_model(
        model_type,
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True,
        dynamic_img_pad=False,
    )

    if finetuned:
        state_dict = torch.hub.load_state_dict_from_url(_DINOv2_SMALL_URL, map_location="cpu")
        model.load_state_dict(state_dict)

    return model


def dinov2_small(*, finetuned: bool = True, **kwargs):

    model = load_model(finetuned=finetuned, model_type=dinov2_small_name, **kwargs)

    return model