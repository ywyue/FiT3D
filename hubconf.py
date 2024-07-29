import timm
import torch

_DINOv2_SMALL_URL = "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/dinov2_small_finetuned.pth"
_DINOv2_REG_SMALL_URL = "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/dinov2_reg_small_finetuned.pth"
_CLIP_BASE_URL = "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/clip_base_finetuned.pth"
_MAE_BASE_URL = "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/mae_base_finetuned.pth"
_DEIT3_BASE_URL = "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/deit3_base_finetuned.pth"

dinov2_small_name = "vit_small_patch14_dinov2.lvd142m"
dinov2_reg_small_name = "vit_small_patch14_reg4_dinov2.lvd142m"
clip_base_name = "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k"
mae_base_name = "vit_base_patch16_224.mae"
deit3_base_name = "deit3_base_patch16_224.fb_in1k"



def load_model(name, url):

    model = timm.create_model(
        name,
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True,
        dynamic_img_pad=False,
    )

    ### override weights
    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model.load_state_dict(state_dict)

    return model


def dinov2_small_fine():

    model = load_model(name=dinov2_small_name, url=_DINOv2_SMALL_URL)

    return model


def dinov2_reg_small_fine():

    model = load_model(name=dinov2_reg_small_name, url=_DINOv2_REG_SMALL_URL)

    return model


def clip_base_fine():

    model = load_model(name=clip_base_name, url=_CLIP_BASE_URL)

    return model


def mae_base_fine():

    model = load_model(name=mae_base_name, url=_MAE_BASE_URL)

    return model


def deit3_base_fine():

    model = load_model(name=deit3_base_name, url=_DEIT3_BASE_URL)

    return model