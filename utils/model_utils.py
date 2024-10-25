############################################################
# Code for FiT3D 
# by Yuanwen Yue
############################################################


import numpy as np
import albumentations as A
import timm
import torch
import types

from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms


ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255


timm_model_card = {
    "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_base": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_reg_small": "vit_small_patch14_reg4_dinov2.lvd142m",
    "clip_base": "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
    "mae_base": "vit_base_patch16_224.mae",
    "deit3_base": "deit3_base_patch16_224.fb_in1k"
}

def viz_feat(feat, file_path):

    _, _, h, w = feat.shape
    feat = feat.squeeze(0).permute((1,2,0))
    projected_featmap = feat.reshape(-1, feat.shape[-1]).cpu()

    pca = PCA(n_components=3)
    pca.fit(projected_featmap)
    pca_features = pca.transform(projected_featmap)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    res_pred = Image.fromarray(pca_features.reshape(h, w, 3).astype(np.uint8))
    res_pred.save(file_path)
    print("... saved to: ", file_path)

def get_intermediate_layers(
    self,
    x: torch.Tensor,
    n=1,
    reshape: bool = False,
    return_prefix_tokens: bool = False,
    return_class_token: bool = False,
    norm: bool = True,
):
    outputs = self._intermediate_layers(x, n)
    if norm:
        outputs = [self.norm(out) for out in outputs]
    if return_class_token:
        prefix_tokens = [out[:, 0] for out in outputs]
    else:
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
    outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

    if reshape:
        B, C, H, W = x.shape
        grid_size = (
            (H - self.patch_embed.patch_size[0])
            // self.patch_embed.proj.stride[0]
            + 1,
            (W - self.patch_embed.patch_size[1])
            // self.patch_embed.proj.stride[1]
            + 1,
        )
        outputs = [
            out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
            .permute(0, 3, 1, 2)
            .contiguous()
            for out in outputs
        ]

    if return_prefix_tokens or return_class_token:
        return tuple(zip(outputs, prefix_tokens))
    return tuple(outputs)

def make_transform(size):
    return A.Compose([
            A.Resize(width=size[1], height=size[0]),
            A.Normalize(mean=list(ADE_MEAN), std=list(ADE_STD)),
    ])

# def build_dino(model='s'):
#     if model=='s':
#         conf = load_and_merge_config('eval/vits14_pretrain')
#         model = build_model_for_eval(conf, 'dinov2_weights/dinov2_vits14_pretrain.pth')
#     elif model=='b': 
#         conf = load_and_merge_config('eval/vitb14_pretrain')
#         model = build_model_for_eval(conf, 'dinov2_weights/dinov2_vitb14_pretrain.pth')
#     elif model=='l': 
#         conf = load_and_merge_config('eval/vitl14_pretrain')
#         model = build_model_for_eval(conf, 'dinov2_weights/dinov2_vitl14_pretrain.pth')
#     elif model=='g': 
#         conf = load_and_merge_config('eval/vitg14_pretrain')
#         model = build_model_for_eval(conf, 'dinov2_weights/dinov2_vitg14_pretrain.pth')


#     return model

def build_2d_model(model_name="dinov2_small"):

    assert model_name in timm_model_card.keys(), "invalid model name"
    model = timm.create_model(
        timm_model_card[model_name],
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True,
        dynamic_img_pad=False,
    )

    model.get_intermediate_layers = types.MethodType(
            get_intermediate_layers,
            model,
    )

    return model


def forward_2d_model(image, feature_extractor):

    _, height, width = image.shape
    
    stride = feature_extractor.patch_embed.patch_size[0]
    width_int = (width // stride)*stride
    height_int = (height // stride)*stride

    transform = make_transform((height_int, width_int))

    transformed_image = transform(image=np.transpose(image, (1,2,0)))['image']
    transformed_image = torch.Tensor(transformed_image).cuda()
    transformed_image = transformed_image.permute((2,0,1))
    batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image

    featmap = feature_extractor.get_intermediate_layers(
                    batch,
                    n=[len(feature_extractor.blocks)-1],
                    reshape=True,
                    return_prefix_tokens=False,
                    return_class_token=False,
                    norm=True,
                )[-1]

    # result = feature_model.forward_features(batch)
    # features = result['x_norm_patchtokens']

    # features = features.squeeze(0)

    # fmap_width = int(width_int / 14)
    # fmap_height = int(height_int / 14)

    # featmap = features.reshape(fmap_height, fmap_width, features.shape[-1]).permute((2,0,1)).unsqueeze(0)

    return featmap


def forward_2d_model_batch(images, feature_extractor):

    B, _, height, width = images.shape
    
    stride = feature_extractor.patch_embed.patch_size[0]
    width_int = (width // stride)*stride
    height_int = (height // stride)*stride

    batch = torch.nn.functional.interpolate(images, size=(height_int, width_int), mode='bilinear')

    featmap = feature_extractor.get_intermediate_layers(
                    batch,
                    n=[len(feature_extractor.blocks)-1],
                    reshape=True,
                    return_prefix_tokens=False,
                    return_class_token=False,
                    norm=True,
                )[-1]

    return featmap
