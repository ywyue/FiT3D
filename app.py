import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import spaces
import timm 
import torch
import torchvision.transforms as T
import types
import albumentations as A

from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from torch_kmeans import KMeans, CosineSimilarity

cmap = plt.get_cmap("tab20")
MEAN = np.array([123.675, 116.280, 103.530]) / 255
STD = np.array([58.395, 57.120, 57.375]) / 255

transforms = A.Compose([
            A.Normalize(mean=list(MEAN), std=list(STD)),
    ])

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

def viz_feat(feat):

    _,_,h,w = feat.shape
    feat = feat.squeeze(0).permute((1,2,0))
    projected_featmap = feat.reshape(-1, feat.shape[-1]).cpu()

    pca = PCA(n_components=3)
    pca.fit(projected_featmap)
    pca_features = pca.transform(projected_featmap)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    res_pred = Image.fromarray(pca_features.reshape(h, w, 3).astype(np.uint8))

    return res_pred

def plot_feats(model_option, ori_feats, fine_feats, ori_labels=None, fine_labels=None):

    ori_feats_map = viz_feat(ori_feats)
    fine_feats_map = viz_feat(fine_feats)

    fig, ax = plt.subplots(2, 2, figsize=(6, 5))
    ax[0][0].imshow(ori_feats_map)
    ax[0][0].set_title("Original " + model_option, fontsize=15)
    ax[0][1].imshow(fine_feats_map)
    ax[0][1].set_title("Fine-tuned", fontsize=15)
    ax[1][0].imshow(ori_labels)
    ax[1][1].imshow(fine_labels)
    for xx in ax:
      for x in xx:
        x.xaxis.set_major_formatter(plt.NullFormatter())
        x.yaxis.set_major_formatter(plt.NullFormatter())
        x.set_xticks([])
        x.set_yticks([])
        x.axis('off')

    plt.tight_layout()
    plt.close(fig)
    return fig

    
def download_image(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)


def process_image(image, stride, transforms):
    transformed = transforms(image=np.array(image))
    image_tensor = torch.tensor(transformed['image'])
    image_tensor = image_tensor.permute(2,0,1)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    h, w = image_tensor.shape[2:]

    height_int = (h // stride)*stride
    width_int = (w // stride)*stride

    image_resized = torch.nn.functional.interpolate(image_tensor, size=(height_int, width_int), mode='bilinear')

    return image_resized


def kmeans_clustering(feats_map, n_clusters=20):
    if n_clusters == None:
        n_clusters = 20
    print('num clusters: ', n_clusters)
    B, D, h, w = feats_map.shape
    feats_map_flattened = feats_map.permute((0, 2, 3, 1)).reshape(B, -1, D)

    kmeans_engine = KMeans(n_clusters=n_clusters, distance=CosineSimilarity)
    kmeans_engine.fit(feats_map_flattened)
    labels = kmeans_engine.predict(
        feats_map_flattened
        )
    labels = labels.reshape(
        B, h, w
        ).float()
    labels = labels[0].cpu().numpy()

    label_map = cmap(labels / n_clusters)[..., :3]
    label_map = np.uint8(label_map * 255)
    label_map = Image.fromarray(label_map)

    return label_map
    
def load_model(options):
    original_models = {}
    fine_models = {}
    for option in tqdm(options):
        print('Please wait ...')
        print('loading weights of ', option)
        original_models[option] = timm.create_model(
                timm_model_card[option],
                pretrained=True,
                num_classes=0,
                dynamic_img_size=True,
                dynamic_img_pad=False,
            ).to(device)
        original_models[option].get_intermediate_layers = types.MethodType(
            get_intermediate_layers,
            original_models[option]
        )
            
        fine_models[option] = torch.hub.load("ywyue/FiT3D", our_model_card[option]).to(device)
        fine_models[option].get_intermediate_layers = types.MethodType(
            get_intermediate_layers,
            fine_models[option]
        )
    print('Done! Now play the demo :)')
    return original_models, fine_models
        
if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("device: ")
    print(device)

    example_urls = {
        "library.jpg": "https://n.ethz.ch/~yuayue/assets/fit3d/demo_images/library.jpg",
        "livingroom.jpg": "https://n.ethz.ch/~yuayue/assets/fit3d/demo_images/livingroom.jpg",
        "airplane.jpg": "https://n.ethz.ch/~yuayue/assets/fit3d/demo_images/airplane.jpg",
        "ship.jpg": "https://n.ethz.ch/~yuayue/assets/fit3d/demo_images/ship.jpg",
        "chair.jpg": "https://n.ethz.ch/~yuayue/assets/fit3d/demo_images/chair.jpg",
    }

    example_dir = "/tmp/examples"

    os.makedirs(example_dir, exist_ok=True)


    for name, url in example_urls.items():
        save_path = os.path.join(example_dir, name)
        if not os.path.exists(save_path):
            print(f"Downloading to {save_path}...")
            download_image(url, save_path)
        else:
            print(f"{save_path} already exists.")
        
    image_input = gr.Image(label="Choose an image:",
                           height=500,
                           type="pil",
                           image_mode='RGB',
                           sources=['upload', 'webcam', 'clipboard']
                           )

    options = ['DINOv2', 'DINOv2-reg', 'CLIP', 'MAE', 'DeiT-III']
    model_option = gr.Radio(options, value="DINOv2", label='Choose a 2D foundation model')
    kmeans_num = gr.Number(
                            label="Number of K-Means clusters", value=20
                        )
    
    timm_model_card = {
        "DINOv2": "vit_small_patch14_dinov2.lvd142m",
        "DINOv2-reg": "vit_small_patch14_reg4_dinov2.lvd142m",
        "CLIP": "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
        "MAE": "vit_base_patch16_224.mae",
        "DeiT-III": "deit3_base_patch16_224.fb_in1k"
    }
    
    our_model_card = {
        "DINOv2": "dinov2_small_fine",
        "DINOv2-reg": "dinov2_reg_small_fine",
        "CLIP": "clip_base_fine",
        "MAE": "mae_base_fine",
        "DeiT-III": "deit3_base_fine"
    }


    os.environ['TORCH_HOME'] = '/tmp/.cache'
    # os.environ['GRADIO_EXAMPLES_CACHE'] = '/tmp/gradio_cache'

    # Pre-load all models
    original_models, fine_models = load_model(options)

    @spaces.GPU
    def fit3d(image, model_option, kmeans_num):
        
        # Select model
        original_model = original_models[model_option]
        fine_model = fine_models[model_option]
        
        # Data preprocessing
        p = original_model.patch_embed.patch_size
        stride = p if isinstance(p, int) else p[0]
        image_resized = process_image(image, stride, transforms)


        with torch.no_grad():
            ori_feats = original_model.get_intermediate_layers(image_resized, n=[8,9,10,11], reshape=True, return_prefix_tokens=False,
                                            return_class_token=False, norm=True)
            fine_feats = fine_model.get_intermediate_layers(image_resized, n=[8,9,10,11], reshape=True, return_prefix_tokens=False,
                                            return_class_token=False, norm=True)

        ori_feats = ori_feats[-1]
        fine_feats = fine_feats[-1]

        ori_labels = kmeans_clustering(ori_feats, kmeans_num)
        fine_labels = kmeans_clustering(fine_feats, kmeans_num)
        
        return plot_feats(model_option, ori_feats, fine_feats, ori_labels, fine_labels)
        

    demo = gr.Interface(
        title="<div> \
        <h1>FiT3D</h1> \
        <h2>Improving 2D Feature Representations by 3D-Aware Fine-Tuning</h2> \
        <h2>ECCV 2024</h2> \
        </div>",
        description="<div style='display: flex; justify-content: center; align-items: center; text-align: center;'> \
        <a href='https://arxiv.org/abs/2407.20229'><img src='https://img.shields.io/badge/arXiv-2407.20229-red'></a> \
        &nbsp; \
        <a href='https://ywyue.github.io/FiT3D'><img src='https://img.shields.io/badge/Project_Page-FiT3D-green' alt='Project Page'></a> \
        &nbsp; \
        <a href='https://github.com/ywyue/FiT3D'><img src='https://img.shields.io/badge/Github-Code-blue'></a> \
        </div>",
        fn=fit3d, 
        inputs=[image_input, model_option, kmeans_num],
        outputs="plot",
        examples=[
                    ["/tmp/examples/library.jpg", "DINOv2", 20],
                    ["/tmp/examples/livingroom.jpg", "DINOv2", 20],
                    ["/tmp/examples/airplane.jpg", "DINOv2", 20],
                    ["/tmp/examples/ship.jpg", "DINOv2", 20],
                    ["/tmp/examples/chair.jpg", "DINOv2", 20],
            ],
        cache_examples=True)
    demo.launch()
