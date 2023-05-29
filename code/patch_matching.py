import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional
from torchvision import transforms
from dinov2.eval.utils import ModelWithIntermediateLayers, evaluate
import hubconf

import cv2
import numpy as np
import tensorwatch as tw
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch
from PIL import Image, ImageDraw
from torch.nn.functional import one_hot, softmax

import dinov2.distributed as distributed
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.eval.metrics import AccuracyAveraging, build_topk_accuracy_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import ModelWithNormalize, evaluate, extract_features


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = [],
    add_help: bool = True,
):
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the train features on cpu, slower"
        "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
    )
    parser.set_defaults(
    )
    return parser

def patch_feature_pca(features, filename):
    pca = PCA(n_components=3)
    pca.fit(features)

    pca_features = pca.transform(features)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max()- pca_features .min())
    pca_features = pca_features * 255
    plt.imshow(pca_features.reshape(16,16,3).astype(np.uint8))
    plt.savefig(filename)

def patch_feature_matching(img1, img2, feature1, feature2):
    
    def draw_lines(event, x, y, flags, param):
        unit = 224 // 14
        if (event == cv2.EVENT_LBUTTONDBLCLK) & (x < 224):
            p1 = (y // 14) * 16 + x // 14
            p2 = index[0][p1]
            p1y, p1x = p1 // unit + 0.5, p1 % unit + 0.5
            p2y, p2x = p2 // unit + 0.5, p2 % unit + 0.5
            cv2.line(full_img,(int(p1x * 14), 
                     int(p1y * 14)),(int(p2x * 14 + 224), 
                     int(p2y * 14)),(0,255,0),1)
    
    sim_matrix = torch.bmm(feature1, feature2.permute(0, 2, 1))
    value, index = sim_matrix.max(-1)
    index = index.to('cpu', torch.uint8).numpy()
    img1 = Image.fromarray(img1[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
    img2 = Image.fromarray(img2[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
    full_img = Image.new('RGB', (224 * 2, 224))
    
    full_img.paste(img1, (0, 0))
    full_img.paste(img2, (224, 0))
    
    full_img = cv2.cvtColor(np.asarray(full_img),cv2.COLOR_RGB2BGR)  
      
    cv2.namedWindow('path_feature_matching')
    cv2.setMouseCallback('path_feature_matching', draw_lines)

    while (1):
        cv2.imshow('path_feature_matching', full_img)
        if cv2.waitKey(1)&0xFF == ord('q'):#按q键退出
            break
    cv2.destroyAllWindows()


def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    # model = ModelWithNormalize(model)
    # dinov2_vits14 = hubconf.dinov2_vits14().cuda()
        
    transform = None
    
    transforms_list = [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    transform = transforms.Compose(transforms_list)
    
    image1 = Image.open("/home/hqlab/workspace/closure/EE5346_2023_project/test_data/origin/test.jpg")
    image1 = transform(image1)[:3].unsqueeze(0).cuda()

    image2 = Image.open("/home/hqlab/workspace/closure/EE5346_2023_project/test_data/origin/test1.jpg")
    image2 = transform(image2)[:3].unsqueeze(0).cuda()

    # feature size:[b,16*16,1536]
    with torch.no_grad():
        model.eval()
        feature1 = model.forward_features(image1)['x_norm_patchtokens']
        feature2 = model.forward_features(image2)['x_norm_patchtokens']
    
    # patch_feature_pca(feature1,"image1_pca.png")
    # patch_feature_pca(feature2,"image2_pca.png")
    
    patch_feature_matching(image1,image2,feature1,feature2)
 
    return 0

if __name__ == "__main__":
    description = "DINOv2 model patch matching"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))