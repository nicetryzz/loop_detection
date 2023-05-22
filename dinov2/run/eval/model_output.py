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
from PIL import Image
from torch.nn.functional import one_hot, softmax

import dinov2.distributed as distributed
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.eval.metrics import AccuracyAveraging, build_topk_accuracy_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import ModelWithNormalize, evaluate, extract_features

def sift_algo(img1_tensor, img2_tensor):
    img1_cv = img1_tensor.permute(1, 2, 0)
    img1_cv = img1_cv * 255
    img1_cv = img1_cv.numpy()
    img1_cv = img1_cv.astype(np.uint8).copy()
    img2_cv = img2_tensor.permute(1, 2, 0)
    img2_cv = img2_cv * 255
    img2_cv = img2_cv.numpy()
    img2_cv = img2_cv.astype(np.uint8).copy()
    img = np.append(img1_cv, img2_cv, axis=1)

    img1_grey = cv2.cvtColor(img1_cv, cv2.COLOR_RGB2GRAY)
    img2_grey = cv2.cvtColor(img2_cv, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=None, nOctaveLayers=None, contrastThreshold=None, edgeThreshold=None, sigma=None)
    keyPointsLeft, describesLeft = sift.detectAndCompute(img1_grey, None)
    keyPointsRight, describesRight = sift.detectAndCompute(img2_grey, None)
    # K-D tree建立索引方式的常量参数
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # checks指定索引树要被遍历的次数
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_1 = flann.knnMatch(describesLeft, describesRight, k=2)  # 进行匹配搜索，参数k为返回的匹配点对数量
    # 把保留的匹配点放入good列表
    good1 = []
    T = 0.85  # 阈值
    # 筛选特征点
    for i, (m, n) in enumerate(matches_1):
        if m.distance < T * n.distance:  # 如果最近邻点的距离和次近邻点的距离比值小于阈值，则保留最近邻点
            good1.append(m)
        #  双向交叉检查方法
    matches_2 = flann.knnMatch(describesRight, describesLeft, k=2)  # 进行匹配搜索
    # 把保留的匹配点放入good2列表
    good2 = []
    for (m, n) in matches_2:
        if m.distance < T * n.distance:  # 如果最近邻点的距离和次近邻点的距离比值小于阈值，则保留最近邻点
            good2.append(m)
    match_features = []  # 存放最终的匹配点
    for i in good1:
        for j in good2:
            if (i.trainIdx == j.queryIdx) & (i.queryIdx == j.trainIdx):
                match_features.append(i)
    src_pts = [keyPointsLeft[m.queryIdx].pt for m in match_features]
    dst_pts = [keyPointsRight[m.trainIdx].pt for m in match_features]
    for i in range(min(len(src_pts), 10)):
        img = cv2.line(img, (int(src_pts[i][0]), int(src_pts[i][1])),
                       (int(dst_pts[i][0] + img1_cv.shape[1]), int(dst_pts[i][1])), (255, 255, 0), 2)
    cv2.imshow('sift', img)
    cv2.waitKey(0)
    return img


def patch2piex(patch, h, w, patch_num):
    patch_num_per_direction = patch_num ** 0.5
    patch_w = w / patch_num_per_direction
    patch_h = h / patch_num_per_direction
    patch_h_id = patch // patch_num_per_direction
    patch_w_id = patch % patch_num_per_direction
    return (int(0.5 * patch_h + patch_h_id * patch_h), int(0.5 * patch_w + patch_w_id * patch_w))

def draw_patch_corresponding_lines(img1_tensor, img1_patches, img2_tensor, img2_patches, patch_num):
    img1_cv = img1_tensor.permute(1, 2, 0)
    img1_cv = img1_cv * 255
    img1_cv = img1_cv.numpy()
    img1_cv = img1_cv.astype(np.uint8).copy()
    img2_cv = img2_tensor.permute(1, 2, 0)
    img2_cv = img2_cv * 255
    img2_cv = img2_cv.numpy()
    img2_cv = img2_cv.astype(np.uint8).copy()
    img = np.append(img1_cv, img2_cv, axis=1)
    # img = np.ascontiguousarray(img, dtype=np.uint8)
    for patch_i in range(len(img1_patches)):
        piex1 = patch2piex(img1_patches[patch_i], img1_cv.shape[0], img1_cv.shape[1], patch_num)
        piex2 = patch2piex(img2_patches[patch_i], img2_cv.shape[0], img2_cv.shape[1], patch_num)
        img = cv2.line(img, (piex1[1], piex1[0]), (img1_cv.shape[1] + piex2[1], piex2[0]), (255, 255, 0), 2)

    cv2.imshow('dino', img)
    # cv2.waitKey(0)
    return img

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
    # parser.add_argument(
    #     "--train-dataset",
    #     dest="train_dataset_str",
    #     type=str,
    #     help="Training dataset",
    # )
    # parser.add_argument(
    #     "--val-dataset",
    #     dest="val_dataset_str",
    #     type=str,
    #     help="Validation dataset",
    # )
    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the train features on cpu, slower"
        "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
    )
    # parser.add_argument(
    #     "--batch-size",
    #     type=int,
    #     help="Batch size.",
    # )
    parser.set_defaults(
        # train_dataset_str="ImageNet:split=TRAIN",
        # val_dataset_str="ImageNet:split=VAL",
        # batch_size=256,
    )
    return parser


def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    # model = ModelWithNormalize(model)
    # dinov2_vits14 = hubconf.dinov2_vits14().cuda()
    # model = model.eval()
        
    transform = None
    
    transforms_list = [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    transform = transforms.Compose(transforms_list)
    
    transform = transform or make_classification_eval_transform()
    
    image1 = Image.open("/home/hqlab/workspace/closure/EE5346_2023_project/test_data/origin/test.jpg")
    image1 = transform(image1)[:3].unsqueeze(0).cuda()

    image2 = Image.open("/home/hqlab/workspace/closure/EE5346_2023_project/test_data/origin/test1.jpg")
    image2 = transform(image2)[:3].unsqueeze(0).cuda()

    with torch.no_grad():
        model.eval()
        features = model.forward_features(image1)['x_norm_patchtokens'][0].cpu()
        features_1 = model.forward_features(image2)['x_norm_patchtokens'][0].cpu()
        # patch_feature_1 = model(image2.to("cuda"))
        # patch_feature_1 = patch_feature_1.detach().cpu()
        # patch_feature_2 = model(image2.to("cuda"))
        # patch_feature_2 = patch_feature_2.detach().cpu()

    # print(torch.all(torch.eq(patch_feature_1, patch_feature_2)))
    # print(patch_feature_2.shape)

    pca = PCA(n_components=3)
    pca.fit(features)

    pca_features = pca.transform(features)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max()- pca_features .min())
    pca_features = pca_features * 255
    plt.imshow(pca_features.reshape(16,16,3).astype(np.uint8))
    plt.savefig('meta dog features 2.png')

    # patch_num = int(features.shape[0])
    # similarity = torch.zeros(size=( patch_num, patch_num), device='cpu')
    # patch_pairs = []  # [batch,pairs,[sim,patch1,patch2]]
    # for patch_i in range(patch_num):
    #     similarity[ patch_i, :] = torch.cosine_similarity(features[patch_i, :],
    #                                                                 features_1[ :, :], dim=-1)
    #     max_id = torch.argmax(similarity[ patch_i, :], dim=-1)
    #     patch_pairs.append([float(similarity[ patch_i, max_id]), patch_i, int(max_id)])
    # patch_pairs = sorted(patch_pairs, key=lambda x: x[0], reverse=True)
    # patch_pairs = patch_pairs[:10]

    # # draw the corresponding patches
    # draw_patch_corresponding_lines(image1[0].cpu(), [x[1] for x in patch_pairs], image2[0].cpu(),
    #                                 [x[2] for x in patch_pairs], patch_num)
    # sift_algo(image1[0].cpu(), image2[0].cpu())
 
    # tw.draw_model(model, [1, 3, 224, 224])
    # output:[1,1536]
    return 0

if __name__ == "__main__":
    description = "DINOv2 model output"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))