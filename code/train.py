import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional
from torchvision import transforms

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

from pair_loader import get_data_loader
from model import MLP
from loss import ContrastiveLoss

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

def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    # Define hyperparameters
    batch_size = 4
    num_epochs = 200
    learning_rate = 1e-4

    folder_path="/home/hqlab/workspace/closure/EE5346_2023_project/"

    train_loader, valid_loader = get_data_loader(folder_path,batch_size)

    # Create an instance of the model
    model_back = MLP().cuda()

    # Define the loss function and the optimizer
    # criterion = ContrastiveLoss(margin=2)
    criterion = nn.CosineEmbeddingLoss(margin=0.5)
    optimizer = optim.SGD(model_back.parameters(), lr=learning_rate) # optimizer is stochastic gradient descent

    # Train the model using metric learning with pairs of images
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            img1, img2, labels = data # get the first batch of inputs from the data loader
            img1 = img1.cuda()
            img2 = img2.cuda()
            labels = labels.cuda()
            labels = labels*2 -1
            optimizer.zero_grad() # zero the parameter gradients
            inputs1 = model(img1)
            inputs2 = model(img2)
            
            # Create pairs of embeddings and labels based on whether they have the same label or not
            embeddings1 = model_back(inputs1) # forward pass to get embeddings for the first batch
            embeddings2 = model_back(inputs2) # forward pass to get embeddings for the second batch
            
            loss = criterion(embeddings1, embeddings2, labels) # 计算损失

            loss.backward() # 反向传播
            optimizer.step() # 更新参数

            running_loss += loss.item()
            # if i % 200 == 199: # print statistics every 200 mini-batches
            #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            #     running_loss = 0.0
                
        
        total_loss = 0.0 # 记录总损失
        total_acc = 0.0 # 记录总准确率
        total_num = 0 # 记录总样本数
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        max_thre = 0

        with torch.no_grad(): # 不计算梯度
            
            for i, data in enumerate(valid_loader):
                img1, img2, labels = data # get the first batch of inputs from the data loader
                img1 = img1.cuda()
                img2 = img2.cuda()
                labels = labels.cuda()
                optimizer.zero_grad() # zero the parameter gradients
                inputs1 = model(img1)
                inputs2 = model(img2)
            
                # Create pairs of embeddings and labels based on whether they have the same label or not
                embeddings1 = model_back(inputs1) # forward pass to get embeddings for the first batch
                embeddings2 = model_back(inputs2) # forward pass to get embeddings for the second batch
                sim = nn.functional.cosine_similarity(embeddings1, embeddings2)
                # Compare the similarity with the threshold and the target
                mask = torch.eq(labels,0)
                masked_sim = torch.masked_select(sim, mask)
                if torch.numel(masked_sim) != 0:
                    max_value = torch.max(masked_sim)
                    max_thre = max(max_value,max_thre)
            
            for i, data in enumerate(valid_loader):
                img1, img2, labels = data # get the first batch of inputs from the data loader
                img1 = img1.cuda()
                img2 = img2.cuda()
                labels = labels.cuda()
                labels = labels*2 -1
                optimizer.zero_grad() # zero the parameter gradients
                inputs1 = model(img1)
                inputs2 = model(img2)
            
                # Create pairs of embeddings and labels based on whether they have the same label or not
                embeddings1 = model_back(inputs1) # forward pass to get embeddings for the first batch
                embeddings2 = model_back(inputs2) # forward pass to get embeddings for the second batch
                sim = nn.functional.cosine_similarity(embeddings1, embeddings2)
                pred = torch.where(sim > max_thre, torch.tensor(1), torch.tensor(-1))
                loss = criterion(embeddings1, embeddings2, labels) # 计算损失
                acc = (pred == labels).float().mean() # 计算准确率
                
                tp += torch.sum((pred == 1) & (labels == 1)).item()
                tn += torch.sum((pred == -1) & (labels == -1)).item()
                fp += torch.sum((pred == 1) & (labels == -1)).item()
                fn += torch.sum((pred == -1) & (labels == 1)).item()

                total_loss += loss.item() * embeddings1.size(0) # 累加损失
                total_acc += acc.item() * embeddings1.size(0) # 累加准确率
                total_num += embeddings1.size(0) # 累加样本数

        mean_loss = total_loss / total_num # 计算平均损失
        mean_acc = total_acc / total_num # 计算平均准确率
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        print(f"epoch:{epoch},Test Loss: {mean_loss:.4f}, Test Accuracy: {mean_acc:.4f}, threshold:{max_thre:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}") # 打印结果

    print('Finished Training')
    return 0

if __name__ == "__main__":
    description = "DINOv2 model patch matching"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))