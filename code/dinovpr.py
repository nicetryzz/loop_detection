import argparse
import logging
import os
import sys
import csv
from typing import List, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import AveragePrecision
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, PrecisionRecallCurve

from pair_loader import get_data_loader, get_without_label_data_loader
from model import MLP
from util import EarlyStopper

from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model

logger = logging.getLogger("dinov2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    parser.add_argument(
        '--test', 
        action='store_true',
        help="Test the pretrained MLP",)
    parser.add_argument(
        "--train-path",
        type=str,
        help="Train data path.",
    )
    parser.add_argument(
        "--valid-path",
        type=str,
        help="Validation data path.",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        help="Test data path.",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        help="Test spcific file path.",
    )
    parser.add_argument(
        "--MLP-weight-path",
        type=str,
        help="Pretrained MLP weight path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--split",
        action='store_true',
        help="Need split train-val dataset.",
    )
    parser.add_argument(
        "--use-pretrain-MLP",
        action='store_true',
        help="Use pretrain MLP.",
    )
    parser.add_argument(
        '--test-without-label', 
        action='store_true',
        help="Test the pretrained MLP and output label",)
    parser.add_argument(
        "--output-label-path",
        type=str,
        help="Output label path.",)
    parser.set_defaults(
        train_path="/home/hqlab/workspace/closure/dataset/robotcar_dataset/Downloads/",
        valid_path="/home3/hqlab/chenqilong/EE5346_2023_project/",
        test_path="/home3/hqlab/chenqilong/ee5346_dataset/",
        test_file="/home3/hqlab/chenqilong/EE5346_2023_project/eval/robotcar_qAutumn_dbSunCloud_val_final.txt",
        MLP_weight_path="../result/learn_back_small_continue/best.pth",
        batch_size=6,
    )
    return parser

def test_MLP(args):
    model, _= setup_and_build_model(args)
    writer = SummaryWriter(args.output_dir+'/runs/testing')
    
    batch_size = args.batch_size
    test_path=args.test_path
    weight_path = args.MLP_weight_path

    model_back = MLP().to(device)
    model_weight = torch.load(weight_path)

    model= nn.DataParallel(model)
    model_back= nn.DataParallel(model_back)

    model_back.load_state_dict(model_weight['model_back_state_dict'])

    if args.test_without_label:
        test_file = args.test_file
        valid_loader = get_without_label_data_loader(test_path,test_file,batch_size,need_translate = False)
        write_path = test_file[:-9]+"result.txt"
        eval_MLP_withoutlabel(model,model_back,valid_loader, write_path)
    else:
        writer = SummaryWriter(args.output_dir+'/runs/testing')
        valid_loader = get_data_loader(test_path,batch_size,split=False,need_translate = False)
        criterion = nn.CosineEmbeddingLoss()
        
        if 'model_state_dict' in model_weight:
            model.load_state_dict(model_weight['model_state_dict'])
            
        eval_MLP(model,model_back,valid_loader,criterion,0,writer,True)
        writer.close()
        return 0


def train_MLP(args):
    model, _ = setup_and_build_model(args)
    writer = SummaryWriter(args.output_dir+'/runs/training')
    batch_size = args.batch_size
    num_epochs = 1000
    learning_rate = 1e-3
    min_loss = 10

    if args.split:
        train_path =args.train_path
        train_loader,valid_loader = get_data_loader(train_path,batch_size,split=True,need_translate = False)
    else:
        train_path=args.train_path
        valid_path=args.valid_path
        train_loader = get_data_loader(train_path,batch_size,split=False,need_translate = True)
        valid_loader = get_data_loader(valid_path,batch_size,split=False,need_translate = False)

    model_back = MLP().to(device)

    if args.use_pretrain_MLP:
        weight_path = args.MLP_weight_path
        model_weight = torch.load(weight_path)
        model.load_state_dict(model_weight['model_state_dict'])
        model_back.load_state_dict(model_weight['model_back_state_dict'])

    model= nn.DataParallel(model)
    model_back= nn.DataParallel(model_back)

    criterion = nn.CosineEmbeddingLoss()
    optimizer_params = [{"params": model_back.parameters(), "lr": learning_rate},]
                        # {"params": model.parameters(), "lr": 1e-2*learning_rate},]
    optimizer = optim.SGD(optimizer_params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
    early_stopper = EarlyStopper(patience=5, min_delta=0.01)

    for epoch in range(num_epochs):
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning rate', lr, epoch)

        train_total_loss = 0
        total_num = 0
        for i, data in enumerate(train_loader):
            img1, img2, labels = data 
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            labels = labels*2 -1
            optimizer.zero_grad()
            
            inputs1 = model(img1)
            inputs2 = model(img2)
            embeddings1 = model_back(inputs1)
            embeddings2 = model_back(inputs2)
                        
            loss = criterion(embeddings1, embeddings2, labels)
            train_total_loss += loss.item() * embeddings1.size(0)
            total_num += embeddings1.size(0)
            loss.backward()
            optimizer.step()
        
        train_mean_loss = train_total_loss / total_num
        writer.add_scalar('training loss', train_mean_loss, epoch)

        val_mean_loss = eval_MLP(model,model_back,valid_loader,criterion,epoch,writer,False)
        
        if val_mean_loss < min_loss:
            model_path = args.output_dir + "/best.pth"
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'model_back_state_dict': model_back.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_mean_loss,
            }, model_path)

        scheduler.step(val_mean_loss)
        if early_stopper.early_stop(val_mean_loss):
            break
    logger.info('Finished Training')
    writer.close()
    return 0


def eval_MLP(model,model_back,valid_loader,criterion,epoch,writer,save_pr_curve):
    total_loss = 0.0
    total_num = 0
    ap = AveragePrecision(task="binary").to(device)
    accuracy = BinaryAccuracy(threshold=0.5).to(device)
    precision = BinaryPrecision(threshold=0.5).to(device)
    recall = BinaryRecall(threshold=0.5).to(device)

    with torch.no_grad():
        prediction_list = []
        label_list = []
        for i, data in enumerate(valid_loader):
            img1, img2, labels = data
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            label_list.append(labels)
            labels = labels*2 -1
            
            inputs1 = model(img1)
            inputs2 = model(img2)
            embeddings1 = model_back(inputs1)
            embeddings2 = model_back(inputs2)
            
            sim = nn.functional.cosine_similarity(embeddings1, embeddings2)
            preds = torch.where(sim < 0, torch.zeros_like(sim), sim)
            prediction_list.append(preds)
            loss = criterion(embeddings1, embeddings2, labels)

            total_loss += loss.item() * embeddings1.size(0)
            total_num += embeddings1.size(0)

    predictions = torch.cat(prediction_list)
    labels = torch.cat(label_list)
    
    ap.update(predictions, labels)
    
    ap_score = ap.compute()
    acc_score = accuracy(predictions, labels)
    prec_score = precision(predictions, labels)
    rec_score = recall(predictions, labels)
    
    writer.add_pr_curve('PR-curve', labels, predictions, global_step=epoch)

    mean_loss = total_loss / total_num

    writer.add_scalar('validation loss', mean_loss, epoch)
    writer.add_scalar('validation average precision', ap_score, epoch)
    writer.add_scalar('validation precision', prec_score, epoch)
    writer.add_scalar('validation recall', rec_score, epoch)
    writer.add_scalar('validation accuracy', acc_score, epoch)

    if save_pr_curve:
        pr_curve = PrecisionRecallCurve(task="binary").to(device)
        P, R, T = pr_curve(predictions, labels)
        with open(args.output_dir + '/pr_curve.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['precision', 'recall'])
            for p, r in zip(P, R):
                writer.writerow([p.item(), r.item()])

    logger.info(f"epoch:{epoch},Test Loss: {mean_loss:.4f}, average precision: {ap_score.item():.4f}")
    return mean_loss


def eval_MLP_withoutlabel(model,model_back,valid_loader, write_path):
    prediction_list = []
    img1_list = []
    img2_list = []
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            img1, img2, img1_path, img2_path = data
            img1 = img1.to(device)
            img2 = img2.to(device)
            img1_list.append(img1_path)
            img2_list.append(img2_path)
            
            inputs1 = model(img1)
            inputs2 = model(img2)
            embeddings1 = model_back(inputs1)
            embeddings2 = model_back(inputs2)
            
            sim = nn.functional.cosine_similarity(embeddings1, embeddings2)
            preds = torch.where(sim < 0.5, torch.zeros_like(sim), torch.ones_like(sim))
            prediction_list.append(preds)

    with open(write_path, 'w') as f:
        for i in range(len(prediction_list)):
            for j in range(len(prediction_list[i])):
                f.write(str(int(prediction_list[i][j].item()))+"\n")


def main(args):
    if args.test | args.test_without_label:
        test_MLP(args)
    else:
        train_MLP(args) 


if __name__ == "__main__":
    description = "DINOv2 based vpr"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))