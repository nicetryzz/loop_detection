### use vits14 train
cd code
export PYTHONPATH=$PYTHONPATH:/home/hqlab/workspace/closure/dinov2/
python train.py --config-file ../dinov2/configs/eval/vits14_pretrain.yaml --pretrained-weights ../dinov2/weights/dinov2_vits14_pretrain.pth --output-dir ../result/test_pretrain/
python train.py --config-file ../dinov2/configs/eval/vitl14_pretrain.yaml --pretrained-weights ../dinov2/weights/dinov2_vitl14_pretrain.pth --output-dir ../result/test_pretrain/
<!-- python train.py --config-file ../dinov2/configs/eval/vitg14_pretrain.yaml --pretrained-weights ../dinov2/weights/dinov2_vitg14_pretrain.pth --output-dir ../result/test_pretrain/ -->

python train.py --config-file ../dinov2/configs/eval/vitl14_pretrain.yaml --pretrained-weights ../dinov2/weights/dinov2_vitl14_pretrain.pth --output-dir ../result/test_tensorboard --split --train-path /home/hqlab/workspace/closure/EE5346_2023_project/

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

python train.py --config-file ../dinov2/configs/eval/vitl14_pretrain.yaml --pretrained-weights ../dinov2/weights/dinov2_vitl14_pretrain.pth --output-dir ../result/learn_back_small --split --train-path /home3/hqlab/chenqilong/EE5346_2023_project --batch-size 32

python train.py --config-file ../dinov2/configs/eval/vitl14_pretrain.yaml --pretrained-weights ../dinov2/weights/dinov2_vitl14_pretrain.pth --output-dir ../result/learn_back_large --train-path /home3/hqlab/chenqilong/robot_car/modified --valid-path /home3/hqlab/chenqilong/EE5346_2023_project --batch-size 32