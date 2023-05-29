### use vits14 train
cd code
export PYTHONPATH=$PYTHONPATH:/home/hqlab/workspace/closure/dinov2/
python train.py --config-file ../dinov2/configs/eval/vits14_pretrain.yaml --pretrained-weights ../dinov2/weights/dinov2_vits14_pretrain.pth --output-dir ../result/test_pretrain/
python train.py --config-file ../dinov2/configs/eval/vitl14_pretrain.yaml --pretrained-weights ../dinov2/weights/dinov2_vitl14_pretrain.pth --output-dir ../result/test_pretrain/
<!-- python train.py --config-file ../dinov2/configs/eval/vitg14_pretrain.yaml --pretrained-weights ../dinov2/weights/dinov2_vitg14_pretrain.pth --output-dir ../result/test_pretrain/ -->

python train.py --config-file ../dinov2/configs/eval/vitl14_pretrain.yaml --pretrained-weights ../dinov2/weights/dinov2_vitl14_pretrain.pth --output-dir ../result/test_tensorboard --split --train-path /home/hqlab/workspace/closure/EE5346_2023_project/