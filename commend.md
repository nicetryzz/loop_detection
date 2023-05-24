### use vits14 train
cd code
python train.py --config-file ../dinov2/configs/eval/vits14_pretrain.yaml --pretrained-weights ../dinov2/weights/dinov2_vits14_pretrain.pth --output-dir ../dinov2/result/test_pretrain/
python train.py --config-file ../dinov2/configs/eval/vitl14_pretrain.yaml --pretrained-weights ../dinov2/weights/dinov2_vitl14_pretrain.pth --output-dir ../dinov2/result/test_pretrain/
python train.py --config-file ../dinov2/configs/eval/vitg14_pretrain.yaml --pretrained-weights ../dinov2/weights/dinov2_vitg14_pretrain.pth --output-dir ../dinov2/result/test_pretrain/