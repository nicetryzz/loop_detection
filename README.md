# DINOVPR

DINOVPR is a VPR(visual position recognization) based on [DINOv2](https://github.com/facebookresearch/dinov2). It use DINOv2 as backbone and learn a simple MLP to fetch low level global feature. It calculate the cosine similarity of global feature to check global closure.

## Installation

The training and evaluation code requires PyTorch 2.0 and [xFormers](https://github.com/facebookresearch/xformers) 0.0.18 as well as a number of other 3rd party packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup all the required dependencies for training and evaluation, please follow the instructions below:

*[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)* **(Recommended)** - Clone the repository and then create and activate a `dinov2` conda environment using the provided environment definition:

```shell
conda env create -f conda.yaml
conda activate dinovpr
```

## Pretrain model

We have trained the [MLP model]()(upload later) on dataset below.

You also can use pretrain [DINOv2 model](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) to train your model.

## Data preparation

We test the data given by [SLAM course](https://github.com/MedlarTea/EE5346_2023_project), which labeled from the Oxford RobotCar dataset.

### trainging dataset formate

The root directory of the dataset should hold the following contents:

- `<ROOT>/Autumn_mini_query/`
- `<ROOT>/Kudamm_mini_query/`
- `<ROOT>/Kudamm_mini_ref/`
- `<ROOT>/Night_mini_ref/`
- `<ROOT>/Suncloud_mini_ref/`
- `<ROOT>/Kudamm_diff_final.txt`
- `<ROOT>/[...]`
- `<ROOT>/robotcar_qAutumn_dbSunCloud_easy_final.txt`

It only can contain training .txt file in the `<ROOT>` folder.
The train file need contain both two image path and label, which seperated by comma.

*warning: To execute the commands provided in the next sections for training and evaluation, the `dinov2` package should be included in the Python module search path, i.e. simply prefix the command to run with `PYTHONPATH=.`.*


## Training

Run DINOVPR training on Tesla P100 nodes (8 GPUs):

```shell
python code/dinovpr.py --config-file ./dinov2/configs/eval/vitl14_pretrain.yaml --pretrained-weights ./dinov2/weights/dinov2_vitl14_pretrain.pth --output-dir ./result/test_tensorboard --split --train-path <PATH/TO/DATASET> --batch-size 32
```

If you use pretrainde MLP, you can edit `--use-pretrain-MLP` and `--MLP-weight-path`.

## Evaluation(upload later)

