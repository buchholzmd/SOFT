# Softmax-free Linear Transformers

![image](resources/structure.png)

> [**Softmax-free Linear Transformers**](https://arxiv.org/abs/2207.03341),            
> Jiachen Lu, Li Zhang, Junge Zhang, Xiatian Zhu, Hang Xu, Jianfeng Feng        

## What's new
1. We propose a normalized softmax-free self-attention with stronger generalizability.
2. SOFT is now avaliable on more vision tasks (object detection and semantic segmentation).


## Requirments
* timm==0.3.2

* torch>=1.7.0 and torchvision that matches the PyTorch installation

* cuda>=10.2

Compilation may be fail on cuda < 10.2.  
We have compiled it successfully on `cuda 10.2` and `cuda 11.2`. 

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```
## Installation
```shell script
git clone https://github.com/fudan-zvg/SOFT.git
python -m pip install -e SOFT
```

## Main results
### ImageNet-1K Image Classification

| Model       | Resolution | Params | FLOPs | Top-1 % | Config |Pretrained Model|
|-------------|:----------:|:------:|:-----:|:-------:|--------|--------
| SOFT-Tiny-Norm   | 224        | 13M    | 1.9G  | 79.4    |[SOFT_Tiny_norm.yaml](config/SOFT_Tiny_norm.yaml)|[SOFT_Tiny_norm](https://drive.google.com/file/d/1Isy5b9v_4pyIXDqhKPNRq3WKH0etDlfl/view?usp=sharing)
| SOFT-Small-Norm  | 224        | 24M    | 3.3G  | 82.4    |[SOFT_Small_norm.yaml](config/SOFT_Small_norm.yaml)|
| SOFT-Medium-Norm | 224        | 45M    | 7.2G  | 83.1    |[SOFT_Meidum_norm.yaml](config/SOFT_Medium_norm.yaml)|
| SOFT-Large-Norm  | 224        | 64M    | 11.0G | 83.3    |[SOFT_Large_norm.yaml](config/SOFT_Large_norm.yaml)|
| SOFT-Huge-Norm   | 224        | 87M    | 16.3G | 83.4    |[SOFT_Huge_norm.yaml](config/SOFT_Huge_norm.yaml)|

### COCO Object Detection (2017 val)
| Backbone     | Method | lr schd | box mAP | mask mAP | Params |
|-------------|:----------:|:------:|:-----:|:-------:|:--------:|
|SOFT-Tiny-Norm | RetinaNet | 1x | 40.0 | - | 23M|
|SOFT-Tiny-Norm | Mask R-CNN | 1x | 41.2 | 38.2 | 33M|
|SOFT-Small-Norm | RetinaNet | 1x | 42.8 | - | 34M|
|SOFT-Small-Norm | Mask R-CNN | 1x | 43.8 | 40.1 | 44M|
|SOFT-Medium-Norm | RetinaNet | 1x | 44.3 | - | 55M|
|SOFT-Medium-Norm | Mask R-CNN | 1x | 46.6 | 42.0 | 65M|
|SOFT-Large-Norm | RetinaNet | 1x | 45.3 | - | 74M|
|SOFT-Large-Norm | Mask R-CNN | 1x | 47.0 | 42.2 | 84M|

### ADE20K Semantic Segmentation (val)
| Backbone     | Method | Crop size| lr schd | mIoU | Params |
|-------------|:----------:|:----------:|:------:|:-----:|:-------:|
|SOFT-Small-Norm | UperNet |512x512| 1x | 46.2 | 54M|
|SOFT-Medium-Norm | UperNet |512x512 | 1x | 48.0 | 76M|
## Get Started

### Train
We have two implementations of Gaussian Kernel: `PyTorch` version and 
the exact form of Gaussian function implemented by `cuda`. The config file containing `cuda` is the 
cuda implementation. Both implementations yield same performance. 
Please **install** SOFT before running the `cuda` version. 
```shell
./dist_train.sh ${GPU_NUM} --data ${DATA_PATH} --config ${CONFIG_FILE}
# For example, train SOFT-Tiny on Imagenet training dataset with 8 GPUs
./dist_train.sh 8 --data ${DATA_PATH} --config config/SOFT_Tiny.yaml
```

### Test

```shell

./dist_train.sh ${GPU_NUM} --data ${DATA_PATH} --config ${CONFIG_FILE} --eval_checkpoint ${CHECKPOINT_FILE} --eval

# For example, test SOFT-Tiny on Imagenet validation dataset with 8 GPUs

./dist_train.sh 8 --data ${DATA_PATH} --config config/SOFT_Tiny.yaml --eval_checkpoint ${CHECKPOINT_FILE} --eval

```
## Reference

```bibtex
@inproceedings{SOFT,
    title={SOFT: Softmax-free Transformer with Linear Complexity}, 
    author={Lu, Jiachen and Yao, Jinghan and Zhang, Junge and Zhu, Xiatian and Xu, Hang and Gao, Weiguo and Xu, Chunjing and Xiang, Tao and Zhang, Li},
    booktitle={NeurIPS},
    year={2021}
}
```

## License

[MIT](LICENSE)


## Acknowledgement

Thanks to previous open-sourced repo:  
[Detectron2](https://github.com/facebookresearch/detectron2)  
[T2T-ViT](https://github.com/yitu-opensource/T2T-ViT)  
[PVT](https://github.com/whai362/PVT)   
[Nystromformer](https://github.com/mlpen/Nystromformer)   
[pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
