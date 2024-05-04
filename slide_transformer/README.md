# Slide Transformer

This folder contains the implementation of Slide Transformer based on PVT, Swin models for image classification.

## Dependencies

- Python 3.9
- PyTorch == 1.11.0
- torchvision == 0.12.0
- numpy
- timm == 0.4.12
- einops
- yacs

## Data preparation

The ImageNet dataset should be prepared as follows:

```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```


## Train Models from Scratch

- To train `Slide-PVT/Slide-Swin` on ImageNet from scratch, run:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg <path-to-config-file> --data-path <imagenet-path> --output <output-path>
```



## Citation

If you find this repo helpful, please consider citing us.

```latex
@inproceedings{pan2023slide,
  title={Slide-transformer: Hierarchical vision transformer with local self-attention},
  author={Pan, Xuran and Ye, Tianzhu and Xia, Zhuofan and Song, Shiji and Huang, Gao},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={2082--2091},
  year={2023}
}
```
