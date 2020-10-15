# MCN-MT

## Learning From Self-Discrepancy via Multiple Co-Teaching for Cross-Domain Person Re-Identification (MCN-MT)

<img src='images/MCN-MT.png'/>

The official implementation of MCN-MT in PyTorch, and our paper is submitted to ICASSP 2021, it is available at [http://arxiv.org/abs/2020.00000](http://arxiv.org/abs/2020.00000).

### News
- Support Market1501, DukeMTMC-reID and CUHK03 datasets.
- The current version supports training on multi-GPUs.


### TODO
Write the documents.

### Requirements
- Python3
- Numpy==1.16.4
- Matplotlib==3.1.1
- Torch==1.3.1
- Metric_learn==0.4.0
- tqdm==4.32.2
- torchvision==0.2.0
- scipy==1.1.0
- h5py==2.9.0
- Pillow==6.2.1
- six==1.13.0
- scikit_learn==0.21.3

### How to use it?
This repo. supports training on multiple GPUs and the default setting is also multi-GPU.

1. Download all necessry datasets, e.g. DukeMTMC-reID, Market-1501 and CUHK03 datasets and move them to 'data'.  

2. Before performing training from scratch, please download all models ([Baidu NetDisk](https://pan.baidu.com/s/1WcjIyvoW8RI-FrEdsnLVgg), Password: 102s) pretrained on DukeMTMC-reID and Market-1501, and then move them in the 'MCN-MT'

3. If you want to restart the train process using MCN with 3 models when trained on DukeMTMC-reID, while tested on Market-1501, the command you can type as
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python selftrainingACT_3model.py --src_dataset dukemtmc --tgt_dataset market1501 --resume ./MCN_pretrain/Duke/Duke2Market.pth --data_dir ./data --logs_dir ./logs/dukemar_3model
```

If you want to restart the train process using MCN-MT (with meannet) with 3 models when trained on DukeMTMC-reID, while tested on Market-1501, the command you can type as
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python selftrainingACT_3model_meannet.py --src_dataset dukemtmc --tgt_dataset market1501 --resume ./MCN_pretrain/Duke/Duke2Market.pth --data_dir ./data --logs_dir ./logs/dukemar_3model_meannet
```

### Experiment Results on DukeMTMC-reID and Market-1501 datasets.
<img src='images/results.png'/>

### Acknowledgments
This work was supported by the National Natural Science Foundation of China under Project(Grant No.61977045).
If you find this code useful in your research, please consider citing：
```
@article{xiang2020learning,
    title={Learning From Self-Discrepancy via Multiple Co-Teaching for Cross-Domain Person Re-Identification},
    author={Suncheng Xiang and Yuzhuo Fu and Mengyuan Guan and Ling Liu},
    year={2020},
    journal={arXiv preprint arXiv:2020.00000}
}
```