# TopNet: Transformer-Efficient Occupancy Prediction Network for Octree-Structured Point Cloud Geometry Compression. CVPR 2025 [Paper](https://ieeexplore.ieee.org/document/11093721)


## Abstract

Efficient Point Cloud Geometry Compression (PCGC) with a lower bits per point (BPP) and higher peak signal-to-noise ratio (PSNR) is essential for the transportation of large-scale 3D data. Although octree-based entropy models can reduce BPP without introducing geometry distortion, existing CNN-based models struggle with limited receptive fields to capture long-range dependencies, while Transformer-built architectures always neglect fine-grained details due to their reliance on global self-attention. In this paper, we propose a Transformer-efficient occupancy prediction Network, termed TopNet, to overcome these challenges by developing several novel components: Locally-enhanced Context Encoding (LeCE) for enhancing the translation-invariance of the octree nodes, Adaptive-Length Sliding Window Attention (AL-SWA) for capturing both global and local dependencies while adaptively adjusting attention weights based on the input window length, Spatial-Gated-enhanced Channel Mixer (SG-CM) for efficient feature aggregation from ancestors and siblings, and Latent-guided Node Occupancy Predictor (LNOP) for improving prediction accuracy of spatially adjacent octree nodes. Comprehensive experiments across both indoor and outdoor point cloud datasets demonstrate that our TopNet achieves state-of-the-art performance with fewer parameters, further advancing the reduction-efficiency boundaries of PCGC.

## Requirements
- python 3.7
- PyTorch 1.9.0+cu102
- `file/environment.sh` to help you build this environment

## Download and Prepare Training and Testing Data
- ### Download data
    ### For LiDAR compression
    
    [SemanticKITTI](http://www.semantic-kitti.org/dataset.html) (80G)  
    23201/20351 frames in 00-10/11-21 folders for training/testing. 

    ### For Object compression

    [MPEG 8iVFBv2](http://plenodb.jpeg.org/pc/8ilabs)  (5.5GB)  
    300/300 frames in soldier10 and longdress10 for training.  
    300/300 frames in loot10 and redandblack10 for testing. 

    [JPEG MVUB](http://plenodb.jpeg.org/pc/microsoft) (8GB)  
    318/216/207 frames in andrew10, david10 and sarah10 for training.  
    245/245/216/216 frames in Phil9/10 and Ricardo9/10 for testing.  
    (Note: We rotated the MVUB data to make it consistent with MPEG 8i. Please set `rotation=True` in the `dataPrepare` function when processing MVUB data in training and testing.)

- ### Prepare data
Please set `oriDir` in `dataPrepare.py` before. 
```
python dataPrepare.py
```
To prepare train and test data. It will generate `*.mat` data in the directory `Data`.  
    
## Train
```
python TopNet.py 
```
You should set the Network parameters `expName,DataRoot`etc. in `networkTool.py`.
This will output checkpoint in `expName` folder, e.g. `Exp/Kitti`. (Note: You should run `DataFolder.calcdataLenPerFile()` in `dataset.py` for a new dataset, and you can comment it after you get the parameter `dataLenPerFile`)

## Encode and Decode
You may need to run the following command to provide `pc_error` and `tmc13v14` execute permission.
```
chmod +x file/pc_error file/tmc13v14 
``` 
- ### Encode
```
python encoder.py  
```
This will output binary codes saved in `.bin` format in `Exp(expName)/data`, and will generate `*.mat` data in the directory `Data/testPly`.

- ### Decode
```
python decoder.py 
```
This will load `*.mat` data for check and calculate PSNR by `pc_error`.

## Test TMC
We provide the test code for [TMC13](https://github.com/MPEGGroup/mpeg-pcc-tmc13) v14 (G-PCC) for Object and LiDAR point cloud compression.
```
python testTMC.py
```

## Citation

If you find this repository useful in your research, please consider giving a star ⭐ and a citation
```bibtex
@InProceedings{Wang_2025_CVPR,
    author    = {Wang, Xinjie and Zhang, Yifan and Liu, Ting and Liu, Xinpu and Xu, Ke and Wan, Jianwei and Guo, Yulan and Wang, Hanyun},
    title     = {TopNet: Transformer-Efficient Occupancy Prediction Network for Octree-Structured Point Cloud Geometry Compression},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {27305-27314}
}
```

## Acknowledgments

This code borrows heavily from [OctAttention](https://github.com/zb12138/OctAttention).
