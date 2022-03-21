[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/convolutional-hough-matching-networks/semantic-correspondence-on-spair-71k)](https://paperswithcode.com/sota/semantic-correspondence-on-spair-71k?p=convolutional-hough-matching-networks)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/convolutional-hough-matching-networks/semantic-correspondence-on-pf-pascal)](https://paperswithcode.com/sota/semantic-correspondence-on-pf-pascal?p=convolutional-hough-matching-networks)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/convolutional-hough-matching-networks/semantic-correspondence-on-pf-willow)](https://paperswithcode.com/sota/semantic-correspondence-on-pf-willow?p=convolutional-hough-matching-networks)

# Convolutional Hough Matching Networks
Update 09/14/21: Our paper has been extended for journal submission [[link](https://arxiv.org/abs/2109.05221)]. The code will be updated soon.

This is the implementation of the paper "Convolutional Hough Matching Network" by J. Min and M. Cho. Implemented on Python 3.7 and PyTorch 1.3.1.

![](https://juhongm999.github.io/pic/chm.png)

For more information, check out project [[website](http://cvlab.postech.ac.kr/research/CHM/)] and the paper on [[arXiv](https://arxiv.org/abs/2103.16831)]

## Web Demo

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/taesiri/ConvolutionalHoughMatchingNetworks)

### Overall architecture:

![](https://juhongm999.github.io/pic/chm_architecture.png)

## Requirements

- Python 3.7
- PyTorch 1.3.1
- cuda 10.1
- pandas
- requests

Conda environment settings:
```bash
conda create -n chm python=3.7
conda activate chm

conda install pytorch=1.3.1 torchvision cudatoolkit=10.1 -c pytorch
conda install -c anaconda requests
conda install -c conda-forge tensorflow
pip install tensorboardX
conda install -c anaconda pandas
```

## Training	
The code provides three types of CHM kernel: position-sensitive isotropic (psi), isotropic (iso), vanilla Nd (full).
```bash
python train.py --ktype {psi, iso, full} 
                --benchmark {spair, pfpascal}
```

## Testing
Trained models are available on [[Google drive](https://drive.google.com/drive/folders/1myRkb2ow6ltAWaAJtJsh23NVB6d4YFuo?usp=sharing)].
```bash
python test.py --ktype {psi, iso, full} 
               --benchmark {spair, pfpascal, pfwillow} 
               --load 'path_to_trained_model'
```
For example, to reproduce our results in Table 1, refer following scripts.
```bash
python test.py --ktype psi --benchmark spair --load 'path_to_trained_model/spr_psi.pt'
python test.py --ktype psi --benchmark spair --load 'path_to_trained_model/pas_psi.pt'
python test.py --ktype psi --benchmark pfpascal --load 'path_to_trained_model/pas_psi.pt'
python test.py --ktype psi --benchmark pfwillow --load 'path_to_trained_model/pas_psi.pt'
```

## BibTeX
If you use this code for your research, please consider citing:
````BibTeX
@InProceedings{min2021chm, 
    author    = {Min, Juhong and Cho, Minsu},
    title     = {Convolutional Hough Matching Networks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {2940-2950}
}
````
