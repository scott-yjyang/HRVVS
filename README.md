# HRVVS

<div align="center">
<br>
<h3>[MICCAI 2025] HRVVS: A High-resolution Video Vasculature Segmentation Network via Hierarchical Autoregressive Residual Priors</h3>

[Xincheng Yao](https://yijun-yang.github.io/)<sup>1</sup>&nbsp;
[Yijun Yang](https://yijun-yang.github.io/)<sup>1,*</sup>&nbsp;
[Kangwei Guo](https://github.com/scott-yjyang/HRVVS)<sup>1</sup>&nbsp;
[Ruiqiang Xiao](https://keeplearning-again.github.io/)<sup>1</sup>&nbsp;
[Haipeng Zhou](https://haipengzhou856.github.io/)<sup>1</sup>&nbsp;
[Haisu Tao](https://github.com/scott-yjyang/HRVVS)<sup>2</sup>&nbsp;
[Jian Yang](https://github.com/scott-yjyang/HRVVS)<sup>2</sup><br>
[Lei Zhu](https://sites.google.com/site/indexlzhu/home)<sup>1,3</sup>&nbsp;

<sup>1</sup> The Hong Kong University of Science and Technology (Guangzhou) &nbsp; <sup>2</sup> Southern Medical University &nbsp; <sup>3</sup> The Hong Kong University of Science and Technology  <sup>*</sup> Project Lead

<p align="center">
<!--   <a href="https://yijun-yang.github.io/MeWM/"><img src="https://img.shields.io/badge/project-page-red" alt="Project Page"></a> -->
  <a href="https://arxiv.org/abs/2507.22530"><img src="https://img.shields.io/badge/ArXiv-<2507.22530>-<COLOR>.svg" alt="arXiv"></a>
<!--   ![visitors](https://visitor-badge.laobi.icu/badge?page_id=scott-yjyang/HRVVS) -->
<!--   <a href="https://huggingface.co/papers/2506.02327"><img src="https://img.shields.io/badge/huggingface-page-yellow.svg" alt="huggingface"></a> -->
 <p align="center">
  
</div>

## Abstract
The segmentation of the hepatic vasculature in surgical videos holds substantial clinical significance in the context of hepatectomy procedures. However, owing to the dearth of an appropriate dataset and the inherently complex task characteristics, few researches have been reported in this domain. To address this issue, we first introduce a high quality frame-by-frame annotated hepatic vasculature dataset containing 35 long hepatectomy videos and 11442 high-resolution frames. On this basis, we propose a novel high-resolution video vasculature segmentation network, dubbed as HRVVS. We innovatively embed a pretrained visual autoregressive modeling (VAR) model into different layers of the hierarchical encoder as prior information to reduce the information degradation generated during the downsampling process. In addition, we designed a dynamic memory decoder on a multi-view segmentation network to minimize the transmission of redundant information while preserving more details between frames. Extensive experiments on surgical video datasets demonstrate that our proposed HRVVS significantly outperforms the state-of-the-art methods.

## Environment
```bash
conda create -n HRVVS python
conda activate HRVVS
pip install -r requirements.txt
```

## Dataset
The Hepa-SEG dataset is undergoing qualification review in hospital.

## Checkpoints
```bash
mkdir checkpoints && cd checkpoints
wget https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth
wget https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth
gdown https://drive.google.com/uc?id=1FixkCY4KOlZ0gae_7Uf9zhAG4sKDFZBV
```

## Train & Test
### Train
```bash
python train.py
```

### Test
```bash
python test.py
```

## Acknowledgement
We thank the great work from [VAR](https://github.com/FoundationVision/VAR).
