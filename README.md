<p align="center">
  <h1 align="center">A Unified SAM-Guided Self-Prompt Learning Framework for Infrared Small Target Detection (TGRS' 2025)</h1>
  <p align="center">
    <a href="https://github.com/fuyimin96"><strong>Yimin Fu</strong></a>&nbsp;&nbsp;
    <a href="https://github.com/jialinlvcn"><strong>Jialin Lyu</strong></a>&nbsp;&nbsp;
    <strong>Peiyuan Ma</strong></a>&nbsp;&nbsp;
    <strong>Zhunga Liu</strong></a>&nbsp;&nbsp;
    <strong>Michael K. Ng</strong></a>
  </p>
  <br>

Pytorch implementation for "[**A Unified SAM-Guided Self-Prompt Learning Framework for Infrared Small Target Detection**](https://ieeexplore.ieee.org/document/11172325)"

> **Abstract:** Infrared small target detection (ISTD) aims to precisely capture the location and morphology of small targets under all-weather conditions. Compared with generic objects, infrared targets in remote fields of view are smaller in size and exhibit lower signal-to-clutter ratios. This poses a significant challenge in simultaneously preserving low-level target details and understanding high-level contextual semantics, forcing a trade-off between reducing miss detection and suppressing false alarms. In addition, most existing ISTD methods are designed for specific target types under certain infrared platforms, rather than as a unified framework broadly applicable across diverse infrared sensing scenarios. To address these challenges, we propose a unified self-prompt learning framework for ISTD under the guidance of the Segment Anything Model (SAM). Specifically, the model is incorporated with SAM in the encoding stage through a consult-guide manner, adapting the general knowledge to facilitate task-specific contextual understanding. Then, shallow-layer features are employed to generate self-derived prompts, which bidirectionally interact with encoded latent representations to complement subtle low-level details. Moreover, the semantic inconsistency during resolution recovery is mitigated by integrating a mutual calibration module into skip connections, ensuring coherent spatial-semantic fusion. Extensive experiments are conducted on four public ISTD datasets, and the results demonstrate that the proposed method consistently achieves superior performance across different infrared sensing platforms and target types.

<p align="center">
    <img src=./assets/sam-spl.png width="800">
</p>

## Requirements
To install the requirements, you can run the following in your environment first:
```bash
pip install -r requirements.txt
```
To run the code with CUDA properly, you can comment out `torch` and `torchvision` in `requirement.txt`, and install the appropriate version of `torch` and `torchvision` according to the instructions on [PyTorch](https://pytorch.org/get-started/locally/).

Or you can use `uv` to install dependencies:
```bash
uv sync
```

## Datasets
For the dataset used in this paper, please download the following datasets [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection) / [IRSTD-1k](https://github.com/RuiZhang97/ISNet) / [IRSTDID-SKY](https://github.com/xdFai/IRSTDID-800) / [NUDT-Sea](https://github.com/TianhaoWu16/Multi-level-TransUNet-for-Space-based-Infrared-Tiny-ship-Detection) and move them to `./dataset`.

Or you can access all the datasets we have collected via [Baidu Netdisk](https://pan.baidu.com/s/1FKV1m-RilwqQMcOjMyECbg?pwd=eq52).

## Run The Code

1. **Download SAM2 Pretrained Weights**

Before running the code, please download the SAM-related pretrained weights by executing:

```bash
bash sam_spl/sam2_ckpt/download_ckpts.sh
```

Or you can access them via [Baidu Netdisk](https://pan.baidu.com/s/10VmNT1u_YwEAmw3SqH-ygA?pwd=6swv).

2. **Train or Test the Model**

You can train or test the model using the provided scripts. For example:

```bash
# Training example
python training.py --batch_size 12 --image_size 256 --lr 1e-2 --dataset NUDT-SIRST --save_dir ./checkpoints/NUDT-SIRST --gpu 0

# Distributed Data Parallel Training example
CUDA_VISIBLE_DEVICES="0, 1, 2, 3" torchrun --nproc_per_node=4 --nnodes=1 training.py --batch_size 12 --image_size 256 --lr 1e-2 --dataset NUDT-SIRST --save_dir ./checkpoints/NUDT-SIRST --use_ddp

# Testing example
python testing.py --dataset NUDT-SIRST --image_size 256 --weights ./checkpoints/NUDT-SIRST.pt --device cuda:0
```


## Results
### Quantative Results

| Dataset | IoU (%) | F1 (%) | Pd (%) | Fa (10^-6) | Weight |
|--------|---------|--------|--------|------------|--------|
| NUDT-SIRST | **94.63** | **97.24** | **99.47** | **2.55** | [Weight](https://pan.baidu.com/s/1_Uq6EMcCvOZ9MlV4n7ZvQQ?pwd=tzuh) |
| IRSTD-1k | **74.09** | **85.11** | **92.59** | **9.28** | [Weight](https://pan.baidu.com/s/1k-EospCpbJIgUph9LiSHXQ?pwd=ckqs) |
| IRSTDID-SKY | **73.40** | **84.66** | **98.72** | **0.97** | [Weight](https://pan.baidu.com/s/1LqUA6ekPy1bV-HANtV3T3w?pwd=geic) |
| NUDT-sea | **60.93** | **75.72** | **82.33** | **14.92** | [Weight](https://pan.baidu.com/s/1cinhAaCKALtX6b-SLYWy9w?pwd=5j6g) |

## Qualitative Results
<p align="center">
    <img src=./assets/vis_sea.png width="900">
</p>

## Citation
If you find our work and this repository useful. Please consider giving a star :star: and citation.
```bibtex
@article{fu2025unified,
  title={A Unified SAM-Guided Self-Prompt Learning Framework for Infrared Small Target Detection},
  author={Fu, Yimin and Lyu, Jialin and Ma, Peiyuan and Liu, Zhunga and Ng, Michael K},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  volume={63},
  pages={1-14},
  doi={10.1109/TGRS.2025.3610919},
  publisher={IEEE}
}
```
