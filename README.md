# NDLPNet: A Location-Aware Network for Nighttime Image Deraining with a Semi-Real Synthetic Paired Benchmark Dataset

[![arXiv](https://img.shields.io/badge/arXiv-Submitted-b31b1b.svg)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A Location-Aware Network for Nighttime Image Deraining with a Semi-Real Synthetic Paired Benchmark Dataset**

This is the official PyTorch implementation of **NDLPNet**, a Transformer-based framework for location-aware nighttime image deraining. NDLPNet explicitly models the spatial heterogeneity of rain-related degradation under complex nighttime illumination through a combination of Rain Prior Extractor (RPE) and Position Perception Module (PPM), which jointly utilize Spatial Position Coding (SPC) and Efficient Channel Attention (ECA).

<p align="center">
  <img src="Figures/Framework.png" width="90%">
</p>

## 📖 Abstract

Visual degradation caused by rain streak artifacts in low-light conditions significantly reduces nighttime image quality and affects vision-based applications. Existing image deraining techniques are primarily designed for daytime conditions and perform poorly under nighttime illumination due to the spatial heterogeneity of rain distribution and the impact of light-dependent stripe visibility. To address the unique challenges of nighttime deraining, we propose **NDLPNet**, a Transformer-based framework that explicitly models the location-dependent visibility and spatial heterogeneity of rain-related degradation under complex nighttime illumination. The framework employs a Rain Prior Extractor to obtain a coarse estimate of rain-streak distribution, then introduces a Position Perception Module where Spatial Position Coding and Efficient Channel Attention collaboratively refine this prior into more discriminative position-aware representations. Additionally, we construct the **NSR dataset** (900 training pairs and 100 testing pairs) built on real nighttime backgrounds with synthetically generated rain streaks, which provides a more realistic benchmark for nighttime deraining research than fully virtual nighttime datasets. Extensive qualitative and quantitative experimental evaluations demonstrate that NDLPNet achieves strong overall performance and obtains the best results on most nighttime deraining metrics among compared methods.

## ✨ Highlights

- 🎯 **Location-Aware Design**: Explicitly models location-dependent visibility and spatial heterogeneity of rain-related degradation under nighttime illumination
- 🌤️ **Rain Prior Extractor (RPE)**: Generates coarse rain-streak location information from rainy nighttime images
- 📍 **Position Perception Module (PPM)**: Combines Spatial Position Coding (SPC) and Efficient Channel Attention (ECA) for position-aware feature modulation
- 📡 **Spatial Position Coding (SPC)**: Three-axis encoding scheme incorporating horizontal position, vertical position, and RPE-derived rain-location cue
- ⚡ **Efficient Channel Attention (ECA)**: Performs channel-wise recalibration on SPC-enhanced representations
- 🌊 **Restormer-Based Backbone**: Hierarchical encoder-decoder architecture with 4 stages, dim=128, maximum depth=1024
- 📊 **NSR Dataset**: Semi-real synthetic paired nighttime deraining benchmark (1000 image pairs) built on real nighttime backgrounds
- 🎯 **Strong Performance**: Achieves best results on most nighttime deraining metrics (PSNR, SSIM, VIF, LPIPS)

## 🔥 Visual Results

### Nighttime Deraining Results

<p align="center">
  <img src="Figures/nighttime_results.png" width="90%">
</p>

### Comparison with State-of-the-art Methods

<p align="center">
  <img src="Figures/comparison.png" width="90%">
</p>


## 🌐 Overview

NDLPNet addresses the unique challenges of nighttime rain removal by introducing a prior-guided location-aware restoration framework. Key aspects include:

- **Problem**: Nighttime rain-related degradation is highly non-uniform and location-dependent due to complex illumination
- **Key Innovation**: Combines RPE with PPM (SPC + ECA) to explicitly model spatial heterogeneity
- **Quantitative Validation**: Rain degradation 2.1× stronger near light sources vs. far regions (coefficient of variation: 0.8545)
- **End-to-End Learning**: Joint optimization of rain localization and restoration without circular self-reinforcement


## 📊 Datasets & Benchmark

### 🎯 NSR Dataset: Our Core Contribution

**NSR (Semi-Real Nighttime)** is a semi-real synthetic paired nighttime deraining benchmark dataset that we construct specifically for this work. Unlike fully virtual datasets, NSR combines real nighttime backgrounds with synthetically generated rain streaks, providing a more realistic and challenging evaluation setting.

#### Overview

| Property | Value |
|:---|:---|
| **Dataset Name** | NSR (Semi-Real Nighttime) |
| **Total Pairs** | 1000 image pairs |
| **Training Set** | 900 pairs |
| **Testing Set** | 100 pairs |
| **Image Source** | Real UAV-captured nighttime backgrounds |
| **Rain Source** | Synthetically generated from real rain video masks |
| **Availability** | Available in [releases](../../releases) |

#### Dataset Characteristics & Uniqueness

<p align="center">
  <img src="Figures/nsr_dataset.png" width="90%">
</p>

**Data Acquisition & Construction**:
- **Real Backgrounds**: Captured by DJI M30T unmanned aerial vehicles over diverse nighttime scenes
- **Scene Diversity**: Campus, parking lot, park, urban road, stadium, subway station (~150 unique scenes)
- **Rain Synthesis Pipeline**: 
  - Mask-based compositing using rain streak masks extracted from real rain videos
  - Carefully calibrated synthesis parameters for photorealism
  - Preserves authentic rain streak morphology and motion blur

**Synthesis Parameters**:
- **Opacity Control**: Variable transparency for realistic rain visibility
- **Motion Blur**: Streak-specific motion vectors simulating camera/rain motion
- **Brightness & Contrast Adjustment**: Scene-adaptive intensity for nighttime authenticity
- **Particle Effects**: Subtle secondary particles for additional realism
- **Noise Intensity**: Sensor noise matching real nighttime imaging characteristics

**Key Advantages Over Alternatives**:
| Aspect | NSR | Fully Virtual (GTAV) | Fully Real | Daytime (Rain200L/H) |
|:---|:---:|:---:|:---:|:---:|
| Real Backgrounds | ✓ | ✗ | ✓ | ✓ |
| Nighttime Scenes | ✓ | ✓ | ✗ | ✗ |
| Controlled Rain | ✓ | ✓ | ✗ | ✓ |
| Pair Annotation | ✓ | ✓ | ✗ | ✓ |
| Sensor Noise | ✓ | ✗ | ✓ | ✗ |


**Directory Structure**:
```
NSR/
├── train/
│   ├── rainy/   # Rainy nighttime images (900 pairs)
│   └── clean/   # Clean reference ground truth (900 pairs)
└── test/
    ├── rainy/   # Rainy nighttime images (100 pairs)
    └── clean/   # Clean reference ground truth (100 pairs)
```

### Other Training Datasets

| Dataset | Type | Pairs | Scenes | Purpose |
|:---|:---|:---:|:---|:---|
| **GTAV-NightRain** | Fully Virtual | 2000+100 | Virtual nighttime urban | Nighttime training | 

## ⚙️ Installation

### Requirements
- Python >= 3.8
- PyTorch >= 1.9.0
- CUDA (recommended for GPU acceleration)

### Dependencies

```bash
pip install torch torchvision
pip install timm einops
pip install opencv-python pillow scipy numpy
pip install kornia  # For evaluation metrics
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/NDLPNet.git
cd NDLPNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets:
   - GTAV-NightRain: [Dataset Link]
   - Rain200L/Rain200H: [Dataset Link]
   - NSR: [Release with dataset]

4. Organize dataset structure as described above

## 🚀 Quick Start

### 📥 Pre-trained Models

We provide two pre-trained checkpoints:

| Model | Dataset | Download |
|:---|:---|:---|
| **NDLPNet-NSR** | NSR (Semi-Real) | [checkpoints/NSR.pth](checkpoints/NSR.pth) |
| **NDLPNet-GTAV** | GTAV-NightRain | [checkpoints/GTAV-Nightrain.pth](checkpoints/GTAV-Nightrain.pth) |

### 🎯 Training

```bash
cd NDLPNet
./train.sh Deraining/Options/NDLPNet_Deraining.yml
```

### 🔍 Testing & Inference

```bash
python Deraining/test.py
```

## 📉 Loss Functions

The training employs a comprehensive multi-component loss function designed for effective nighttime rain removal:

- **L1 Loss**: Primary reconstruction loss for pixel-level accuracy
- **Perceptual Loss**: Feature-level perceptual quality using pre-trained VGG
- **SSIM Loss**: Structural similarity preservation
- **Optional**: Adversarial loss for enhanced perceptual quality

Total Loss: $L = L_1 + λ_{perceptual} L_{perceptual} + λ_{ssim} L_{ssim}$

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{ndlpnet2026,
  title={A Location-Aware Network for Nighttime Image Deraining with a Semi-Real Synthetic Paired Benchmark Dataset},
  author={Author Name and Co-authors},
  journal={Engineering Applications of Artificial Intelligence (EAAI)},
  year={2026}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature requests.

## 🙏 Acknowledgments

This project builds upon and acknowledges:

- **[Restormer](https://github.com/swz30/Restormer)** - Efficient Transformer architecture for image restoration
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[CBAM](https://github.com/Jongchan/attention-is-all-you-need-pytorch)** - Attention mechanism design
- **[Kornia](https://github.com/edgeai-technology/kornia)** - Image processing and evaluation metrics
- **[timm](https://github.com/rwightman/pytorch-image-models)** - PyTorch image models library

We thank all contributors and the open-source community for their valuable tools and insights.

---

## 📧 Contact & Support

For questions, suggestions, or collaboration inquiries:

- **Email**: Feecuin@outlook.com
- **GitHub Issues**: [Report issues or ask questions](../../issues)
- **Discussions**: [Join our discussion forum](../../discussions)

<!-- ## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this software for both academic and commercial purposes. -->

---

**Last Updated**: June 22, 2026  
<!-- **Project Status**: ✅ Active Development & Maintenance  
**Latest Release**: [v1.0.0](../../releases/tag/v1.0.0) -->
