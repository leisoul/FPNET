
```markdown
# Image Restoration Framework

This repository contains the implementation of an image restoration framework that supports multiple tasks, including denoising, deblurring, low-light enhancement, and raindrop removal.  
The framework is based on a U-shaped encoder-decoder architecture with novel attention modules to balance efficiency and performance.

---

## 📂 Project Structure
```

project\_root/
│── models/              # Network architecture definitions
│── option/              # Configuration files and hyperparameters
│── utils/               # Utility functions
│── AIO\_dataset.py       # Dataset loader and preprocessing
│── train.py             # Training entry point
│── validation.py        # Validation and evaluation scripts
│── README.md            # Project description and usage

````

---

## 🚀 Features
- **Enhanced U-shaped architecture** with improved head and tail design.
- **Multi-Branch Directional Convolution Mechanism (MBDM)** for structure recovery.
- **Shallow Feature Channel Attention Module (SF-CAM)** for detail refinement.
- **Lightweight attention mechanism** that captures both global and local features.
- Supports **multiple restoration tasks** with competitive efficiency.

---

## ⚙️ Requirements
- Python 3.8+
- PyTorch >= 1.9
- torchvision
- numpy, scipy, opencv-python
- tqdm, matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
````

---

## 🏋️ Training

To start training, run:

```bash
python train.py --config option/train_config.yaml
```

---

## 📊 Validation

To evaluate the model:

```bash
python validation.py --config option/val_config.yaml --weights path/to/checkpoint.pth
```

---

## 📂 Dataset

Prepare datasets according to the task (e.g., SIDD for denoising, GoPro for deblurring).
Modify `AIO_dataset.py` to set the dataset paths.

---

## 🔥 Results

The proposed method achieves state-of-the-art performance on multiple benchmarks:

* **Denoising:** SIDD, DND
* **Deblurring:** GoPro
* **Low-light enhancement:** LOL dataset
* **Raindrop removal:** RainDrop dataset

---

## 📄 Citation

If you find this work useful, please cite:

```
@article{your_paper,
  title={Efficient Feature-Guided Approach for Image Restoration},
  author={Your Name},
  journal={To appear in ...},
  year={2025}
}
```

---

## 🙌 Acknowledgments

This repository builds upon various open-source implementations of CNN and Transformer-based image restoration models.

```

---

要不要我幫你再生一份 **中文版 README**，比較方便你交給團隊內部使用？
```

