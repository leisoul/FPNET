
# Image Restoration Framework

This repository contains the implementation of an image restoration framework that supports multiple tasks, including denoising and deblurring.
The framework is based on a U-shaped encoder-decoder architecture with novel attention modules to balance efficiency and performance.


## 📂 Project Structure
```

project_root/
│── models/              # Network architecture definitions
│── option/              # Configuration files and hyperparameters
│── utils/               # Utility functions
│── AIO_dataset.py       # Dataset loader and preprocessing
│── train.py             # Training entry point
│── validation.py        # Validation and evaluation scripts
│── README.md            # Project description and usage

````

---

## 🚀 Features
- **Feature Pick (FP) Module**: Adaptively selects task-relevant features while discarding redundancy, reducing unnecessary computations and improving efficiency.
- **Detail Auxiliary Block (DAB)**: Enhances fine-grained details by dynamically adjusting weights, allowing critical structures to bypass the main restoration network.
- **Plug-and-Play Framework**: FP and DAB can be seamlessly integrated into various image restoration models with minimal computational overhead.
- **Computational Lightweighting**: Achieves up to **88% reduction in FLOPs** compared to Restormer while delivering superior restoration quality.
- **Versatile Restoration Capability**: Demonstrates state-of-the-art performance on multiple tasks, including **denoising** and **deblurring**.

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
python train.py 
```

---

## 📊 Validation

To evaluate the model:

```bash
python validation.py 
```

---

## 📂 Dataset

Prepare datasets according to the task (e.g., SIDD for denoising, GoPro for deblurring).
Modify `AIO_dataset.py` to set the dataset paths.

---

## 🔥 Results

The proposed method achieves state-of-the-art performance on multiple benchmarks:

* **Denoising:** SIDD
* **Deblurring:** GoPro

---

## 📄 Citation

If you find this work useful, please cite:

```
@inproceedings{wang2025efficient,
  title={Efficient Feature-Guided Approach for Image Restoration},
  author={Wang, Chan-Yu and Liu, Tsung-Jung and Liu, Kuan-Hsien},
  booktitle={2025 IEEE International Conference on Image Processing (ICIP)},
  pages={2886--2891},
  year={2025},
  organization={IEEE}
}
```

