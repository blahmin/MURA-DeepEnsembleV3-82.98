# MURA-RL-DeepEnsembleV3-82.98

## Overview  
MURA-RL-DeepEnsembleV3-82.98 is the latest iteration in reinforcement learning-enhanced musculoskeletal X-ray classification, refining **RL-based ensemble weighting** while implementing **stronger generalization techniques**. Inspired by **DeepSeek-R1**, this model integrates SE blocks, increased dropout regularization, and **misclassified image oversampling**.  

### **Generalization Challenges & Overfitting Issues**  
This model reached its **peak validation accuracy of 82.98% early in training**, but over time, it exhibited **difficulty in generalizing**, with **train/val accuracy gaps of 10%+ appearing later in training**. Despite extensive regularization, **overfitting remains a key limitation**.  
- **The saved model represents the best-performing checkpoint before accuracy decline.**  
- **Future iterations will focus on improving generalization, reducing overfitting, and stabilizing long-term performance.**  

## Model Performance  
- **Validation Accuracy:** 82.98%  
- **Cohen’s Kappa Score:** 0.6565  
- **Average Gate Value:** 0.4051  

### **Per-Body-Part Validation Accuracy:**  
| Body Part  | Accuracy | Total Samples | Correct Predictions |
|------------|----------|---------------|---------------------|
| **XR_ELBOW**  | 86.45% | 465 | 402 |
| **XR_FINGER** | 81.34% | 461 | 375 |
| **XR_FOREARM** | 82.00% | 300 | 246 |
| **XR_HAND** | 80.87% | 460 | 372 |
| **XR_HUMERUS** | 85.07% | 288 | 245 |
| **XR_SHOULDER** | 77.44% | 563 | 436 |
| **XR_WRIST** | 87.41% | 659 | 576 |

## Key Changes from **V2 (82.4%)** → **V3 (82.98%)**
- **Squeeze-and-Excitation (SE) Blocks** refine feature extraction.  
- **Increased Dropout Regularization (0.35 → 0.55)** reduces overfitting.  
- **Reinforcement Learning Temperature Scaling** for dynamic ensemble weighting.  
- **DenseNet121-Based Expert** for misclassified cases.  
- **WeightedRandomSampler** prioritizes misclassified examples.  
- **Gating Network Dropout** stabilizes final predictions.  

## Model Architecture  
### **1. BaseModel (Residual CNN with Attention)**
- **EnhancedBlock** for residual learning and deep feature extraction.  
- **Squeeze-and-Excitation (SE) Blocks** dynamically adjust feature weighting.  
- **Adaptive dropout (0.35 → 0.55)** to improve generalization.  

### **2. RL-Inspired Weighting with Temperature Scaling**
- Learns **dynamic Softmax weights** for ensemble predictions.  
- **Temperature scaling** allows for more flexible weighting distributions.  

### **3. DenseNet121-Based Expert for Misclassified Cases**
- Specialized **DenseNet121 expert model** for handling difficult predictions.  
- Expert activation dynamically weighted via **gating network dropout**.  

### **4. Misclassified Image Oversampling**
- **WeightedRandomSampler** prioritizes **previously misclassified images**.  

## Training Pipeline  
- **Dataset:** MURA (Musculoskeletal Radiographs)  
- **Augmentations:** Color jitter, affine transforms, RandomErasing.  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** AdamW  
- **Scheduler:** OneCycleLR  
- **Batch Size:** 32  

## Installation  
Install dependencies:  
pip install torch torchvision albumentations pillow

python
Copy
Edit

## Download the RL Model
The RL-trained model ultimate_ensemble_v3.pth is too large for GitHub version control and is available in GitHub Releases here - https://github.com/blahmin/MURA-DeepEnsembleV3-82.98/releases/tag/model
