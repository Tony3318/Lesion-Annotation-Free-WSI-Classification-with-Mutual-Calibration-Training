# Lesion-Annotation-Free-WSI-Classification-with-Mutual-Calibration-Training
A deep learning framework for skin lesion classification on whole slide images (WSIs) without lesion-level annotations. Integrating weakly supervised learning and mutual calibration with noise-aware labeling, it achieves 82.29% accuracy across 14 lesion types, improving performance by 3.6% over the baseline.

# Lesion-Annotation-Free WSI Classification with Mutual Calibration Training

## 🧠 Overview
**A deep learning framework for skin lesion classification on whole slide images (WSIs) without lesion-level annotations.**  
By integrating **weakly supervised learning**, **mutual calibration training**, and **noise-aware labeling**, the model achieves **82.29% accuracy** across 14 lesion types — a **3.6% improvement** over the baseline.

---

## 📊 Dataset
- **Total slides:** 2,580 histopathological WSIs  
- **Lesion categories (14):**  
  Dermatofibroma, Epidermal cyst, Fibroepithelial polyp, Granuloma, Hemangioma, Lipoma, Neurofibroma, Nevus, Pilomatricoma, Seborrheic keratosis, Verruca vulgaris, Basal cell carcinoma (BCC), Melanoma, Squamous cell carcinoma (SCC)  
- **Split ratio:** Train 60% / Validation 20% / Test 20%  
- Data collected under clinical collaboration; not publicly available due to privacy regulations.

---

## ⚙️ Methodology

### 1. Weakly Supervised Learning  
Training uses only **slide-level labels**, eliminating the need for manual lesion annotations.

### 2. Mutual Calibration Training  
Training data is divided into two subsets (odd/even).  
Two models (Model A, Model B) are independently trained and then cross-predict each other’s data to reduce overfitting.

### 3. Noise-Aware Labeling  
Misclassified samples are reassigned to “noise” categories, expanding the dataset from **14 → 28 classes**, improving discrimination on boundary and ambiguous cases.

### 4. Confidence-Weighted Ensemble Voting  
During inference, both models vote on predictions.  
Only predictions with **confidence > 0.7** are retained; noise-class results are filtered out.

---

## 🧩 Architecture
Input WSIs → Patch Extraction → Two Subset Training
↓
Model A ↔ Model B (Cross-Prediction)
↓
Noise-Aware Relabeling (14 + 14 Classes)
↓
Dual 28-Class Models
↓
Confidence-Weighted Voting → Final Prediction


---

## 🚀 Results

| Metric | Baseline | Proposed |
|:-------|:----------|:----------|
| Accuracy | 78.67% | **82.29%** |
| F1-score | 0.76 | **0.81** |

### 🧱 Ablation Study
| Variant | Description | Accuracy |
|----------|--------------|-----------|
| Dual-Model (Consensus Only) | Both models must agree | 82.10% |
| Dual-Model + Weighted Voting | Without noise classes | 81.14% |
| **Proposed (Noise-aware + Voting)** | Final version | **82.29%** |

---

## 🔍 Model Interpretability
The model identifies histopathological features consistent with expert pathology:
- **Basal Cell Carcinoma:** Palisading cell nests and irregular dark-purple clusters  
- **Melanoma:** Atypical melanocytes with melanin deposition  
- **Squamous Cell Carcinoma:** Keratin pearls and clustered atypical cells  
- **Noise Class:** Normal connective or adipose tissue  

---

