# ReViT — Radiomics-Enhanced Vision Transformer

Binary classification of muscle ultrasound images 
(Normal vs Abnormal) using cross-attention fusion 
of radiomics features and Vision Transformers.

## Architecture
- SAM-based ROI segmentation
- ViT backbone (google/vit-base-patch16-224-in21k)
- RadiomicsEmbedder: 93-dim → 768-dim projection
- CrossAttentionFusion: bidirectional attention 
  between radiomics and ViT patch tokens
- SHAP explainability for radiomics contributions

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 86.9% |
| ROC-AUC | 0.884 |
| Baseline (ViT only) | 70% |

## Dataset
Clinical muscle ultrasound dataset — 305 images  
across 4 Heckmatt grades (NIT Calicut, not public)

## Tech Stack
PyTorch, HuggingFace Transformers, SHAP,  
scikit-learn, pyradiomics, NumPy, Pandas

## Run
Open in Google Colab with T4 GPU.  
Update BASE path in Cell 2 to your dataset location.
