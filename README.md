# Grammar Scoring Engine for Spoken English

Automated assessment of grammatical correctness from audio recordings using multi-modal deep learning.

## Project Overview

AI-powered grammar scoring engine that evaluates spoken English from audio samples, predicting grammar scores on a 1-5 scale. Achieved **0.569 RMSE** (2nd place) and **0.80 Pearson correlation**.

## Results

| Metric | Score |
|--------|-------|
| Test Score | 0.569 |
| Validation RMSE | 0.4995 |

## Architecture

### Feature Extraction (6169-dim)

- Whisper Embeddings (768-dim): Semantic/linguistic content
- Wav2Vec2 7-Layer (5376-dim): Multi-scale acoustic features
- Prosody Features (12-dim): Pitch, energy, pauses, rhythm
- Grammar Features (13-dim): Punctuation, lexical diversity

### Model Ensemble

Stacking ensemble with 5 base models:

1. XGBoost (n_estimators=800, max_depth=6)
2. LightGBM (n_estimators=800, max_depth=7)
3. CatBoost (iterations=900, depth=7)
4. Ridge Regression (alpha=2.0)
5. Random Forest (n_estimators=600, max_depth=14)

Meta-learner: Ridge regression (alpha=0.8)

## Installation

Install dependencies:

    pip install numpy pandas librosa scikit-learn
    pip install transformers torch xgboost lightgbm catboost scipy

## Usage

### Step 1: Feature Extraction

Run CELL 1 to extract all features (20-25 minutes).

### Step 2: Model Training

Run CELL 5 to train ensemble (3-5 minutes).

### Output Files

- submission_cell5_best.csv - Final predictions
- model_metrics_comprehensive.csv - Performance metrics
- TECHNICAL_REPORT.txt - Detailed methodology

## Project Structure

    grammar-scoring-engine/
    ├── cell1_feature_extraction.py
    ├── cell5_model_training.py
    ├── TECHNICAL_REPORT.txt
    ├── README.md
    └── requirements.txt

## Performance

### Individual Models (Validation)

| Model | Train RMSE | Val RMSE | Pearson |
|-------|------------|----------|---------|
| LightGBM | 0.4821 | 0.5608 
| XGBoost | 0.4856 | 0.5624 
| CatBoost | 0.5012 | 0.5920 
| Ridge | 0.6142 | 0.6487 
| Random Forest | 0.4321 | 0.6116
| Stacking | 0.4124 | 0.4995 

## Technical Details

### Hardware

- GPU: NVIDIA P100 (16GB VRAM)
- RAM: 13GB+
- Runtime: 30 minutes total

### Dependencies

- Python 3.8+
- PyTorch 2.1.0
- Transformers 4.35.0
- Librosa 0.10.1
- XGBoost 2.0.3
- LightGBM 4.1.0
- CatBoost 1.2

## Key Insights

### What Worked

- Multi-modal features (audio + text) essential
- 7-layer Wav2Vec2 captures richer acoustic info
- Stacking ensemble superior to weighted averaging
- Conservative hyperparameters for small datasets

### What Didn't Work

- Advanced prosody features caused overfitting
- Deep neural networks struggled (dataset too small)

## Future Improvements

- K-fold cross-validation
- Multi-task learning
- Data augmentation

## References

- OpenAI Whisper: https://github.com/openai/whisper
- Facebook Wav2Vec2: https://arxiv.org/abs/2006.11477
- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- CatBoost: https://catboost.ai/

## License

MIT License

## Author

[Your Name]
