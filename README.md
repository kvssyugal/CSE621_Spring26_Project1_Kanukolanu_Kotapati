# CSE621 – Spring 2026 – Project 1
**Sentiment Classification of Movie Reviews (IMDb 50k)**  
**Team:** Yugal Kanukolanu, Varshika Kotapati  
**Report + Slides:** in `deliverables/`

## Links
- **Video:** https://cardmaillouisville-my.sharepoint.com/:v:/g/personal/v0kanu01_louisville_edu/IQBDkEe5-1JQTaP4AC8hyOeNATUmFzNWsy1ZlTbEHhkMwfk?e=pcEDqe
- **Code:** https://github.com/kvssyugal/CSE621_Spring26_Project1_Kanukolanu_Kotapati

## What we did (quick)
We built a sentiment classification pipeline and compared:
- Preprocessing: baseline vs stopword removal + unigram/bigram TF-IDF
- Feature selection: none vs chi-square (top-k)
- Models: Naive Bayes, SGDClassifier, Linear SVM (LinearSVC)
- Ensemble: stacking to combine complementary models

## Best Results (test set)
- **Best single model:** LinearSVC — Accuracy 0.9068, Weighted F1 0.9068  
- **Ensemble (stacking):** Accuracy 0.9118, Weighted F1 0.9118  

## How to run (local)
1) Create and activate a virtual environment:
- macOS/Linux:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`

2) Install requirements:
- `pip install -r requirements.txt`

3) Run experiments:
- `python run_project1_experiments.py`

## How to run on LARCC
- `sbatch run_proj1.sbatch`

Outputs are written to: `project1_outputs/`
