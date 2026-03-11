# Machine Learning in Python: Video Classification - Claim vs Opinion 🚀

## Introduction 🌟

As a data professional at TikTok, I built a high-performance machine learning model to classify videos as either **"claim"** or **"opinion"**. This helps reduce the massive backlog of user-reported videos by automatically prioritizing content that violates platform terms of service (claims are far more likely to break rules).

Using a real-world dataset of ~19,000 TikTok videos, the project demonstrates end-to-end machine learning: ethical considerations, feature engineering (including NLP tokenization), model training, hyperparameter tuning, and rigorous evaluation. The champion model achieved **near-perfect recall** (the key metric), ensuring almost no harmful claims are missed.

> **Business Impact**: Videos classified as opinions can skip human review, while claims are escalated — directly supporting TikTok’s content moderation goals.

## Skills Showcased 🛠️

- **Ethical Analysis & PACE Framework**: Evaluated real-world consequences of false negatives vs. false positives and selected **recall** as the primary metric.
- **Feature Engineering & NLP**: Created `text_length` feature, dummy-encoded categoricals, and used **CountVectorizer** (2-3 n-grams) to extract 15 most frequent tokens from video transcriptions.
- **Advanced Modeling**: Built and tuned **Random Forest** and **XGBoost** classifiers with GridSearchCV (cross-validation on 60/20/20 train/val/test split).
- **Model Evaluation**: Confusion matrices, classification reports, feature importance analysis, and champion model selection.
- **Libraries**: pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn.

## Visualizations Utilized 📈

- **Histogram**: Distribution of transcription text length for claims vs. opinions (claims average ~13 characters longer).  
<img width="919" height="683" alt="image" src="https://github.com/user-attachments/assets/56add876-38ac-49de-92c0-f74287cbec8a" />

- **Confusion Matrices**: Near-perfect classification on validation and test sets (only ~5–10 total misclassifications).  
<img width="773" height="683" alt="image" src="https://github.com/user-attachments/assets/532392c3-7c8a-4348-8f07-b320bd444c5b" />

<img width="773" height="683" alt="image" src="https://github.com/user-attachments/assets/ef6f4201-4b8f-4839-a243-47f75a772c96" />

- **Feature Importance Plot**: Engagement metrics (views, likes, shares) dominate predictions.  
<img width="940" height="702" alt="image" src="https://github.com/user-attachments/assets/13f35273-4dbe-4c2e-a366-f371193b935b" />

- **Additional EDA**: Boxplots, correlation insights, and class balance confirmation (perfectly balanced ~50/50).


## Project Overview 🔎

**PACE-Structured Workflow** (Plan → Analyze → Construct → Execute):

1. **Plan**: Defined business objective, chose **recall** as metric (ethical priority: never miss a claim), planned 60/20/20 split.
2. **Analyze**: Handled missing values, confirmed no duplicates/outliers needed (tree models are robust), verified balanced target.
3. **Construct**:
   - Engineered `text_length` + n-gram features from `video_transcription_text`.
   - Trained Random Forest & XGBoost with hyperparameter tuning.
   - Random Forest won: **0.995 recall** on CV, near-perfect precision.
4. **Execute**: Validated on holdout sets, selected champion, analyzed feature importances.

The model learned that **high engagement + longer transcriptions = claim** — exactly as prior EDA suggested.

## Conclusion

This project proves the power of thoughtful machine learning for real-world moderation. The champion Random Forest model delivers **~100% recall** while maintaining exceptional precision, making it production-ready for TikTok’s claim/opinion triage.

**Key Takeaways**:
- Ethical metric selection (recall-first) was critical.
- Engagement features + simple NLP tokens were enough for near-perfect performance.
- Ready for deployment or further enhancement (e.g., adding report counts).

A fantastic demonstration of my skills in **classification modeling, NLP feature engineering, hyperparameter tuning, and ethical data science**.

**Technologies**: Python • scikit-learn • XGBoost • Pandas • NLP (CountVectorizer) • Matplotlib/Seaborn

---
