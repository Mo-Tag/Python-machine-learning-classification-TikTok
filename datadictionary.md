# Data Dictionary for TikTok Dataset 📊

This data dictionary describes the columns in the `tiktok_dataset.csv` file used in the TikTok Claim vs. Opinion Classifier project. The dataset contains metadata and engagement metrics for ~19,000 TikTok videos, aimed at classifying content as "claim" or "opinion" to aid moderation. The target is `claim_status` (binary: 'claim' or 'opinion').

## Dataset Overview
- **Rows**: 19,382 (original, before preprocessing; ~19,084 after dropping NaNs)
- **Columns**: 12 (original) + 1 derived (`text_length`) + token count features from NLP
- **Target Variable**: `claim_status` (balanced ~50/50 after checks)
- **Key Preprocessing Steps**:
  - Missing values: Dropped 298 rows with NaNs.
  - No outlier handling (tree models are robust).
  - Feature Engineering: Added `text_length` from `video_transcription_text`; used CountVectorizer for top 15 token counts (2-3 n-grams).
  - Encoding: Dummy-encoded `verified_status` and `author_ban_status`.

## Column Descriptions

| Column Name              | Data Type | Description                                                                 | Possible Values/Range                  | Notes                                                                 |
|--------------------------|:---------:|:---------------------------------------------------------------------------:|----------------------------------------|-----------------------------------------------------------------------|
| #                        | int64     | Row index or identifier.                                                    | 1 to 19,382                            | Not used as a feature; sequential ID.                                 |
| claim_status             | object    | Target: Indicates if the video is a claim (fact-based, more likely violative) or opinion. | 'claim', 'opinion', NaN                | Balanced (~50.3% claim, 49.7% opinion); encoded as binary (1 for claim). |
| video_id                 | int64     | Unique video identifier.                                                    | ~1.23e+09 to ~9.99e+09                 | Dropped from modeling (no predictive value).                          |
| video_duration_sec       | int64     | Video length in seconds.                                                    | 5 to 60                                | Numerical; minor correlation with engagement.                         |
| video_transcription_text | object    | Transcribed audio/text from the video.                                      | Free-form strings                      | Used for NLP: Derived `text_length`; tokenized via CountVectorizer (e.g., columns like `token_0`, `token_1` for top n-grams). Claims average longer text (~95 chars vs. ~83 for opinions). |
| verified_status          | object    | Whether the author is verified.                                             | 'verified', 'not verified'             | Categorical; dummy-encoded.                                           |
| author_ban_status        | object    | Author's ban status.                                                        | 'active', 'under review', 'banned'     | Categorical; dummy-encoded; 'banned' may correlate with claims.       |
| video_view_count         | float64   | Total views.                                                                | 20 to 999,817                          | Numerical; highly predictive (top feature importance); correlated with other engagement metrics. |
| video_like_count         | float64   | Total likes.                                                                | 0 to 657,830                           | Numerical; key engagement feature.                                    |
| video_share_count        | float64   | Total shares.                                                               | 0 to 256,130                           | Numerical; strong predictor.                                          |
| video_download_count     | float64   | Total downloads.                                                            | 0 to 14,994                            | Numerical; moderate importance.                                       |
| video_comment_count      | float64   | Total comments.                                                             | 0 to 9,599                             | Numerical; part of engagement signals.                                |
| text_length              | int64     | Derived: Character length of `video_transcription_text`.                    | Varies (~83–95 average)                | Engineered; helps distinguish claims (longer) from opinions.          |
| token_[0-14]             | int64     | Derived: Count of top 15 tokens (2-3 n-grams) from CountVectorizer on transcriptions. | 0+ (counts)                            | NLP features; e.g., 'token_0' might be "someone shared"; added 15 columns for modeling. |

## Additional Notes
- **Correlations**: High among engagement metrics (e.g., views-likes ~0.8+); models handled via tree-based methods.
- **Class Balance**: Near-perfect (50.3% claim); no resampling needed.
- **NLP Insights**: Claims use more structured phrases; histograms (`image1.png`) show text length separation.
- **Model Features**: Final X includes all numerics + dummies + text_length + token counts (dropped non-predictive like video_id).
- **Data Source**: `pd.read_csv("tiktok_dataset.csv")`; real TikTok video data for ML classification.

For details, see the project README or Jupyter Notebook.
