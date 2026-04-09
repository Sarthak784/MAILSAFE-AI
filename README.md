# MailSafe AI: Privacy-Preserving Behavioral Anomaly Detection

## Overview
MailSafe AI is an advanced, fully offline behavioral anomaly detection framework designed to identify automated bulk-senders, mailing list daemons, and statistically extreme outliers within enterprise email networks. 

Unlike traditional content-based classifiers (e.g., Naive Bayes or NLP models) that suffer from adversarial text obfuscation and raise severe data-privacy concerns, MailSafe AI operates entirely on **SMTP metadata**. Message bodies are systematically ignored. By profiling temporal rhythms, multi-recipient graph topology, and sending burst frequency, MailSafe AI identifies statistically anomalous senders without exposing private communications.

This repository contains the deployable, fully standalone Streamlit dashboard.

## Technical Pipeline
The application executes the following pipeline when a raw Email CSV is uploaded:
1. **Metadata Parsing:** Extracts `From`, `To`, `Cc`, `Date`, and subject length using robust Python email libraries and Pandas.
2. **Feature Engineering (Sender Profiling):** Aggregates logs to compute 10 behavioral indicators per sender, including:
   - *Temporal Signals:* Night-email ratio, weekend ratio, and hourly burst score.
   - *Topological Signals:* Unique recipient count, Shannon entropy of the recipient distribution, and recipient response concentration.
3. **Scoring:** The pipeline standardizes the generated profiles and scores them against an **Isolation Forest** model (pre-trained on 50,000 Enron dataset emails with a 10% contamination prior).
4. **Visual Analytics:** The dashboard automatically plots 2D PCA clustering configurations, score histograms, multidimensional radar charts, and tabular risk summaries for manual investigation.

## Streamlit Cloud Deployment Instructions
1. Fork or push this repository to GitHub. Ensure both `isolation_forest.pkl` and `scaler.pkl` are located in the repository's root directory.
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and authenticate your GitHub account.
3. Select this repository and specify `app.py` as the main application file.
4. Deploy the application. Streamlit Cloud will resolve all dependencies listed in `requirements.txt`.

## Usage
Once deployed, users can upload any `csv` containing `file` and `message` columns (matching the structure of the Enron Corpus). The application evaluates the new data against the pre-trained Enron baseline, instantly flagging highly suspicious sending behaviors.
