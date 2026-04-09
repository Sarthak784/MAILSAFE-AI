# MailSafe AI - GitHub Deployment

## What this is
MailSafe AI detects anomalous email senders using behavioral pattern analysis and Isolation Forest.  
Upload a raw Enron-style CSV to run the full pipeline in browser.

## Deploy on Streamlit Cloud
1. Fork/push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set main file to `app.py`
4. Deploy

## Demo
Download sample data from the Enron dataset (Kaggle) and upload via the sidebar.
App runs full pipeline: parse → feature engineering → anomaly detection → charts.
