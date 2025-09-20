# PDF Email Classifier — Streamlit App

This project is a **machine learning email classifier** built with Streamlit.  
It classifies emails from an uploaded PDF into 15 categories (Promotions, Shopping, Social, Finance, Travel, HealthCare, Technology, Education, JobAlerts, News, Support/Services, Events, Legal, Government, Spam).

---

## Features
- Generate **10,000 synthetic training samples** automatically.
- Train a **Logistic Regression model** with TF-IDF features.
- Upload a **PDF of emails** (unlabeled) and classify them.
- View:
  - Category-wise counts
  - Detailed predictions per email
- Sophisticated UI with Streamlit.

---

## Setup

1. Clone this repo:
   git clone https://github.com/<your-username>/pdf-email-classifier.git
   cd pdf-email-classifier
   
2. Install dependencies:
   pip install -r requirements.txt
   
3. Run the Streamlit app:
   streamlit run streamlit_pdf_email_classifier.py
