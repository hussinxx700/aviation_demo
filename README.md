# ✈️ Aviation Incident Risk Assessment System

This repository contains a demo project showcasing an AI-powered system for assessing aviation incident risk based on human, technical, and environmental factors. The system is built using synthetic data and serves as a proof-of-concept for academic or prototype use.

## 📌 Project Overview

The model analyzes risk based on three key areas:
- **Human Factor**: Pilot and Co-pilot experience, fatigue, training level
- **Technical Factor**: Aircraft model, system status, maintenance records
- **Environmental Factor**: Weather, bird activity, visibility, season

It predicts:
- Whether an incident is likely (binary classification)
- The probability (risk percentage)
- The **top 2 local features** influencing each individual prediction (via SHAP)

---

## 🧠 Tech Stack

- Python 3.10+
- [XGBoost](https://xgboost.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)
- [Streamlit](https://streamlit.io/)
- Pandas, scikit-learn, imbalanced-learn

---

## 📊 Data

The dataset includes 500 synthetic samples with realistic patterns.  
Each row represents one flight. The dataset is divided into:

- `Human` features  
- `Technical` features  
- `Environmental` features  
- Target: `Incident` (0 = No, 1 = Yes)

---

## 🚀 How to Run the App

### 1. Clone the repo
```bash
git clone https://github.com/your-username/aviation-risk-demo.git
cd aviation-risk-demo
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 📂 File Structure

```
.
├── app.py                 # Streamlit user interface
├── processor.py           # Model loading and SHAP explainability
├── sample_input_0.csv     # Preloaded sample 1
├── sample_input_1.csv     # Preloaded sample 2
├── sample_input_2.csv     # Preloaded sample 3
├── incident_risk_pipeline.pkl  # Trained XGBoost model
├── requirements.txt
└── README.md
```

---

## 🧪 Example Use

Select a sample from the sidebar and the system will:
- Predict risk probability (e.g., 76.32%)
- Output incident class (Yes/No)
- Show top 2 contributing factors

---

## 📘 License

This project is for academic and demo purposes only.  
Developed by **Husseini** for Master's thesis support.