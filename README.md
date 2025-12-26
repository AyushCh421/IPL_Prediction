# 🏏 IPL Match Prediction System

A two-stage Machine Learning based IPL prediction system that estimates the **first innings score** and predicts the **winning probability during the second innings** using real-time match conditions.

---

## 🔗 Live Application

👉 **Streamlit App:**  
https://your-app-name.streamlit.app/

*(Replace this link with your actual deployed Streamlit URL)*

---

## 📌 Project Overview

This project mimics real-world cricket analytics systems by dividing prediction into two logical stages:

### 1️⃣ First Innings Score Prediction  
Predicts the approximate final score of the team batting first using match context.

### 2️⃣ Second Innings Winner Prediction  
Predicts the winning probability of the chasing team based on live match conditions.

This design avoids data leakage and reflects real-time prediction scenarios.

---

## 🧠 Machine Learning Models

### 🔹 First Innings Score Predictor
- **Type:** Regression  
- **Algorithms:** Random Forest / XGBoost Regressor  
- **Metrics:** MAE, RMSE  
- **Output:** Predicted final score with confidence range  

**Input Features:**
- Batting Team  
- Bowling Team  
- City  
- Toss Winner  
- Toss Decision  
- Current Score  
- Overs Left  
- Balls Left  
- Wickets Fallen  
- Current Run Rate (CRR)

---

### 🔹 Second Innings Winner Predictor
- **Type:** Binary Classification  
- **Algorithms:** Logistic Regression / XGBoost Classifier  
- **Metric:** ROC-AUC (~0.89)  
- **Output:** Win probability for both teams  

**Input Features:**
- Batting Team (Chasing)  
- Bowling Team (Defending)  
- City  
- Runs Left  
- Balls Left  
- Wickets Remaining  
- Target Score  
- Current Run Rate (CRR)  
- Required Run Rate (RRR)

---

## 🚀 Features

- Two-stage ML prediction system  
- Real-time win probability estimation  
- Clean ML pipelines with no data leakage  
- Interactive Streamlit UI  
- Probability-based predictions  

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Joblib  

---

## 📂 Project Structure


ipl-prediction/
│
├── app.py                     # Streamlit application
├── requirements.txt           # All dependencies (version pinned)
├── README.md                  # Project documentation
├── .gitignore                 # Ignore data & cache files
│
├── Innings1.pkl               # First innings score prediction model
├── IPL_model.pkl              # Second innings winner prediction model
│
├── notebooks/
│   ├── Inning1.ipynb          # Model training notebook (1st innings)
│   └── Inning2.ipynb          # Model training notebook (2nd innings)
│
└── data/                     
    └── IPL.csv                






---

## ▶️ How to Run Locally

```bash
git clone https://github.com/your-username/ipl-predictor.git
cd ipl-predictor
pip install -r requirements.txt
streamlit run app.py


📈 Results

First Innings Model

RMSE ≈ 16–18 runs

Second Innings Model

ROC-AUC ≈ 0.889

Stable and reliable probability predictions

🔮 Future Improvements

Venue-level pitch statistics

Ball-by-ball probability visualization

Player-level features

Probability calibration

Mobile-friendly UI

👤 Author

Ayush Chauhan
Second Year Undergraduate | Machine Learning Enthusiast