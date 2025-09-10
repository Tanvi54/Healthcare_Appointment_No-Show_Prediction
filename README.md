# 🏥 Healthcare Appointment No-Show Prediction  


## 📌 Introduction  
Missed appointments create major challenges in healthcare systems.  
This project predicts **No-Shows** using patient information such as scheduling date, appointment date, demographics, and neighborhood.  

The aim is to:  
✅ Improve hospital efficiency  
✅ Assist doctors in better scheduling  
✅ Reduce patient waiting times  

---

## 🔄 Project Workflow  

1. **Data Preprocessing** – Cleaning missing values, handling categorical features, creating new features (waiting days, weekdays). 
2. **Feature Engineering** – Encoded categorical columns (Gender, Neighborhood).  
3. **Handling Imbalance** – Applied **SMOTE** to balance No-Show vs Show classes.  
4. **Model Training** – Used a Decision Tree Classifier  
5. **Prediction Interface** – Built using **Streamlit** (single & multiple patient predictions).  
6. **Visualization** – Power BI dashboard for healthcare insights.  

---

## 🛠 Tech Stack  

- **Python** (Pandas, Scikit-learn, Imbalanced-learn, Joblib)  
- **Streamlit** – Interactive prediction app  
- **Power BI** – Dashboard for insights  
- **Jupyter Notebook** – Exploratory Data Analysis (EDA)  
- **Git & GitHub** – Version control  


---

## 📂 Project Structure  

```bash
Healthcare_Appointment_NoShow_Prediction/
│
├── data/
│   ├── model/               # Saved ML models & encoders
│   ├── processed/           # Cleaned datasets & prediction outputs
│   └── raw/                 # Original/raw datasets
│
├── notebooks/               # Jupyter notebooks (exploratory analysis, experiments)
├── powerbi/                 # Power BI dashboard files
├── reports/                 # Final reports, PDFs, documentation
│
├── src/
│   ├── data_preprocessing.py  # Data cleaning & preprocessing
│   ├── evaluate_model.py      # Model evaluation metrics & reports
│   ├── predict.py             # Streamlit UI for predictions
│   └── utils.py               # Helper functions
│
├── venv/                    # Virtual environment (not pushed to GitHub, add to .gitignore)
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
```
---

## 🚀 How to Run
1️⃣ Clone the Repository: 

git clone https://github.com/yourusername/Healthcare_Appointment_NoShow_Prediction.git

cd Healthcare_Appointment_NoShow_Prediction

2️⃣ Install Dependencies:

pip install -r requirements.txt

3️⃣ Train the Model:

python src/train_model.py


✔️ This will:
- Clean & encode data
- Handle imbalance using SMOTE
- Train a Decision Tree model
- Save model & encoders in data/model/
- Generate predictions in data/processed/test_with_predictions.csv

4️⃣ Run Streamlit App:

streamlit run src/predict.py

🧍 Enter single patient details or upload multiple records
-  Get real-time prediction → Show / NoShow

5️⃣ Power BI Dashboard
- Import data/processed/test_with_predictions.csv into Power BI
- Use prebuilt visuals:
- Show vs NoShow ratio
- Gender breakdown
- Waiting Days impact
- Customize with hospital theme color palette

---

## ▶️ Usage

- Single Prediction: Enter patient details in Streamlit UI.
- Batch Prediction: Input multiple patients in a table form.
- Dashboard: Import test_with_predictions.csv into Power BI and explore insights.

---


## 📊 Results & Insights
- Model Used → Decision Tree Classifier 🌳
- Achieved Accuracy → 0.696 (~70%)
- Balanced using SMOTE for fair prediction of Show vs NoShow
- Key metrics analyzed in Power BI:
- Show vs No-Show ratio
- Gender & Neighborhood analysis
- Average waiting days distribution
- Accuracy of ML predictions

---

## 🏁 Conclusion

This project demonstrates how data science and visualization can assist hospitals in predicting appointment no-shows.
By integrating ML models, Streamlit UI, and Power BI dashboards, we can help healthcare providers improve resource planning, reduce patient wait times, and increase efficiency.
