# Rain_predict_-23
# 🌧️ Rainfall Prediction using Random Forest

This project predicts whether it will **rain** or **not** using weather parameters such as temperature, pressure, humidity, dewpoint, sunshine, wind direction, and windspeed.  
It is built with **Python, Scikit-learn, Seaborn, and Pandas** and deployed with a trained **Random Forest Classifier**.

---

## 📌 Project Workflow

### 1. Data Collection
- Dataset: **Rainfall.csv** (366 rows, 12 columns).  
- Features: pressure, temperature, dewpoint, humidity, cloud, sunshine, winddirection, windspeed, and rainfall label (yes/no).

### 2. Data Preprocessing
- Removed extra spaces from column names.  
- Handled missing values (imputation with mode/median).  
- Converted categorical labels (`yes/no`) into binary (`1/0`).  
- Dropped highly correlated features (`maxtemp`, `temparature`, `mintemp`).  
- Balanced dataset using **downsampling**.

### 3. Exploratory Data Analysis (EDA)
- Distribution plots, boxplots, and correlation heatmaps.  
- Rainfall distribution visualization.

### 4. Model Training
- Used **RandomForestClassifier**.  
- Hyperparameter tuning with **GridSearchCV**.  
- Best parameters found:
  ```python
  {'max_depth': None, 'max_features': 'sqrt', 
   'min_samples_leaf': 2, 'min_samples_split': 10, 
   'n_estimators': 200}

5. Model Evaluation
Cross-validation mean score: 81%
Test accuracy: 70%
Confusion matrix & classification report generated.
6. Model Saving
Trained model and feature names saved with Pickle (rainfall_prediction_model.pkl).
Reload model to make predictions on new data.
🚀 Example Prediction
import pickle
import pandas as pd

# Load the trained model
with open("rainfall_prediction_model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
feature_names = model_data["feature_names"]

# Example input
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
input_df = pd.DataFrame([input_data], columns=feature_names)

# Prediction
prediction = model.predict(input_df)
print("Prediction result:", "Rainfall" if prediction[0] == 1 else "No Rainfall")
✅ Output:
Prediction result: Rainfall
📊 Results
Accuracy: ~70% on test set.
Handles both balanced dataset and unseen weather data.
Random Forest provides robust results for classification.

📦 Requirements
Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
Install dependencies:
pip install -r requirements.txt

📂 Project Structure
Rainfall-Prediction/
│── Rainfall.csv
│── Rainfall_Prediction.ipynb   # Main notebook (Colab/Jupyter)
│── rainfall_prediction_model.pkl
│── README.md
│── requirements.txt

✨ Future Improvements
Use SMOTE instead of downsampling to handle class imbalance.
Try other ML models (Logistic Regression, XGBoost, Neural Networks).
Deploy model using Flask/Streamlit for interactive web app.
Hyperparameter optimization with RandomizedSearchCV or Bayesian methods.

📝 Author
Developed by Rohit Negi
📧 Contact: rnegilxm16@gmail.com
