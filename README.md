# 🚗 Vehicle Resale Price Prediction & Value Estimation Engine

A machine learning regression solution that estimates the fair market resale price of used vehicles based on specifications such as vehicle age, distance driven, fuel type, seller type, and transmission.

---

## 🛠️ Tech Stack & Tools
- **Language:** Python
- **Data Processing & EDA:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Linear Regression, Decision Tree Regressor, Random Forest Regressor)
- **Model Persistence:** Joblib
- **Frontend / Web App:** Streamlit

---

## 📊 Dataset & Features
The project utilizes vehicle transaction data containing key specifications:
- `kms_driven`: Odometer reading in kilometers
- `vehicle_age`: Age of vehicle in years
- `fuel_type`: Petrol, Diesel, or CNG
- `seller_type`: Dealer or Individual
- `transmission`: Manual or Automatic
- `selling_price` *(Target Variable)*: Historical resale price in Lakhs (₹)

---

## 📈 Model Performance & Evaluation

Three regression algorithms were evaluated using standard metrics on an 80/20 train-test split:

| Model | R² Score | MAE | RMSE |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | Baseline | Evaluated | Evaluated |
| **Decision Tree Regressor** | Comparison | Evaluated | Evaluated |
| **Random Forest Regressor** | **Best Performance** | Lowest | Lowest |

*The Random Forest Regressor was selected as the final production model due to its optimal accuracy and ensemble handling of non-linear feature interactions.*

---

## 🚀 How to Run Locally

### 1. Clone the Repository & Navigate to Directory
```bash
git clone [https://github.com/YOUR-USERNAME/vehicle-price-predictor.git](https://github.com/YOUR-USERNAME/vehicle-price-predictor.git)
cd vehicle-price-predictor



### Set Up Virtual Environment & Install Dependencies
Bash
python -m venv venv

# On Windows PowerShell
.\venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate

pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib notebook



vehicle-price-predictor/
├── .gitignore          # Git exclusion rules
├── README.md           # Documentation
├── app.py              # Streamlit web application interface
├── car_dataset.csv     # Vehicle transaction dataset
├── car_price_model.pkl # Trained Random Forest model file
├── model_features.pkl  # Encoded feature column structure
└── notebook.ipynb      # EDA, cleaning, model training & evaluation