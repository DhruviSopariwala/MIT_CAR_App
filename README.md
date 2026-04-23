# 🚗 Car Price Prediction App

A machine learning web application that predicts the resale price of used cars based on key features like fuel type, transmission, ownership history, mileage, engine specs, and more.

Built with **Python**, **Scikit-learn**, and **Streamlit**.

---

## 📸 App Screenshots

> Run `streamlit run app.py` and take screenshots, then add them here as `screenshots/app_ui.png` and `screenshots/prediction_result.png`

### Input Form
![App UI](screenshots/Screenshot%202026-04-23%20213044.png)

### Prediction Result
![Prediction Result](screenshots/Screenshot%202026-04-23%20213106.png)

---

## 📁 Project Structure

```
├── app.py                      # Streamlit web app
├── models.ipynb                # Model training notebook
├── train_best_model.py         # Standalone training script
├── Car Dataset Processed.csv   # Cleaned dataset
├── final_model.pkl             # Trained best model
├── scaler.pkl                  # StandardScaler for feature scaling
├── model_info.pkl              # Model metadata & encoding dicts
└── README.md
```

---

## 📊 Dataset

The dataset contains **1499 used car listings** with the following features:

| Feature | Description |
|---|---|
| `insurance_validity` | Type of insurance (Comprehensive, Third Party, etc.) |
| `fuel_type` | Petrol / Diesel / CNG |
| `kms_driven` | Total kilometers driven |
| `ownership` | First / Second / Third / Fourth / Fifth Owner |
| `transmission` | Manual / Automatic |
| `mileage(kmpl)` | Fuel efficiency in km per litre |
| `engine(cc)` | Engine displacement in cc |
| `max_power(bhp)` | Maximum power output |
| `torque(Nm)` | Torque in Newton-metres |
| `manufacturing_year` | Year the car was manufactured |
| `seats` | Number of seats |
| `price(in lakhs)` | **Target variable** — resale price |

---

## ⚙️ How It Works

### 1. Preprocessing
- Categorical columns encoded using label mappings:
  - `insurance_validity`: Comprehensive=0, Third Party=1, Zero Dep=2, Not Available=3
  - `fuel_type`: Petrol=0, Diesel=1, CNG=2
  - `ownership`: First=1, Second=2, Third=3, Fourth=4, Fifth=5
  - `transmission`: Manual=0, Automatic=1
- `car_age` derived from `manufacturing_year` (2024 - year)
- Missing values filled with column median
- Outliers removed using IQR method on target variable

### 2. Model Training
Seven models were trained and compared:

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | L2 regularization |
| KNN-5 | K=5, uses scaled features |
| KNN-7 | K=7, uses scaled features |
| Random Forest | 100 estimators, max_depth=15 |
| Gradient Boosting | 100 estimators, lr=0.1 |
| SVM-RBF | C=100, uses scaled features |

The **best model** is automatically selected by highest R² score and saved.

### 3. Saved Artifacts
- `final_model.pkl` — best trained model
- `scaler.pkl` — fitted StandardScaler
- `model_info.pkl` — model name, feature list, scaling flag, encoding dicts

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install streamlit scikit-learn pandas numpy pickle5
```

### Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Retrain the Model
```bash
python train_best_model.py
```

---

## 🖥️ App Features

- Select insurance type, fuel type, transmission from dropdowns
- Choose ownership via radio buttons
- Adjust KMs driven with a slider
- Input mileage, engine, power, torque, year, and seats
- Click **Predict Price** to get the estimated resale value in lakhs and rupees

---

## 🛠️ Tech Stack

- **Python 3.13**
- **Pandas / NumPy** — data processing
- **Scikit-learn** — model training & evaluation
- **Streamlit** — web interface
- **Pickle** — model serialization

---

## 📌 Notes

- The `final_model.pkl` in this repo is pre-trained and ready to use
- Re-running `train_best_model.py` will overwrite the saved model with the best performer on the current run
- `scaler.pkl` must match the model — always retrain both together
