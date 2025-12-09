# 🍷 Wine Quality Analysis & Prediction (ML + Streamlit)

End-to-end machine learning project to **analyze** and **predict wine quality** using physicochemical properties of wines.  
The project includes:

- Exploratory Data Analysis (EDA)
- Multiple classification models with performance comparison
- An interactive **Streamlit web app** to predict wine quality for new samples

---

## 🔍 Problem Statement

Given the physicochemical properties of red and white wines (such as acidity, alcohol, sulphates, etc.),  
the goal is to **predict the quality score** (`quality`, integer) on a 0–10 scale.

This can help:
- Winemakers quickly estimate wine quality
- Businesses maintain quality control
- Data science learners practice an end-to-end ML workflow

---

## 📁 Dataset

- File: `data/Wine_Quality_Data.csv`
- Each row represents one wine sample.
- Key columns used in the model:

Numeric features:
- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`

Categorical / engineered features:
- `good` – binary label indicating if the wine is considered good (yes/no)
- `color` – wine type (`red` / `white`)
- `quality` – **target variable** (integer quality score)
- `quality_category` – helper label (e.g., "Low", "Medium", "High") used for analysis

> Note: An index column like `Unnamed: 0` is dropped during preprocessing.

---

## 🧠 Machine Learning Approach

1. **Data Cleaning & Preparation**
   - Dropped unnecessary index columns.
   - Ensured the target variable `quality` is properly defined.
   - Used helper labels like `good` and `color` as additional features.

2. **Feature Engineering**
   - Selected feature set:
     - Numeric: `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`,  
       `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`,  
       `sulphates`, `alcohol`
     - Categorical: `good`, `color`
   - Train–test split: `train_test_split(..., train_size=0.75, random_state=100)`

3. **Preprocessing Pipeline**
   - **Scaling numeric features** using `StandardScaler`
   - **Encoding categorical features** (`good`, `color`) using `OneHotEncoder(drop="first")`
   - Combined scaled numeric and encoded categorical features into a single feature matrix

4. **Models Trained**
   The app trains and compares multiple classification models:

   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Gaussian Naive Bayes

5. **Evaluation Metrics**
   - Accuracy
   - Precision (weighted)
   - Recall (weighted)
   - F1-Score (weighted)

   The app automatically **sorts models by F1-Score** and selects the best model for predictions.

---

## 🌐 Streamlit Web App

The Streamlit app (`streamlit_app.py`) provides a full interactive interface with four main pages:

1. **Project Overview**
   - Short description of the project
   - Dataset snapshot (`head`, `shape`, `describe`)
   - Information about target and helper columns (`quality`, `good`, `color`, etc.)

2. **Data Exploration**
   - View top rows of the dataset
   - Display numeric vs categorical feature lists
   - **Quality distribution** plot (countplot of `quality`)
   - **Feature distribution**: select a numeric feature and see its histogram + KDE
   - **Correlation heatmap** between numeric features and `quality`

3. **Model Performance**
   - Table comparing all models by Accuracy, Precision, Recall, F1-Score
   - Highlight of **best model** by F1-Score
   - Detailed **classification report** for the selected model
   - **Confusion matrix** visualization as a heatmap

4. **Predict Wine Quality**
   - User-friendly input controls:
     - Number inputs for all numeric features (acidity, sugar, chlorides, sulphates, alcohol, etc.)
     - Selectboxes for `good` (yes/no) and `color` (red/white)
   - Uses the **best performing model** from training
   - Shows predicted quality score and a simple interpretation:
     - 0–4 → Low Quality
     - 5–6 → Medium Quality
     - 7–10 → High Quality

---

## 🛠 Tech Stack

- **Language**: Python
- **Web App**: Streamlit
- **Data Handling**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: scikit-learn

---

## 📦 Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/wine-quality-analysis.git
cd wine-quality-analysis

```

## Output:


https://github.com/user-attachments/assets/45e791b9-658e-4ad3-bfbd-cdb8125b0397

