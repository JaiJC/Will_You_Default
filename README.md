# Credit Card Default Prediction 💳

A machine learning project that predicts whether customers will default on their credit card payments using advanced ensemble methods and feature analysis.

## 🎯 Project Overview

Credit card defaults pose significant challenges for financial institutions, impacting profitability and operational efficiency. This project develops a predictive model to identify customers likely to default, enabling proactive interventions such as adjusting credit limits or offering personalized repayment plans.

**Problem Type**: Binary Classification (Default: 1, No Default: 0)

## 📊 Dataset

- **Source**: Default of Credit Card Clients Dataset (Kaggle)
- **Size**: 30,000 credit card holders from Taiwan
- **Features**: 23 features including:
  - **Demographic Information**: Age, gender, education, marital status
  - **Financial Attributes**: Credit limit, bill amounts, payment history
  - **Repayment History**: 6 months of payment status data
- **Target Variable**: Binary indicator of default (0/1)

### Key Data Insights
- Strong correlation between bill amounts across months (expenditure smoothing behavior)
- Surprising independence between bill amounts and payment amounts
- Younger population heavily represented in the dataset
- Moderate class imbalance (not severe enough to cause major issues)

## 🔧 Models Tested

| Model | AUC-ROC Score | Notes |
|-------|---------------|-------|
| Dummy Classifier | 0.50 | Baseline |
| Logistic Regression | ~0.82 | Good interpretability |
| **LightGBM** | **0.85** | **Best performance** |

### Final Model Performance (LightGBM)
- **AUC-ROC**: 0.85
- **Precision** (Default class): 0.76
- **Recall** (Default class): 0.73
- **Overall Accuracy**: 82%

## 🚀 Key Features

- **Advanced Feature Analysis**: Correlation analysis revealing spending pattern consistency
- **Efficient Model Selection**: LightGBM chosen for optimal performance-efficiency balance
- **No Overfitting**: Robust model that generalizes well to unseen data
- **Economic Theory Integration**: Expenditure smoothing concepts applied to feature understanding

## 📈 Results & Impact

The optimized LightGBM model demonstrates strong predictive power with an AUC-ROC of 0.85, significantly outperforming the baseline. The model effectively identifies true defaults while minimizing false positives, making it suitable for real-world deployment.

## ⚠️ Limitations & Future Work

### Current Challenges
1. **Class Imbalance Awareness**: While 82% accuracy seems good, baseline accuracy would be 77% - highlighting the need for better evaluation metrics
2. **Multicollinearity**: Strong correlations among bill amounts may require feature engineering
3. **Outlier Sensitivity**: Extreme values in bill and payment amounts could affect predictions
4. **Rare Category Handling**: Some education and marital status categories have insufficient data

### Planned Improvements
- [ ] Implement advanced class imbalance techniques (SMOTE, cost-sensitive learning)
- [ ] Create composite features to address multicollinearity
- [ ] Develop robust outlier detection and handling
- [ ] Group rare categories for better model stability
- [ ] Focus on business-relevant metrics beyond accuracy

## 🛠️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-default-prediction.git
cd credit-card-default-prediction

# Install required packages
pip install -r requirements.txt

# Run the model training
python train_model.py

# Make predictions
python predict.py --input data/new_customers.csv
```

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.2.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📂 Project Structure

```
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned and preprocessed data
├── notebooks/
│   ├── EDA.ipynb           # Exploratory Data Analysis
│   └── model_training.ipynb # Model development
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── models/
│   └── lightgbm_model.pkl  # Saved model
├── requirements.txt
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Jai Choraria**
- Medium: [@chorariajai17](https://medium.com/@chorariajai17)
- LinkedIn: [https://www.linkedin.com/in/jai-choraria/]
- Email: chorariajai17@gmail.com

## 🙏 Acknowledgments

- Kaggle for providing the Default of Credit Card Clients Dataset
- The open-source community for the amazing machine learning libraries
- Economic theory insights that helped understand customer spending patterns

---

*If you found this project helpful, please consider giving it a ⭐!*
