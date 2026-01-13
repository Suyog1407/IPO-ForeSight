# 📈 IPO ForeSight- Intelligent Draft Prospectus Analysis System

## 🌟 Overview
IPO ForeSight is an AI-powered analytics and forecasting project designed to analyze Initial Public Offerings (IPOs) using historical market data and IPO-related news. The goal of the project is to support informed, data-driven IPO investment decisions.

## ⚠️ Problem Statement
IPO investments are highly uncertain due to limited historical data, market volatility, and unstructured information. Retail investors often lack analytical tools to evaluate IPOs before subscription.

## 🔍 Solution
IPO ForeSight provides structured analysis by combining IPO data analysis, news scraping, and machine learning models to generate meaningful insights and forecasts.

## 🚀 Key Features
- IPO data analysis and visualization
- Automated IPO-related news scraping
- Machine learning-based performance forecasting
- Modular and scalable project architecture

## 🛠️ Tech Stack

| Category | Tools |
| --- | --- |
| **Language** | `Python` |
| **Data Science** | `Pandas`, `NumPy`, `Scikit-learn` |
| **Visualization** | `Matplotlib`, `Seaborn` |
| **Extraction** | `BeautifulSoup`, `Requests` |
| **DevOps** | `Docker` (Optional) |

## 🔄 Project Workflow

The system follows a structured pipeline to convert raw data into investment signals:

1. **Data Collection:** Gathering historical IPO prospectus data.
2. **Preprocessing:** Cleaning financial metrics and handling missing data.
3. **Sentiment Analysis:** Scraping news headlines to quantify market mood.
4. **Feature Engineering:** Creating high-impact variables (e.g., P/E ratio, Sector growth).
5. **Model Training:** Training Regressors/Classifiers to predict performance.
6. **Insights:** Generating a final "Score" or prediction for the user.

## 📂 Project Structure

```text
IPO-ForeSight/
├── src/                # Core logic & ML models
├── news_scraper/       # Web scraping modules
├── docs/               # Documentation & Research
├── app.py              # Main Application entry
├── run_app.py          # Execution script
├── Dockerfile          # Containerization (Optional)
└── requirements.txt    # Project dependencies

```
## ⚙️ Setup & Installation

Get the project running locally in minutes:

```bash
# 1. Clone the repository
git clone https://github.com/SuyogKshirsagar/IPO-ForeSight.git

# 2. Navigate to directory
cd IPO-ForeSight

# 3. Install requirements
pip install -r requirements.txt

# 4. Launch the application
python run_app.py

```

## 📊 Results & Impact

* **Feature Importance:** Identified that *Grey Market Premium (GMP)* and *Subscription Ratios* are the strongest predictors.
* **Sentiment Correlation:** Proven that positive news volume in the week prior to listing correlates with higher listing gains.
* **Accuracy:** Achieved significant predictive reliability compared to baseline random-guess strategies.

## 🔮 Future Roadmap

* [ ] **Real-time API:** Integration with live market feeds.
* [ ] **Deep Learning:** Implementing LSTM/Transformers for time-series news analysis.
* [ ] **Dashboard:** A full-stack web interface using Streamlit or React.

**Author:** [Suyog Kshirsagar](https://www.google.com/search?q=https://github.com/SuyogKshirsagar)

*Note: This project is intended for academic and learning purposes.*
