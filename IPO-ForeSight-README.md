# 📈 IPO ForeSight- Intelligent Draft Prospectus Analysis System

## 🌟 Overview
IPO ForeSight is an AI-powered analytics and forecasting project designed to analyze Initial Public Offerings (IPOs) using historical market data and IPO-related news. The goal of the project is to support informed, data-driven IPO investment decisions.

## ⚠️ Problem Statement
IPO investments are highly uncertain due to limited historical data, market volatility, and unstructured information. Retail investors often lack analytical tools to evaluate IPOs before subscription.

## 🔍 Solution
IPO ForeSight provides structured analysis by combining IPO data analysis, news scraping, and machine learning models to generate meaningful insights and forecasts.

## 🚀 Key Features
- Automated analysis of DRHP documents for IPO evaluation
- Intelligent anomaly detection for missing, inconsistent, or abnormal content
- Vector database–based plagiarism detection across IPO documents
- Interactive dashboard with section-wise and severity-based insights
- AI-powered chatbot for natural language queries on analysis results
- Automated generation of summary and detailed analysis reports

## 🛠️ Tech Stack

| Category | Tools |
| --- | --- |
| **Programming Language** | `Python` |
| **Data Processing** | `Pandas`, `NumPy` |
| **Machine Learning** | `Scikit-learn` |
| **NLP & Similarity** | `Vector embeddings`, `cosine similarity` |
| **Visualization** | `Matplotlib`, `Seaborn` |
| **Web Scraping** | `BeautifulSoup`, `Requests` |
| **Backend / App** | `Python (Streamlit-based interface)` |
| **DevOps** | `Docker` |

## 🔄 Project Workflow

**1. Document Upload** :
User uploads the DRHP (IPO prospectus) PDF through the application interface.

**2. Text Extraction** :
Relevant text is extracted from the uploaded document using PDF parsing techniques.

**3. Pre-Processing** : 
Extracted text is cleaned, normalized, and prepared for analysis.

**4. Section Identification** :
The document is segmented into logical sections such as risk factors, capital structure, and management details.

**5. Anomaly Detection** :
The system identifies content anomalies such as missing data, unusually long sentences, and formatting issues.

**6. Plagiarism Detection** :
Vector-based similarity checks are performed to detect potential plagiarism across documents.

**7. Interactive Dashboard** :
Analysis results are visualised through dashboards with severity levels and section-wise insights.

**8. AI Chatbot Interaction** :
Users query the document and analysis results using a natural-language chatbot.

**9. Report Generation** :
Summary and detailed reports are generated for review and compliance purposes.

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

- **Automated DRHP Processing:** Successfully analyzed DRHP documents with accurate section-wise extraction and structured parsing.  
- **Anomaly Detection:** Identified content issues such as missing information, long sentences, and formatting inconsistencies.  
- **Plagiarism Identification:** Detected similarity risks using vector-based document comparison techniques.  
- **Insightful Visualization:** Presented analysis results through interactive dashboards, reducing manual review effort.  
- **Actionable Reporting:** Generated summary and detailed reports to support IPO due diligence and compliance review.  

## 🔮 Future Scope

- **Real-Time Market Integration:** Integrate live IPO and financial market data using external APIs.  
- **Advanced NLP Models:** Apply LSTM and Transformer-based models for deeper semantic and contextual analysis.  
- **Enhanced Plagiarism Detection:** Expand similarity checks across broader regulatory and market datasets.  
- **Enterprise Dashboard:** Develop a scalable web-based dashboard with role-based access control.  
- **Multi-Document Support:** Extend the system to analyze additional regulatory and compliance documents beyond DRHP.  


**Author:** [Suyog Kshirsagar](https://www.google.com/search?q=https://github.com/SuyogKshirsagar)

*Note: This project is intended for academic and learning purposes.*
