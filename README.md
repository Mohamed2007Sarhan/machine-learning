# 🤖 ML Workflow Generator

A comprehensive, AI-powered tool to automate the Machine Learning workflow. From researching data to training models and chatting with them, this app handles it all.

## 🌟 Features

*   **🔍 AI Research & Data Gathering**: Enter a topic (e.g., "California Housing Prices") and let the app scrape the web and aggregate data for you.
*   **🧹 Smart Preprocessing**: Uses **Gemini 3 Pro** (with fallback to 2.5/2.0) to clean, structure, and convert messy text into a usable CSV dataset.
*   **🧠 Advanced Model Selection**:
    *   **Standard**: Auto-selects Scikit-Learn/XGBoost models based on data type (Classification vs. Regression).
    *   **Hugging Face**: Fine-tune state-of-the-art Transformers (like `distilbert`) on your data.
    *   **Local Models**: Load your own `.pkl` or `.joblib` models.
*   **💬 Chat with your Model**: accurate "Chat" interface to test your trained model with new inputs.

## 🚀 Installation

1.  **Clone/Download** this repository.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you have issues on Windows, use `py -m pip install -r requirements.txt`.*

3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

## 🛠 Usage Guide

### 1. Research
Type a topic. The app searches DuckDuckGo, finding relevant datasets or tables.
*   *Tip*: You can manually edit the gathered text before processing.

### 2. Preprocessing
Gemini analyzes the text and creates a DataFrame.
*   *Note*: If you see "Rate Limit" warnings, wait a few seconds. The app has auto-retry built in!

### 3. Training
*   **Standard**: Choose a model from the list (e.g., Random Forest, XGBoost).
*   **Hugging Face**: Enter a model ID (e.g., `bert-base-uncased`) to fine-tune it.
*   **Local**: Upload a model file.

### 4. Chat
Interact with your model!
*   **Tabular**: Enter values for features -> Get prediction.
*   **Text**: Type a sentence -> Get classification.

## 📦 Requirements
*   Python 3.8+
*   `streamlit`
*   `google-genai`
*   `scikit-learn`
*   `transformers`
*   `torch`
*   `duckduckgo-search`

---
*Built with ❤️ Mohamed Sarhan*

