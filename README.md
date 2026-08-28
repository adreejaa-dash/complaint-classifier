# E-Commerce Customer Complaint Classifier

An end-to-end NLP and machine learning system that automatically classifies e-commerce customer complaints into actionable support categories, estimates prediction confidence, highlights influential words behind the prediction, and recommends the appropriate support team for faster ticket routing.

## Overview

Customer support teams often receive large volumes of unstructured complaints that need to be manually reviewed, categorized, and routed to the appropriate department.

This project automates that initial triage process using Natural Language Processing and supervised machine learning. A customer complaint is cleaned and transformed into TF-IDF features, classified into one of nine support categories, and presented with:

- Predicted complaint category
- Prediction confidence
- Model performance
- Influential words contributing to the prediction
- Suggested support team for routing

The project includes the complete pipeline from data preprocessing and model comparison to model persistence and deployment through a Flask web application.

## Key Features

- **Automated Complaint Classification**  
  Classifies customer support complaints into nine e-commerce issue categories.

- **NLP Preprocessing Pipeline**  
  Cleans raw complaint text through lowercasing, URL removal, special-character removal, tokenization, and stopword removal.

- **TF-IDF Feature Engineering**  
  Converts processed complaint text into numerical features using unigram and bigram representations.

- **Model Comparison**  
  Evaluates Logistic Regression, Multinomial Naive Bayes, and XGBoost using accuracy, macro precision, macro recall, and macro F1-score.

- **Confidence Scoring**  
  Displays the probability associated with the predicted complaint category.

- **Prediction Explainability**  
  Identifies influential words from the complaint that contribute positively toward the predicted class.

- **Intelligent Ticket Routing**  
  Maps predicted complaint categories to appropriate support teams such as Billing, Shipping & Fulfillment, Technical Support, and Returns & Refunds.

- **Web Application**  
  Provides a simple Flask interface where users can enter complaints and receive predictions.

- **Docker Support**  
  Includes Docker and Docker Compose configuration for containerized execution.

## Complaint Categories

The classifier maps the original dataset intents into nine broader, actionable support categories:

| Category | Suggested Team |
|---|---|
| Order Cancellation | Orders Team |
| Order Not Received / Shipping Delay | Shipping & Fulfillment |
| Billing / Payment Issue | Billing Department |
| Refund / Return Request | Returns & Refunds |
| Damaged / Defective Product | Returns & Refunds |
| Wrong Item Delivered | Returns & Refunds |
| Account / Login Issue | Technical Support |
| Product Quality / Not as Described | Quality Assurance |
| Customer Service Complaint | Escalation Team |

## Dataset

The project uses the **Bitext Customer Support LLM Chatbot Training Dataset** available through Hugging Face.

- Approximately **26,800 customer-support tickets**
- Input: `instruction`
- Original target: `intent`
- Final target: mapped multi-class complaint category
- Nine consolidated e-commerce support categories

The original intents are mapped into broader operational categories so that predictions can be directly connected to support workflows.

## NLP Pipeline

The preprocessing and feature engineering pipeline consists of:

1. Convert text to lowercase
2. Remove URLs
3. Remove punctuation, numbers, and special characters
4. Tokenize using NLTK
5. Remove English stopwords
6. Remove single-character tokens
7. Generate TF-IDF features
8. Include both unigrams and bigrams
9. Limit the feature space to 8,000 features
10. Train and evaluate multiple classification models

The same preprocessing function is used during both model training and inference to keep the prediction pipeline consistent.

## Machine Learning Pipeline

The dataset is split into training and testing sets using an 80/20 stratified split.

Three classification approaches are evaluated:

### Logistic Regression

A linear classification model that works well with sparse TF-IDF text representations.

### Multinomial Naive Bayes

A probabilistic model particularly suited to discrete text features and sparse document representations.

### XGBoost

A gradient-boosted tree model evaluated as a non-linear alternative to the linear and probabilistic approaches.

The final model is selected automatically based on the highest macro F1-score.

## Model Performance

Current evaluation results on the held-out test set:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 99.67% | 99.69% | 99.56% | 99.63% |
| **Multinomial Naive Bayes** | **99.74%** | **99.78%** | **99.65%** | **99.71%** |
| XGBoost | 99.24% | 99.18% | 99.08% | 99.13% |

**Best-performing model: Multinomial Naive Bayes**

The training pipeline automatically selects the model with the highest macro F1-score rather than hard-coding a specific algorithm.

## Prediction Explainability

Along with the predicted category, the application identifies words that contributed most strongly toward the prediction.

A Logistic Regression model is retained specifically for this purpose because its learned coefficients can be mapped back to TF-IDF feature names.

This provides a lightweight form of model interpretability and helps users understand which terms in a complaint influenced the classification.

## Application Workflow

```text
Customer Complaint
        |
        v
Text Preprocessing
        |
        v
TF-IDF Vectorization
        |
        v
Trained ML Model
        |
        v
Predicted Category
        |
        +--------------------+
        |                    |
        v                    v
Confidence Score       Influential Words
        |
        v
Suggested Support Team
```

## Project Structure

```text
complaint-classifier/
│
├── app.py
├── preprocess.py
├── train.py
│
├── templates/
│   └── index.html
│
├── model.pkl
├── lr_model.pkl
├── vectorizer.pkl
├── label_encoder.pkl
├── model_info.pkl
│
├── confusion_matrix.png
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── test_preprocess.py
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/adreejaa-dash/complaint-classifier.git
cd complaint-classifier/complaint-classifier
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required NLTK Resources

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Training the Models

The training pipeline expects the customer-support dataset at:

```text
data/hf_customer_support.csv
```

with the following columns:

```text
instruction
intent
```

Run the training script:

```bash
python train.py
```

The training pipeline:

- Loads and cleans the dataset
- Maps the original intents into broader complaint categories
- Removes unsupported or low-frequency categories
- Performs an 80/20 stratified train-test split
- Generates TF-IDF features using unigrams and bigrams
- Trains multiple classification models
- Evaluates model performance using accuracy, precision, recall, and macro F1-score
- Automatically selects the best-performing model based on macro F1-score
- Saves the trained model and preprocessing artifacts
- Generates a confusion matrix

## Running the Web Application

After the trained model artifacts are available, run:

```bash
python app.py
```

The application will be available at:

```text
http://127.0.0.1:5001
```

Enter a customer complaint into the interface to receive:

- Predicted complaint category
- Prediction confidence
- Model information
- Influential words contributing to the prediction
- Recommended support team

## Running with Docker

### Build the Image

```bash
docker build -t ecom-classifier .
```

### Run the Container

```bash
docker run -p 5001:5001 ecom-classifier
```

Then open:

```text
http://localhost:5001
```

### Using Docker Compose

```bash
docker compose up --build
```

The application will be available at:

```text
http://localhost:5001
```

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **NLTK**
- **TF-IDF**
- **Flask**
- **Matplotlib**
- **Seaborn**
- **Joblib**
- **Docker**
- **Docker Compose**
- **HTML/CSS**

## Why This Project?

This project demonstrates an end-to-end machine learning workflow for automating customer-support ticket classification.

It transforms unstructured customer complaints into actionable support categories using NLP and supervised machine learning, while also providing prediction confidence, explainability, and automated team routing.

**Raw Complaint → Text Preprocessing → TF-IDF Features → Model Prediction → Complaint Category → Explanation → Support-Team Routing**

The project goes beyond model training by integrating the trained classifier into a Flask web application and adding a practical routing layer for customer-support workflows.

## Future Improvements

- Improve confidence calibration for more reliable prediction probabilities
- Perform deeper class-wise error analysis
- Experiment with transformer-based NLP models
- Add batch complaint classification through CSV upload
- Provide a REST API for external integrations
- Add persistent storage for classified complaints
- Implement automated model retraining with newly labeled data
- Allow configurable support-team routing rules

## License

This project is intended for educational and portfolio purposes.


Then continue with:

```markdown
## Installation
