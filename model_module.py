import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def analyze_dataset(df):
    """
    Analyzes dataset size and target type (using heuristic).
    Assumes last column is target.
    """
    rows, cols = df.shape
    size_category = "large" if rows > 1000 else "small"
    
    target_col = df.columns[-1]
    unique_vals = df[target_col].nunique()
    
    # Heuristic: 
    # 1. If explicitly object/string -> Classification
    # 2. If numeric but very few unique values (< 10) -> Classification
    # 3. If numeric and many unique values -> Regression
    # 4. Special case: If numeric and < 20 unique but looks like float prices -> Regression
    
    is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
    
    if not is_numeric:
        problem_type = "classification"
    elif unique_vals < 10:
        problem_type = "classification"
    else:
        # It is numeric and >= 10 unique values
        problem_type = "regression"
        
    print(f"Dataset Size: {rows} rows (Category: {size_category})")
    print(f"Problem Type: {problem_type} (Target: {target_col})")
    
    return size_category, problem_type, target_col

def select_models(size_category, problem_type):
    """
    Returns a dictionary of available models based on criteria.
    """
    models = {}
    
    if size_category == "large":
        if problem_type == "classification":
            models = {
                "XGBoost": xgb.XGBClassifier(eval_metric='logloss'),
                "LightGBM": lgb.LGBMClassifier(verbose=-1),
                "CatBoost": cb.CatBoostClassifier(verbose=0),
                "RandomForest": RandomForestClassifier() # Added as backup
            }
        else:
            models = {
                "XGBoost": xgb.XGBRegressor(),
                "LightGBM": lgb.LGBMRegressor(verbose=-1),
                "CatBoost": cb.CatBoostRegressor(verbose=0),
                "RandomForest": RandomForestRegressor()
            }
    else: # small
        if problem_type == "classification":
            models = {
                "Logistic Regression": LogisticRegression(),
                "kNN": KNeighborsClassifier(),
                "SVM": SVC(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Naive Bayes": GaussianNB(),
                "Gradient Boosting": GradientBoostingClassifier()
            }
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "SVR": SVR(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }
            
    return models

def train_and_evaluate(df, target_col, selected_model_name, model_instance):
    """
    Trains and evaluates the selected model.
    """
    X = df.drop(columns=[target_col])
    # Handle categorical variables if any (simple encoding)
    X = pd.get_dummies(X, drop_first=True)
    y = df[target_col]
    
    # Encode target if classification
    if y.dtype == 'object' or (len(y.unique()) < 10 and not pd.api.types.is_numeric_dtype(y)): 
         from sklearn.preprocessing import LabelEncoder
         le = LabelEncoder()
         y = le.fit_transform(y)
         mapping = dict(zip(le.classes_, le.transform(le.classes_)))
         print(f"Target Encoded: {mapping}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining {selected_model_name}...")
    model_instance.fit(X_train, y_train)
    
    print("\n--- Evaluation ---")
    
    metrics = {}
    
    if hasattr(model_instance, "predict_proba") or hasattr(model_instance, "predict"): # Classification check usually
        # We use the problem type from analysis implicitly, but here we can check model type
        is_classifier = any(x in str(type(model_instance)) for x in ['Classifier', 'SVC', 'Logistic', 'NB'])
        
        preds_train = model_instance.predict(X_train)
        preds_test = model_instance.predict(X_test)
        
        if is_classifier:
            metrics['Train Accuracy'] = accuracy_score(y_train, preds_train)
            metrics['Test Accuracy'] = accuracy_score(y_test, preds_test)
            print(f"Train Accuracy: {metrics['Train Accuracy']:.4f}")
            print(f"Test Accuracy: {metrics['Test Accuracy']:.4f}")
        else:
            metrics['Train MSE'] = mean_squared_error(y_train, preds_train)
            metrics['Test MSE'] = mean_squared_error(y_test, preds_test)
            metrics['Test R2'] = r2_score(y_test, preds_test)
            print(f"Train MSE: {metrics['Train MSE']:.4f}")
            print(f"Test MSE: {metrics['Test MSE']:.4f}")
            print(f"Test R2: {metrics['Test R2']:.4f}")
            
    return model_instance, X_test, y_test, metrics

    return model_instance, X_test, y_test, metrics

def train_hf_model(df, target_col, model_name="distilbert-base-uncased"):
    """
    Fine-tunes a Hugging Face model (simplified).
    Treats each row as a text string for sequence classification.
    """
    print(f"preparing HF model: {model_name}...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    import torch
    import numpy as np

    # 1. Prepare Data
    # Convert row to text
    df['text'] = df.drop(columns=[target_col]).astype(str).agg(' '.join, axis=1)
    
    # Encode labels
    labels = df[target_col]
    unique_labels = sorted(labels.unique())
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    df['label'] = labels.map(label2id)
    
    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2
    )

    # 2. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)

    # 3. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
    )

    # 4. Train
    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=1,              
        per_device_train_batch_size=8,  
        per_device_eval_batch_size=16,   
        warmup_steps=10,                
        weight_decay=0.01,               
        logging_dir='./logs',            
        logging_steps=10,
        no_cuda=True # Force CPU for compatibility if GPU not set up
    )

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=val_dataset             
    )

    trainer.train()
    
    # 5. Evaluate
    metrics = trainer.evaluate()
    # Normalize keys
    metrics_out = {
        'Test Accuracy': metrics.get('eval_accuracy', metrics.get('eval_loss', 0)), # Trainer default eval is loss often
        'Test Loss': metrics.get('eval_loss')
    }
    
    return trainer, None, None, metrics_out

def load_local_model(path):
    """
    Loads a local model file.
    """
    import joblib
    try:
        return joblib.load(path)
    except:
        return None
