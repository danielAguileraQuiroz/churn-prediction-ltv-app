import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from feature_engineering import prepare_data, get_preprocessor


def train_pipeline(data_path: str):
    # Cargar datos y limpieza
    df = pd.read_csv(data_path)
    df = prepare_data(df)

    target_col = "Churn"
    df[target_col] = df[target_col].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
    X = df.drop(columns=[target_col, "customerID"], errors="ignore")
    y = df[target_col]

    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Procesamiento aislado (Data Leakage Prevention)
    preprocessor = get_preprocessor(cat_cols, num_cols)
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    # Manejo del desbalanceo y entrenamiento con XGBoost
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=4, 
        learning_rate=0.05,
        scale_pos_weight=ratio,
        random_state=42
    )
    model.fit(X_train_prep, y_train)

    # Crear carpeta models si no existe
    os.makedirs('models', exist_ok=True)
    
    # Guardado de artefactos
    joblib.dump(model, 'models/churn_model.joblib')
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    
    # Evaluación y métricas por pantalla
    y_pred_proba = model.predict_proba(X_test_prep)[:, 1]
    y_pred = model.predict(X_test_prep)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*40)
    print("      RESULTADOS DEL ENTRENAMIENTO      ")
    print("="*40)
    print(f"ROC-AUC Score: {auc:.4f}\n")
    print("Matriz de Confusión:")
    print(cm)
    print("\nInforme de Clasificación:")
    print(classification_report(y_test, y_pred))
    print("="*40)
    print("Artefactos guardados en la carpeta 'models/'")


if __name__ == "__main__":
    train_pipeline("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")