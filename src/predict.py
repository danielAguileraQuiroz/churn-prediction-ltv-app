import joblib
import pandas as pd

# Importación robusta de feature_engineering
try:
    from src.feature_engineering import prepare_data
except ModuleNotFoundError:
    from feature_engineering import prepare_data


def predict_churn(input_data: pd.DataFrame) -> pd.DataFrame:
    # 1. Aplicar la ingeniería de características
    df_prepared = prepare_data(input_data.copy())

    # 2. Cargar los artefactos entrenados
    model = joblib.load("models/churn_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")

    # 3. Eliminar columnas que no entran al modelo
    target_col = "Churn"
    X = df_prepared.drop(columns=[target_col, "customerID"], errors="ignore")

    # 4. Transformar e inferir
    X_prep = preprocessor.transform(X)
    probabilities = model.predict_proba(X_prep)[:, 1]
    predictions = model.predict(X_prep)

    # 5. Formatear salida
    output_df = input_data.copy()
    output_df["churn_probability"] = probabilities
    output_df["churn_prediction"] = predictions

    return output_df


if __name__ == "__main__":
    sample_customer = pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 70.35,
            }
        ]
    )

    results = predict_churn(sample_customer)
    prob = results["churn_probability"].values[0]
    pred = results["churn_prediction"].values[0]

    print("\n" + "=" * 40)
    print("      RESULTADO DE LA PREDICCIÓN        ")
    print("=" * 40)
    print(f"Probabilidad de fuga: {prob:.2%}")
    print(f"Estado predicho: {'RIESGO DE CHURN' if pred == 1 else 'RETENIDO'}")
    print("=" * 40)