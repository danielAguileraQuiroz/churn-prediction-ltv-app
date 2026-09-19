import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gradio as gr
import pandas as pd
from src.explainability import generate_shap_plot
from src.predict import predict_churn


def predict_interface(
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
    TotalCharges,
):
    input_data = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": 1 if SeniorCitizen == "Sí" else 0,
                "Partner": Partner,
                "Dependents": Dependents,
                "tenure": tenure,
                "PhoneService": PhoneService,
                "MultipleLines": MultipleLines,
                "InternetService": InternetService,
                "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": OnlineBackup,
                "DeviceProtection": DeviceProtection,
                "TechSupport": TechSupport,
                "StreamingTV": StreamingTV,
                "StreamingMovies": StreamingMovies,
                "Contract": Contract,
                "PaperlessBilling": PaperlessBilling,
                "PaymentMethod": PaymentMethod,
                "MonthlyCharges": MonthlyCharges,
                "TotalCharges": TotalCharges,
            }
        ]
    )

    result = predict_churn(input_data)
    prob = result["churn_probability"].values[0]
    pred = result["churn_prediction"].values[0]

    estado = (
        "⚠️ ALTO RIESGO DE CHURN"
        if pred == 1
        else "✅ CLIENTE RETENIDO / BAJO RIESGO"
    )
    prob_str = f"{prob:.2%}"

    # Generar gráfico SHAP con nombres limpios
    shap_img_path = generate_shap_plot(input_data)

    return estado, prob_str, shap_img_path


demo = gr.Interface(
    fn=predict_interface,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Género", value="Female"),
        gr.Dropdown(["No", "Sí"], label="Jubilado / Mayor 65", value="No"),
        gr.Dropdown(["Yes", "No"], label="Tiene Pareja", value="No"),
        gr.Dropdown(["Yes", "No"], label="Tiene Dependientes", value="No"),
        gr.Slider(
            0, 72, value=1, step=1, label="Meses en la empresa (Tenure)"
        ),
        gr.Dropdown(["Yes", "No"], label="Servicio Telefónico", value="Yes"),
        gr.Dropdown(
            ["Yes", "No", "No phone service"],
            label="Líneas Múltiples",
            value="No",
        ),
        gr.Dropdown(
            ["DSL", "Fiber optic", "No"],
            label="Servicio Internet",
            value="Fiber optic",
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Seguridad Online",
            value="No",
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Copia de Seguridad",
            value="No",
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Protección Dispositivo",
            value="No",
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Soporte Técnico",
            value="No",
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Streaming TV",
            value="No",
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Streaming Películas",
            value="No",
        ),
        gr.Dropdown(
            ["Month-to-month", "One year", "Two year"],
            label="Contrato",
            value="Month-to-month",
        ),
        gr.Dropdown(
            ["Yes", "No"], label="Facturación Electrónica", value="Yes"
        ),
        gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            label="Método de Pago",
            value="Electronic check",
        ),
        gr.Number(label="Cargo Mensual ($)", value=70.35),
        gr.Number(label="Cargos Totales ($)", value=70.35),
    ],
    outputs=[
        gr.Textbox(label="Diagnóstico del Cliente"),
        gr.Textbox(label="Probabilidad de Abandono"),
        gr.Image(label="Explicabilidad del Modelo (Factores SHAP)"),
    ],
    title="📉 Predictor de Churn con Explicabilidad SHAP",
    description="Ajuste los parámetros del cliente para obtener la predicción y el desglose de factores de decisión del modelo XGBoost.",
)

if __name__ == "__main__":
    demo.launch()