# 📉 Customer Churn Prediction & Retention Strategy

Este proyecto implementa un pipeline end-to-end de Machine Learning diseñado para predecir la tasa de cancelación de clientes (*Churn*) en una empresa de telecomunicaciones. El objetivo principal es identificar proactivamente a los usuarios en riesgo de abandono para permitir intervenciones tempranas y optimizar la estrategia de retención reduciendo la pérdida de LTV (*Lifetime Value*).

---

## 🛠️ Arquitectura del Proyecto

```text
project-1-churn-ltv/
├── data/
│   └── raw/                    # Dataset original de Telco Churn
├── models/                     # Artefactos serializados (.joblib)
│   ├── churn_model.joblib
│   └── preprocessor.joblib
├── src/                        # Código fuente modular
│   ├── feature_engineering.py  # Limpieza y transformaciones aisladas
│   └── train.py                # Pipeline de entrenamiento y evaluación
├── .gitignore                  # Exclusión de entornos y datos
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación principal