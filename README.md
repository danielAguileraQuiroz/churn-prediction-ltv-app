#  Customer Churn Prediction & Retention Strategy

Este proyecto implementa un pipeline end-to-end de Machine Learning diseñado para predecir la tasa de cancelación de clientes (*Churn*) en una empresa de telecomunicaciones. El objetivo principal es identificar proactivamente a los usuarios en riesgo de abandono para permitir intervenciones tempranas y optimizar la estrategia de retención reduciendo la pérdida de LTV.

---

##  Arquitectura del Proyecto

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


## Interfaz de Inferencia (Gradio)

Para lanzar el cuadro de mando interactivo y realizar predicciones en tiempo real:

```bash
python app_gradio.py


3 PERFILES DE CLIENTES CONTRASTADOS

Perfil 1: Cliente de Alto Riesgo (Candidato Inminente a Baja)
    Diagnóstico esperado: ⚠️ ALTO RIESGO DE CHURN (~85% - 95%)
    
    ParámetroValor a seleccionar
    Género  Female
    Jubilado / Mayor 65 No
    Tiene Pareja    No
    Tiene Dependientes      No
    Meses en la empresa (Tenure)    1
    Servicio Telefónico     Yes
    Líneas Múltiples    No
    Servicio Internet   Fiber optic
    Seguridad Online        No
    Copia de Seguridad      No
    Protección Dispositivo      No
    Soporte Técnico     No
    Streaming TV        No
    Streaming Películas     No
    Contrato        Month-to-month  
    Facturación Electrónica     Yes
    Método de Pago      Electronic check    
    Cargo Mensual ($)       70.35
    Cargos Totales ($)      70.35
    
    Qué verás en SHAP: Predominio absoluto de barras rojas hacia la derecha. La combinación de poca antigüedad, contrato mes a mes y pago por cheque electrónico empuja el riesgo hacia arriba.
    
    
    Perfil 2: Cliente Fidelizado (Bajo Riesgo / Retenido)
        Diagnóstico esperado: ✅ CLIENTE RETENIDO / BAJO RIESGO (~2% - 8%)
        ParámetroValor a seleccionar
        Género      Male
        Jubilado / Mayor 65     No
        Tiene Pareja        Yes
        Tiene Dependientes      Yes
        Meses en la empresa (Tenure)65
        Servicio TelefónicoYes
        Líneas MúltiplesYes
        Servicio InternetDSL
        Seguridad OnlineYes
        Copia de SeguridadYes
        Protección DispositivoYes
        Soporte TécnicoYes
        Streaming TVNo
        Streaming PelículasNo
        ContratoTwo year
        Facturación ElectrónicaNo
        Método de PagoCredit card (automatic)
        Cargo Mensual ($)60.00
        Cargos Totales ($)3900.00
        
        Qué verás en SHAP: Predominio de barras azules hacia la izquierda. La alta antigüedad, el contrato a 2 años y tener servicios de soporte/seguridad neutralizan casi por completo el riesgo.
        
        
        Perfil 3: Caso Ambiguo / Umbral Medio
        
        Diagnóstico esperado: RIESGO MODERADO (~40% - 55%)
        Parámetro   Valor a seleccionar
        GéneroFemale
        Jubilado / Mayor 65Sí
        Tiene ParejaYes
        Tiene DependientesNo
        Meses en la empresa (Tenure)18
        Servicio TelefónicoYes
        Líneas MúltiplesYes
        Servicio InternetFiber optic
        Seguridad OnlineYes
        Copia de SeguridadNo
        Protección DispositivoYes
        Soporte TécnicoNo
        Streaming TVYes
        Streaming PelículasYes
        ContratoOne year
        Facturación ElectrónicaYes
        Método de PagoBank transfer (automatic)
        Cargo Mensual ($)95.50
        Cargos Totales ($)1719.00


        Qué verás en SHAP: Un estira y afloja entre barras rojas y azules. El contrato anual y el pago automático reducen el riesgo (azules), pero la fibra óptica con cuotas altas ($95.50) y la falta de soporte técnico tiran en sentido contrario (rojas).