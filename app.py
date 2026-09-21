import streamlit as st
import pandas as pd
import joblib

# 1. Configuración de la ventana
st.set_page_config(page_title="Gestión de Cartera - Municipio de Mejia", layout="wide")

# 2. Títulos institucionales
st.title("💧 Sistema IA de Priorización de Cobranza")
st.subheader("Empresa Pública Municipal de Agua Potable")
st.write("Cargue el archivo CSV de facturas del mes para que el modelo XGBoost determine la prioridad de gestión.")

# 3. Cargar el cerebro de la IA
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_Mejia.pkl')

modelo = cargar_modelo()

# 4. Botón para subir archivo
archivo_subido = st.file_uploader("Subir matriz de clientes (formato CSV)", type=['csv'])

if archivo_subido is not None:
    st.info("Procesando datos con Inteligencia Artificial...")
    
    # Leer el archivo que subió el operador
    df = pd.read_csv(archivo_subido)
    
    # Salvaguardar los códigos de cliente y factura (si existen en el CSV subido)
    if 'serial_cli' in df.columns and 'serial_caf' in df.columns:
        resultados = df[['serial_cli', 'serial_caf']].copy()
    else:
        resultados = pd.DataFrame(index=df.index)
        
    # Extraer solo las variables matemáticas que la IA necesita para predecir
    # (Asegúrate de que el CSV que suba el operador tenga estas columnas)
    columnas_ia = ['numeromesesatraso_caf', 'consumo_caf', 'consumopromedio_ins', 'serial_ccl', 'valorapagar_def']
    
    try:
        datos_ia = df[columnas_ia].fillna(0) # Llenamos nulos con 0 por seguridad
        
        # 5. La IA calcula la probabilidad
        probabilidades = modelo.predict_proba(datos_ia)[:, 1]
        
        # 6. Armar la tabla de resultados
        resultados['Meses_Atraso'] = datos_ia['numeromesesatraso_caf']
        resultados['Monto_Deuda_$'] = datos_ia['valorapagar_def']
        resultados['Prob_Pago'] = probabilidades
        
        # Semáforos
        resultados['Prioridad'] = resultados['Prob_Pago'].apply(
            lambda x: '🔴 ALTA (Llamar)' if x >= 0.70 else ('🟡 MEDIA (Visitar)' if x >= 0.40 else '⚫ CRÍTICA (Coactiva)')
        )
        
        # Ordenar de mayor a menor probabilidad
        resultados = resultados.sort_values(by='Prob_Pago', ascending=False)
        
        # Mostrar en pantalla
        st.success("¡Análisis completado!")
        st.write("### 📋 Matriz Operativa de Cobranza")
        st.dataframe(resultados)
        
        # Botón de Descarga
        csv_export = resultados.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Lista de Trabajo en CSV",
            data=csv_export,
            file_name='Priorizacion_Diaria_Mejia.csv',
            mime='text/csv',
        )
        
    except Exception as e:
        st.error(f"Error: Asegúrese de que el archivo CSV contenga las columnas necesarias. Detalle: {e}")
