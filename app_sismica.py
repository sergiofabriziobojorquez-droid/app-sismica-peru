import streamlit as st
import pandas as pd
import numpy as np
from scipy.linalg import eigh

# Configuración de página para móviles
st.set_page_config(page_title="Sismica App", layout="centered")

st.title("🏗️ Análisis Modal E.030")
st.caption("Calculadora de Frecuencias y Periodos")

# --- 1. CONFIGURACIÓN ---
st.markdown("### 1. Configuración")
n_pisos = st.number_input("Número de pisos:", min_value=1, max_value=20, value=3, step=1)

# --- 2. DATOS ---
st.markdown("### 2. Ingreso de Datos")
st.info("Despliega cada piso para editar Masa y Rigidez")

datos_masa = []
datos_rigidez = []

# Usamos expanders para ahorrar espacio en la pantalla del celular
for i in range(n_pisos):
    with st.expander(f"Piso {i+1}", expanded=(i==0)): # Solo el primero abierto por defecto
        col1, col2 = st.columns(2)
        m = col1.number_input(f"Masa P{i+1} (Tn)", value=0.0, step=1.0, key=f"m_{i}")
        k = col2.number_input(f"Rigidez P{i+1} (Tn/m)", value=0.0, step=100.0, key=f"k_{i}")
        datos_masa.append(m)
        datos_rigidez.append(k)

# --- 3. CÁLCULO ---
st.divider()
if st.button("CALCULAR RESULTADOS", type="primary", use_container_width=True):
    
    # Validar datos
    if sum(datos_masa) <= 0 or sum(datos_rigidez) <= 0:
        st.error("⚠️ Error: La masa y rigidez deben ser mayores a 0.")
    else:
        try:
            # --- Lógica Matemática (Shear Building) ---
            n = len(datos_masa)
            M = np.diag(datos_masa)
            K = np.zeros((n, n))

            for i in range(n):
                k_act = datos_rigidez[i]
                if i < n - 1:
                    k_sup = datos_rigidez[i+1]
                    K[i, i] = k_act + k_sup
                    K[i, i+1] = -k_sup
                    K[i+1, i] = -k_sup
                else:
                    K[i, i] = k_act

            # Resolver valores propios
            w2, _ = eigh(K, M)
            w = np.sqrt(np.abs(w2))

            # --- Mostrar Resultados ---
            st.success("✅ Cálculo Exitoso")
            
            resultados = []
            for idx, val in enumerate(w):
                T = 2 * np.pi / val if val > 0 else 0
                resultados.append({
                    "Modo": idx + 1,
                    "Periodo T (s)": f"{T:.4f}",
                    "ω (rad/s)": f"{val:.4f}"
                })
            
            df_res = pd.DataFrame(resultados)
            
            # Mostrar tabla limpia
            st.table(df_res)
            
        except Exception as e:
            st.error(f"Ocurrió un error matemático: {e}")