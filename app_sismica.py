import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# --- Configuración Visual ---
st.set_page_config(page_title="Sismica Pro", layout="centered")

st.title("🏗️ Análisis Modal Espectral")
st.caption("Frecuencias, Periodos y Formas Modales")

# --- 1. CONFIGURACIÓN E INPUTS ---
with st.expander("📝 Configuración y Datos", expanded=True):
    n_pisos = st.number_input("Número de pisos:", min_value=1, max_value=20, value=3)
    st.write("---")
    
    col_h, col_d = st.columns([1, 2])
    
    datos_masa = []
    datos_rigidez = []
    
    # Input optimizado para celular
    for i in range(n_pisos):
        c1, c2 = st.columns(2)
        m = c1.number_input(f"Masa P{i+1} (Tn-s²/m)", value=10.0, step=1.0, key=f"m{i}")
        k = c2.number_input(f"Rigidez P{i+1} (Tn/m)", value=1000.0, step=100.0, key=f"k{i}")
        datos_masa.append(m)
        datos_rigidez.append(k)

# --- BOTÓN DE CÁLCULO ---
st.write("---")
if st.button("🚀 CALCULAR ANÁLISIS MODAL", type="primary", use_container_width=True):
    
    if sum(datos_masa) == 0 or sum(datos_rigidez) == 0:
        st.error("⚠️ La masa y rigidez no pueden ser cero.")
    else:
        # --- MOTOR DE CÁLCULO ---
        n = len(datos_masa)
        M = np.diag(datos_masa)
        K = np.zeros((n, n))

        # Armado de Matriz de Rigidez (Shear Building)
        for i in range(n):
            k_act = datos_rigidez[i]
            if i < n - 1:
                k_sup = datos_rigidez[i+1]
                K[i, i] = k_act + k_sup
                K[i, i+1] = -k_sup
                K[i+1, i] = -k_sup
            else:
                K[i, i] = k_act

        # SOLVER: eigh devuelve vectores YA normalizados respecto a la masa
        # Condición: phi.T * M * phi = I
        w2, modos = eigh(K, M)
        w = np.sqrt(np.abs(w2))
        
        # Ordenamos los modos (a veces salen desordenados) de menor a mayor frecuencia (mayor periodo)
        # Aunque eigh suele darlos ordenados, aseguramos indices
        idx = w.argsort()
        w = w[idx]
        modos = modos[:, idx]

        # --- PESTAÑAS DE RESULTADOS ---
        tab1, tab2, tab3 = st.tabs(["📊 Periodos", "🔢 Vectores (φ)", "📈 Gráficos"])

        # Pestaña 1: Periodos y Frecuencias
        with tab1:
            st.subheader("Frecuencias y Periodos")
            res_list = []
            for i in range(n):
                T = 2 * np.pi / w[i] if w[i] > 0 else 0
                res_list.append({
                    "Modo": i+1,
                    "Periodo T (s)": f"{T:.4f}",
                    "ω (rad/s)": f"{w[i]:.4f}",
                    "f (Hz)": f"{w[i]/(2*np.pi):.4f}"
                })
            st.table(pd.DataFrame(res_list))

        # Pestaña 2: Vectores Propios (Matrices)
        with tab2:
            st.subheader("Matriz de Modos (Normalizados)")
            st.info("Nota: Estos modos cumplen con $φ^T \\cdot M \\cdot φ = I$")
            
            # Crear DataFrame para mostrar la matriz phi bonita
            cols = [f"Modo {i+1}" for i in range(n)]
            rows = [f"Piso {i+1}" for i in range(n)]
            df_modos = pd.DataFrame(modos, index=rows, columns=cols)
            
            # Mostramos con degradado de color para ver magnitudes
            st.dataframe(df_modos.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

            st.divider()
            st.subheader("Matriz de Modos (Sin Normalizar / Escalados)")
            st.caption("Escalados para que la azotea sea igual a 1.0 (Solo visualización)")
            
            # Normalización visual (Azotea = 1)
            modos_visual = np.zeros_like(modos)
            for i in range(n):
                factor = modos[-1, i] # Valor del último piso
                if abs(factor) > 1e-6:
                    modos_visual[:, i] = modos[:, i] / factor
                else:
                    modos_visual[:, i] = modos[:, i]

            df_visual = pd.DataFrame(modos_visual, index=rows, columns=cols)
            st.dataframe(df_visual.style.format("{:.4f}"), use_container_width=True)

        # Pestaña 3: Gráficos de Modos
        with tab3:
            st.subheader("Formas Modales")
            
            # Graficamos los primeros 3 modos (o menos si hay menos pisos)
            modos_a_graficar = min(3, n)
            
            fig, ax = plt.subplots(figsize=(4, 6))
            pisos_y = np.arange(n + 1) # 0, 1, 2, ... n
            
            colors = ['#e74c3c', '#3498db', '#2ecc71']
            
            for i in range(modos_a_graficar):
                # Agregamos el punto 0 al inicio (suelo)
                forma = np.concatenate(([0], modos_visual[:, i]))
                ax.plot(forma, pisos_y, marker='o', label=f'Modo {i+1}', color=colors[i], linewidth=2)
            
            ax.set_ylabel("Nivel / Piso")
            ax.set_xlabel("Desplazamiento Relativo")
            ax.set_title("Formas de Modo (Primeros 3)")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            ax.axvline(0, color='black', linewidth=1) # Línea central
            
            st.pyplot(fig)
