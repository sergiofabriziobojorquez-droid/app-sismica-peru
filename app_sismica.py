import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Sismica E.030", layout="centered")
st.title("🇵🇪 Análisis Modal Espectral E.030")
st.caption("Cálculo de Modos y Aceleraciones Espectrales")

# ==========================================
# 1. PARÁMETROS SÍSMICOS (NORMA E.030)
# ==========================================
with st.expander("🌍 1. Parámetros Sísmicos (Z, U, S, R)", expanded=True):
    col1, col2 = st.columns(2)
    
    # --- ZONA (Z) ---
    zona_idx = col1.selectbox("Zona Sísmica", ["Zona 4", "Zona 3", "Zona 2", "Zona 1"])
    mapa_z = {"Zona 4": 0.45, "Zona 3": 0.35, "Zona 2": 0.25, "Zona 1": 0.10}
    Z = mapa_z[zona_idx]
    
    # --- USO (U) ---
    uso_idx = col2.selectbox("Categoría de Edificación", ["A (Esencial)", "B (Importante)", "C (Común)"])
    mapa_u = {"A (Esencial)": 1.5, "B (Importante)": 1.3, "C (Común)": 1.0}
    U = mapa_u[uso_idx]
    
    # --- SUELO (S, Tp, Tl) ---
    suelo_idx = col1.selectbox("Perfil de Suelo", ["S0 (Roca Dura)", "S1 (Roca/Rigido)", "S2 (Intermedio)", "S3 (Blando)"])
    
    # Lógica Tabla N°3 (Factor S)
    # Matriz S[zona][suelo] (0=S0, 1=S1, 2=S2, 3=S3)
    tabla_s = {
        0.45: [0.80, 1.00, 1.05, 1.10], # Z4
        0.35: [0.80, 1.00, 1.15, 1.20], # Z3
        0.25: [0.80, 1.00, 1.20, 1.40], # Z2
        0.10: [0.80, 1.00, 1.60, 2.00]  # Z1
    }
    s_index = ["S0", "S1", "S2", "S3"].index(suelo_idx.split()[0])
    S = tabla_s[Z][s_index]
    
    # Lógica Tabla N°4 (Tp y Tl)
    tabla_tp_tl = {
        "S0": (0.3, 3.0),
        "S1": (0.4, 2.5),
        "S2": (0.6, 2.0),
        "S3": (1.0, 1.6)
    }
    Tp, Tl = tabla_tp_tl[suelo_idx.split()[0]]

    # --- REDUCCIÓN (R) ---
    R = col2.number_input("Coef. Reducción (R)", value=8.0, step=0.1, help="Ej: Pórticos=8, Dual=7, Muros=6")

    # Mostrar resumen de parámetros calculados
    st.info(f"📊 **Parámetros:** Z={Z} | U={U} | S={S} | Tp={Tp}s | Tl={Tl}s | R={R}")

# ==========================================
# 2. DATOS DEL EDIFICIO (MASA Y RIGIDEZ)
# ==========================================
with st.expander("🏢 2. Datos del Edificio (Masa y Rigidez)", expanded=False):
    n_pisos = st.number_input("Número de pisos:", min_value=1, max_value=20, value=3)
    st.write("Ingrese Masa (Ton-s²/m) y Rigidez (Ton/m):")
    
    datos_masa = []
    datos_rigidez = []
    
    for i in range(n_pisos):
        c1, c2 = st.columns(2)
        m = c1.number_input(f"Masa P{i+1}", value=10.0, key=f"m{i}")
        k = c2.number_input(f"Rigidez P{i+1}", value=1000.0, key=f"k{i}")
        datos_masa.append(m)
        datos_rigidez.append(k)

# ==========================================
# 3. CÁLCULO Y RESULTADOS
# ==========================================
def calcular_C(T, Tp, Tl):
    if T <= Tp: return 2.5
    elif T <= Tl: return 2.5 * (Tp / T)
    else: return 2.5 * (Tp * Tl / T**2)

st.write("---")
if st.button("🚀 CALCULAR ESPECTRO Y MODOS", type="primary", use_container_width=True):
    
    if sum(datos_masa) == 0 or sum(datos_rigidez) == 0:
        st.error("⚠️ Ingrese valores de masa y rigidez.")
    else:
        # --- A. ANÁLISIS MODAL (Matemática) ---
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

        w2, modos = eigh(K, M)
        w = np.sqrt(np.abs(w2))
        
        # Ordenar (Mayor periodo a menor)
        idx = w.argsort() # eigh ordena por menor w (mayor T)
        w = w[idx]
        modos = modos[:, idx]

        # --- B. ANÁLISIS ESPECTRAL (E.030) ---
        resultados = []
        g = 9.81 # m/s2
        
        for i in range(n):
            omega = w[i]
            if omega > 0:
                T = 2 * np.pi / omega
                # Calcular C dinámicamente según el periodo
                C = calcular_C(T, Tp, Tl)
                # Aceleración Espectral (m/s2)
                Sa = (Z * U * C * S * g) / R
            else:
                T, C, Sa = 0, 0, 0
            
            resultados.append({
                "Modo": i + 1,
                "Periodo (s)": T,
                "C": C,
                "Sa (m/s²)": Sa,
                "Sa (g)": Sa / g
            })
            
        df_res = pd.DataFrame(resultados)

        # --- C. MOSTRAR RESULTADOS ---
        tab1, tab2, tab3 = st.tabs(["📉 Espectro", "📊 Tabla Sa", "📐 Formas"])

        with tab1:
            st.subheader("Espectro de Diseño E.030")
            # Graficar la curva del espectro continua
            t_plot = np.linspace(0.01, 4.0, 100)
            sa_plot = [(Z * U * calcular_C(t, Tp, Tl) * S * g)/R for t in t_plot]
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(t_plot, sa_plot, label="Espectro de Diseño (Sa)", color="navy")
            
            # Pintar los puntos de los modos del edificio
            periodos_modos = df_res["Periodo (s)"]
            sa_modos = df_res["Sa (m/s²)"]
            ax.scatter(periodos_modos, sa_modos, color="red", zorder=5, label="Modos del Edificio")
            
            for i, txt in enumerate(periodos_modos):
                ax.annotate(f"M{i+1}", (periodos_modos[i], sa_modos[i]), textcoords="offset points", xytext=(0,10), ha='center')

            ax.set_xlabel("Periodo T (s)")
            ax.set_ylabel("Aceleración Espectral Sa (m/s²)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            st.pyplot(fig)

        with tab2:
            st.subheader("Aceleraciones por Modo")
            st.dataframe(df_res.style.format({
                "Periodo (s)": "{:.4f}",
                "C": "{:.2f}",
                "Sa (m/s²)": "{:.4f}",
                "Sa (g)": "{:.4f}"
            }), use_container_width=True)
            
            st.info("Nota: Sa = (Z·U·C·S·g) / R")

        with tab3:
            st.subheader("Formas Modales (Escaladas)")
            # Modos visuales (Azotea=1)
            modos_vis = np.zeros_like(modos)
            for i in range(n):
                f = modos[-1, i]
                modos_vis[:, i] = modos[:, i] / f if abs(f) > 1e-9 else modos[:, i]
            
            cols = [f"Modo {i+1}" for i in range(n)]
            rows = [f"Piso {i+1}" for i in range(n)]
            st.dataframe(pd.DataFrame(modos_vis, index=rows, columns=cols).style.format("{:.3f}"), use_container_width=True)

