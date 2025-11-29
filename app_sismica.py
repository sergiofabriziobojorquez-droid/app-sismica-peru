import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(page_title="Ingeniería Sísmica PE", layout="centered")

st.title("🇵🇪 Análisis Sísmico E.030")
st.caption("Análisis Modal + Espectro de Diseño + Masas Participativas")

# ==========================================
# 1. BLOQUE DE ENTRADA DE DATOS
# ==========================================
with st.expander("📝 1. Configuración del Edificio y Sismo", expanded=True):
    # A. DATOS GEOMÉTRICOS
    st.markdown("### A. Edificio")
    n_pisos = st.number_input("Número de pisos:", min_value=1, max_value=20, value=3)
    
    # B. PARÁMETROS E.030
    st.markdown("### B. Parámetros E.030")
    c1, c2 = st.columns(2)
    
    # Zona Z
    zona_idx = c1.selectbox("Zona (Z)", ["Zona 4", "Zona 3", "Zona 2", "Zona 1"])
    mapa_z = {"Zona 4": 0.45, "Zona 3": 0.35, "Zona 2": 0.25, "Zona 1": 0.10}
    Z = mapa_z[zona_idx]
    
    # Uso U
    uso_idx = c2.selectbox("Uso (U)", ["A (Esencial)", "B (Importante)", "C (Común)"])
    mapa_u = {"A (Esencial)": 1.5, "B (Importante)": 1.3, "C (Común)": 1.0}
    U = mapa_u[uso_idx]
    
    # Suelo S (Con lógica automática de Tp y Tl)
    suelo_lbl = c1.selectbox("Suelo (S)", ["S0 (Roca Dura)", "S1 (Roca/Rigido)", "S2 (Intermedio)", "S3 (Blando)"])
    suelo_tipo = suelo_lbl.split()[0] # S0, S1, etc.
    
    # Lógica Tabla S (Z vs Suelo)
    tabla_s = {
        0.45: [0.80, 1.00, 1.05, 1.10], # Z4
        0.35: [0.80, 1.00, 1.15, 1.20], # Z3
        0.25: [0.80, 1.00, 1.20, 1.40], # Z2
        0.10: [0.80, 1.00, 1.60, 2.00]  # Z1
    }
    idx_s = ["S0", "S1", "S2", "S3"].index(suelo_tipo)
    S = tabla_s[Z][idx_s]
    
    # Tabla Tp y Tl
    tabla_tp_tl = {
        "S0": (0.3, 3.0), "S1": (0.4, 2.5), 
        "S2": (0.6, 2.0), "S3": (1.0, 1.6)
    }
    Tp, Tl = tabla_tp_tl[suelo_tipo]
    
    # Coeficiente R
    R = c2.number_input("Reducción (R)", value=8.0, step=0.1)

    st.info(f"Parámetros: Z={Z} | U={U} | S={S} | Tp={Tp} | Tl={Tl}")

    # C. MASA Y RIGIDEZ
    st.markdown("### C. Masa y Rigidez por Piso")
    datos_masa = []
    datos_rigidez = []
    
    # Grid de entrada optimizado
    for i in range(n_pisos):
        col_m, col_k = st.columns(2)
        m = col_m.number_input(f"Masa P{i+1}", value=10.0, key=f"m{i}")
        k = col_k.number_input(f"Rigidez P{i+1}", value=1000.0, key=f"k{i}")
        datos_masa.append(m)
        datos_rigidez.append(k)

# ==========================================
# 2. MOTOR DE CÁLCULO
# ==========================================
def calcular_C(T, Tp, Tl):
    if T <= Tp: return 2.5
    elif T <= Tl: return 2.5 * (Tp / T)
    else: return 2.5 * (Tp * Tl / T**2)

st.divider()

if st.button("🚀 EJECUTAR ANÁLISIS COMPLETO", type="primary", use_container_width=True):
    if sum(datos_masa) == 0 or sum(datos_rigidez) == 0:
        st.error("⚠️ La masa y rigidez deben ser mayores a 0.")
    else:
        # --- A. CÁLCULO MATRICIAL ---
        n = len(datos_masa)
        M = np.diag(datos_masa)
        K = np.zeros((n, n))

        # Matriz K (Shear Building)
        for i in range(n):
            k_act = datos_rigidez[i]
            if i < n - 1:
                k_sup = datos_rigidez[i+1]
                K[i, i] = k_act + k_sup
                K[i, i+1] = -k_sup
                K[i+1, i] = -k_sup
            else:
                K[i, i] = k_act

        # Valores propios (phi normalizado por masa)
        w2, modos_masa = eigh(K, M)
        w = np.sqrt(np.abs(w2))
        
        # Ordenar resultados (Periodo mayor a menor)
        idx = w.argsort()
        w = w[idx]
        modos_masa = modos_masa[:, idx] # phi normalizado

        # Modos Escalados (Azotea = 1) para visualización
        modos_esc = np.zeros_like(modos_masa)
        for i in range(n):
            val_top = modos_masa[-1, i]
            modos_esc[:, i] = modos_masa[:, i] / val_top if abs(val_top) > 1e-9 else modos_masa[:, i]

        # --- B. CÁLCULO DE FACTORES DE PARTICIPACIÓN ---
        # Vector de influencia r (para sismo horizontal es un vector de unos)
        r = np.ones(n)
        masa_total = np.sum(datos_masa)
        
        lista_participacion = []
        suma_masa_efectiva = 0
        
        for i in range(n):
            # Obtener el modo normalizado i
            phi_i = modos_masa[:, i]
            
            # Cálculo del Factor de Participación Gamma
            # Gamma = phi^T * M * r
            # Como phi está normalizado respecto a masa, el denominador es 1.
            gamma = np.dot(phi_i, np.dot(M, r))
            
            # Masa Efectiva = Gamma^2 (Para modos normalizados por masa)
            masa_efectiva = gamma**2
            porcentaje = (masa_efectiva / masa_total) * 100
            
            suma_masa_efectiva += porcentaje
            
            lista_participacion.append({
                "Modo": i + 1,
                "Gamma (Γ)": gamma,
                "Masa Efec.": masa_efectiva,
                "% Masa": porcentaje,
                "% Acumulado": suma_masa_efectiva
            })
            
        df_part = pd.DataFrame(lista_participacion)

        # --- C. CÁLCULO ESPECTRAL ---
        data_espectro = []
        g = 9.81
        
        for i in range(n):
            T = 2 * np.pi / w[i] if w[i] > 0 else 0
            C = calcular_C(T, Tp, Tl)
            Sa = (Z * U * C * S * g) / R
            
            data_espectro.append({
                "Modo": i+1,
                "T (s)": T,
                "C": C,
                "Sa (m/s²)": Sa,
                "Sa (g)": Sa/g
            })
            
        df_esp = pd.DataFrame(data_espectro)

        # ==========================================
        # 3. RESULTADOS EN PESTAÑAS
        # ==========================================
        tab_din, tab_mat, tab_e030 = st.tabs(["📊 Dinámica", "🔢 Matrices y Factores", "🇵🇪 Espectro E.030"])

        # --- PESTAÑA 1: DINÁMICA ---
        with tab_din:
            st.subheader("1. Frecuencias y Periodos")
            res_list = []
            for i in range(n):
                T_val = 2 * np.pi / w[i] if w[i] > 0 else 0
                res_list.append({
                    "Modo": i+1, 
                    "Periodo T (s)": f"{T_val:.4f}", 
                    "ω (rad/s)": f"{w[i]:.4f}"
                })
            st.table(pd.DataFrame(res_list))

            st.subheader("2. Gráfico de Modos")
            fig, ax = plt.subplots(figsize=(4, 6))
            pisos_y = np.arange(n + 1)
            colores = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f']
            
            for i in range(min(3, n)):
                forma = np.concatenate(([0], modos_esc[:, i]))
                ax.plot(forma, pisos_y, marker='o', label=f'Modo {i+1}', color=colores[i%4], linewidth=2)
            
            ax.set_ylabel("Nivel")
            ax.set_xlabel("Desplazamiento Relativo")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            ax.axvline(0, color='black', linewidth=1)
            st.pyplot(fig)

        # --- PESTAÑA 2: MATRICES Y FACTORES (ACTUALIZADO) ---
        with tab_mat:
            cols = [f"Modo {i+1}" for i in range(n)]
            rows = [f"Piso {i+1}" for i in range(n)]

            st.subheader("A. Modos Escalados (Visual)")
            st.caption("Normalizados tal que la azotea = 1.00")
            st.dataframe(pd.DataFrame(modos_esc, index=rows, columns=cols).style.format("{:.4f}"), use_container_width=True)

            st.divider()
            
            st.subheader("B. Factores de Participación (Nuevo)")
            st.caption("Verificar que % Acumulado supere el 90%")
            st.dataframe(df_part.style.format({
                "Gamma (Γ)": "{:.4f}",
                "Masa Efec.": "{:.2f}",
                "% Masa": "{:.2f}%",
                "% Acumulado": "{:.2f}%"
            }).background_gradient(cmap="Greens", subset=["% Masa"]), use_container_width=True)
            
            st.divider()
            
            st.subheader("C. Modos Masa-Normalizados")
            st.caption("Valores matemáticos ($φ^T M φ = I$)")
            st.dataframe(pd.DataFrame(modos_masa, index=rows, columns=cols).style.background_gradient(cmap="Blues"), use_container_width=True)

        # --- PESTAÑA 3: ESPECTRO E.030 ---
        with tab_e030:
            st.subheader("Aceleraciones Espectrales (Sa)")
            st.markdown(f"**Parámetros:** Z={Z}, U={U}, S={S}, R={R}")
            
            st.dataframe(df_esp.style.format({
                "T (s)": "{:.4f}", 
                "C": "{:.2f}", 
                "Sa (m/s²)": "{:.4f}", 
                "Sa (g)": "{:.4f}"
            }), use_container_width=True)
            
            st.divider()
            st.subheader("Gráfico del Espectro de Diseño")
            
            t_plot = np.linspace(0.01, 4.0, 100)
            sa_plot = [(Z * U * calcular_C(t, Tp, Tl) * S * g)/R for t in t_plot]
            
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(t_plot, sa_plot, label="Espectro E.030", color="navy", linewidth=2)
            
            t_modos = df_esp["T (s)"]
            sa_modos = df_esp["Sa (m/s²)"]
            ax2.scatter(t_modos, sa_modos, color="red", zorder=5, s=50, label="Modos Estructurales")
            
            for i, txt in enumerate(t_modos):
                ax2.annotate(f"M{i+1}", (t_modos[i], sa_modos[i]), xytext=(0,10), textcoords="offset points", ha='center', fontsize=8)
            
            ax2.set_xlabel("Periodo T (s)")
            ax2.set_ylabel("Pseudo-Aceleración Sa (m/s²)")
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend()
            
            st.pyplot(fig2)
            
            cols = [f"Modo {i+1}" for i in range(n)]
            rows = [f"Piso {i+1}" for i in range(n)]
            st.dataframe(pd.DataFrame(modos_vis, index=rows, columns=cols).style.format("{:.3f}"), use_container_width=True)


