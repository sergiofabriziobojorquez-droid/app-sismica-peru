import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(page_title="Ingeniería Sísmica PE", layout="centered")

st.title("🇵🇪 Análisis Sísmico E.030")
st.caption("Análisis Modal + Espectro + Desplazamientos + Fuerzas (Completo)")

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
    
    # Suelo S
    suelo_lbl = c1.selectbox("Suelo (S)", ["S0 (Roca Dura)", "S1 (Roca/Rigido)", "S2 (Intermedio)", "S3 (Blando)"])
    suelo_tipo = suelo_lbl.split()[0]
    
    tabla_s = {
        0.45: [0.80, 1.00, 1.05, 1.10], 0.35: [0.80, 1.00, 1.15, 1.20],
        0.25: [0.80, 1.00, 1.20, 1.40], 0.10: [0.80, 1.00, 1.60, 2.00]
    }
    idx_s = ["S0", "S1", "S2", "S3"].index(suelo_tipo)
    S = tabla_s[Z][idx_s]
    
    tabla_tp_tl = {"S0": (0.3, 3.0), "S1": (0.4, 2.5), "S2": (0.6, 2.0), "S3": (1.0, 1.6)}
    Tp, Tl = tabla_tp_tl[suelo_tipo]
    
    R = c2.number_input("Reducción (R)", value=8.0, step=0.1)
    st.info(f"Parámetros: Z={Z} | U={U} | S={S} | Tp={Tp} | Tl={Tl}")

    # C. MASA Y RIGIDEZ
    st.markdown("### C. Masa y Rigidez por Piso")
    datos_masa = []
    datos_rigidez = []
    
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

        for i in range(n):
            k_act = datos_rigidez[i]
            if i < n - 1:
                k_sup = datos_rigidez[i+1]
                K[i, i] = k_act + k_sup
                K[i, i+1] = -k_sup
                K[i+1, i] = -k_sup
            else:
                K[i, i] = k_act

        w2, modos_raw = eigh(K, M)
        w = np.sqrt(np.abs(w2))
        
        # Ordenar (Periodo mayor a menor)
        idx = w.argsort()
        w = w[idx]
        modos_raw = modos_raw[:, idx] 

        # Modos Escalados (Azotea = 1) -> TUS "VECTORES NORMALIZADOS"
        modos_visual = np.zeros_like(modos_raw)
        for i in range(n):
            val_top = modos_raw[-1, i]
            if abs(val_top) > 1e-9:
                modos_visual[:, i] = modos_raw[:, i] / val_top 
            else:
                modos_visual[:, i] = modos_raw[:, i]

        # --- B. FACTORES DE PARTICIPACIÓN ---
        vector_1 = np.ones(n) 
        masa_total = np.sum(datos_masa)
        
        lista_participacion = []
        suma_masa_efectiva = 0
        gammas = []
        
        for i in range(n):
            Xi = modos_visual[:, i]
            numerador = np.dot(Xi.T, np.dot(M, vector_1))
            denominador = np.dot(Xi.T, np.dot(M, Xi))
            gamma = numerador / denominador
            gammas.append(gamma)
            
            masa_efectiva = (numerador**2) / denominador
            porcentaje = (masa_efectiva / masa_total) * 100
            suma_masa_efectiva += porcentaje
            
            lista_participacion.append({
                "Modo": i + 1, "Gamma (Γ)": gamma, "Masa Efec.": masa_efectiva,
                "% Masa": porcentaje, "% Acumulado": suma_masa_efectiva
            })
            
        df_part = pd.DataFrame(lista_participacion)

        # --- C. CÁLCULO ESPECTRAL Y NUEVOS CÁLCULOS ---
        data_espectro = []
        g = 9.81
        
        desplazamientos_modales = np.zeros((n, n))
        fuerzas_modales = np.zeros((n, n))
        
        sa_values = []

        for i in range(n):
            # 1. Espectro
            T = 2 * np.pi / w[i] if w[i] > 0 else 0
            C = calcular_C(T, Tp, Tl)
            Sa = (Z * U * C * S * g) / R
            sa_values.append(Sa)
            
            data_espectro.append({
                "Modo": i+1, "T (s)": T, "C": C, 
                "Sa (m/s²)": Sa, "Sa (g)": Sa/g,
                "Sd (m)": Sa / (w[i]**2) if w[i]>0 else 0
            })
            
            # 2. Desplazamientos Modales (u_i = Sd * Gamma * Xi)
            Sd = Sa / (w[i]**2) if w[i] > 0 else 0
            u_i = Sd * gammas[i] * modos_visual[:, i]
            desplazamientos_modales[:, i] = u_i
            
            # 3. Fuerzas Laterales Modales (f_i = Sa * Gamma * M * Xi)
            vector_M_Xi = np.dot(M, modos_visual[:, i])
            f_i = Sa * gammas[i] * vector_M_Xi
            fuerzas_modales[:, i] = f_i
            
        df_esp = pd.DataFrame(data_espectro)

        # --- D. COMBINACIONES MODALES (DESPLAZAMIENTOS) ---
        u_sva = np.sum(np.abs(desplazamientos_modales), axis=1)
        u_rcsc = np.sqrt(np.sum(desplazamientos_modales**2, axis=1))
        u_final = 0.25 * u_sva + 0.75 * u_rcsc
        
        derivas = np.zeros(n)
        derivas[0] = u_final[0]
        for k in range(1, n):
            derivas[k] = u_final[k] - u_final[k-1]

        df_desp = pd.DataFrame({
            "Nivel": [f"Piso {k+1}" for k in range(n)],
            "u (SVA) [m]": u_sva,
            "u (RCSC) [m]": u_rcsc,
            "u (25/75) [m]": u_final,
            "Deriva Δ [m]": derivas
        })

        # --- E. COMBINACIONES MODALES (FUERZAS) ---
        f_sva = np.sum(np.abs(fuerzas_modales), axis=1)
        f_rcsc = np.sqrt(np.sum(fuerzas_modales**2, axis=1))
        f_final = 0.25 * f_sva + 0.75 * f_rcsc

        df_fuerzas_comb = pd.DataFrame({
            "Nivel": [f"Piso {k+1}" for k in range(n)],
            "F (SVA) [Tn]": f_sva,
            "F (RCSC) [Tn]": f_rcsc,
            "F (25/75) [Tn]": f_final
        })

        # ==========================================
        # 3. RESULTADOS EN PESTAÑAS (5 TABS)
        # ==========================================
        tabs = st.tabs(["📊 Dinámica", "🔢 Matrices", "🇵🇪 Espectro", "📉 Desplazamientos", "🏗️ Fuerzas Lat."])

        # 1. DINÁMICA
        with tabs[0]:
            st.subheader("1. Frecuencias y Periodos")
            st.table(pd.DataFrame({"Modo": range(1,n+1), "Periodo T (s)": [2*np.pi/val if val>0 else 0 for val in w]}))
            st.subheader("2. Gráfico de Modos")
            fig, ax = plt.subplots(figsize=(4, 6))
            pisos_y = np.arange(n + 1)
            colores = ['#e74c3c', '#3498db', '#2ecc71']
            for i in range(min(3, n)):
                forma = np.concatenate(([0], modos_visual[:, i]))
                ax.plot(forma, pisos_y, marker='o', label=f'Modo {i+1}', color=colores[i%3])
            ax.grid(True, linestyle='--')
            ax.legend()
            st.pyplot(fig)

        # 2. MATRICES
        with tabs[1]:
            cols = [f"Modo {i+1}" for i in range(n)]
            rows = [f"Piso {i+1}" for i in range(n)]
            st.subheader("A. Vectores Normalizados (Visuales)")
            st.dataframe(pd.DataFrame(modos_visual, index=rows, columns=cols).style.format("{:.4f}"), use_container_width=True)
            st.divider()
            st.subheader("B. Vectores sin normalizar (Matemáticos)")
            st.dataframe(pd.DataFrame(modos_raw, index=rows, columns=cols).style.background_gradient(cmap="Blues"), use_container_width=True)
            st.divider()
            st.subheader("C. Factores de Participación")
            st.latex(r"r_i = \frac{X_i^T \cdot M \cdot \{1\}}{X_i^T \cdot M \cdot X_i}")
            st.dataframe(df_part.style.format("{:.4f}", subset=["Gamma (Γ)", "Masa Efec."]), use_container_width=True)

        # 3. ESPECTRO
        with tabs[2]:
            st.subheader("Aceleraciones y Desplazamientos Espectrales")
            st.dataframe(df_esp.style.format("{:.4f}"), use_container_width=True)
            
            t_plot = np.linspace(0.01, 4.0, 100)
            sa_plot = [(Z * U * calcular_C(t, Tp, Tl) * S * g)/R for t in t_plot]
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(t_plot, sa_plot, color="navy", label="Espectro E.030")
            ax2.scatter(df_esp["T (s)"], df_esp["Sa (m/s²)"], color="red", label="Modos")
            ax2.grid(True, linestyle="--")
            ax2.legend()
            st.pyplot(fig2)

        # 4. DESPLAZAMIENTOS (CORREGIDO)
        with tabs[3]:
            st.subheader("A. Desplazamientos por Modo (u_i)")
            st.latex(r"u_i = S_{di} \cdot r_i \cdot X_i")
            df_u_modos = pd.DataFrame(desplazamientos_modales, index=rows, columns=cols)
            st.dataframe(df_u_modos.style.format("{:.4f}"), use_container_width=True)
            
            st.divider()
            st.subheader("B. Combinación Modal (Desplazamientos Totales)")
            st.info("Regla Peruana: 0.25 SVA + 0.75 RCSC")
            
            # --- AQUÍ ESTABA EL ERROR: Ahora especificamos las columnas exactas a formatear ---
            st.dataframe(df_desp.style.format({
                "u (SVA) [m]": "{:.4f}",
                "u (RCSC) [m]": "{:.4f}",
                "u (25/75) [m]": "{:.4f}",
                "Deriva Δ [m]": "{:.4f}"
            }).background_gradient(cmap="Oranges", subset=["u (25/75) [m]"]), use_container_width=True)
            
            st.write("**Gráfico de Perfil de Desplazamientos (25/75):**")
            fig3, ax3 = plt.subplots(figsize=(4, 6))
            perfil = np.concatenate(([0], u_final))
            ax3.plot(perfil, np.arange(n+1), marker='D', color='purple', linewidth=2)
            ax3.set_xlabel("Desplazamiento u (m)")
            ax3.set_ylabel("Nivel")
            ax3.grid(True)
            st.pyplot(fig3)

        # 5. FUERZAS LATERALES (CORREGIDO)
        with tabs[4]:
            st.subheader("A. Fuerzas Laterales por Modo (f_i)")
            st.latex(r"f_i = S_{ai} \cdot r_i \cdot M \cdot X_i")
            df_f_modos = pd.DataFrame(fuerzas_modales, index=rows, columns=cols)
            st.dataframe(df_f_modos.style.format("{:.4f}"), use_container_width=True)
            
            st.divider()
            st.subheader("B. Combinación Modal (Fuerzas de Diseño)")
            st.info("Fuerzas finales distribuidas en altura (Ton)")
            
            # --- CORREGIDO TAMBIÉN AQUÍ ---
            st.dataframe(df_fuerzas_comb.style.format({
                "F (SVA) [Tn]": "{:.4f}",
                "F (RCSC) [Tn]": "{:.4f}",
                "F (25/75) [Tn]": "{:.4f}"
            }).background_gradient(cmap="Reds", subset=["F (25/75) [Tn]"]), use_container_width=True)






