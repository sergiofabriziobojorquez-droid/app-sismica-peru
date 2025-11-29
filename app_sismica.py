import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(page_title="Ingeniería Sísmica PE", layout="centered")

st.title("🇵🇪 Análisis Sísmico E.030")
st.caption("Análisis Modal + Espectro + Desplazamientos + Fuerzas + Derivas")

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

    # C. MASA, RIGIDEZ Y ALTURA
    st.markdown("### C. Datos por Piso (Masa, Rigidez, Altura)")
    datos_masa = []
    datos_rigidez = []
    datos_altura = []
    
    h1, h2, h3 = st.columns(3)
    h1.write("**Masa (Ton-s²/m)**")
    h2.write("**Rigidez (Ton/m)**")
    h3.write("**Altura h (m)**")

    for i in range(n_pisos):
        col_m, col_k, col_h = st.columns(3)
        m = col_m.number_input(f"m{i+1}", value=10.0, label_visibility="collapsed", key=f"m{i}")
        k = col_k.number_input(f"k{i+1}", value=1000.0, label_visibility="collapsed", key=f"k{i}")
        h = col_h.number_input(f"h{i+1}", value=3.0, label_visibility="collapsed", key=f"h{i}")
        
        datos_masa.append(m)
        datos_rigidez.append(k)
        datos_altura.append(h)

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
        
        idx = w.argsort()
        w = w[idx]
        modos_raw = modos_raw[:, idx] 

        # VECTORES NORMALIZADOS (Visuales, Azotea=1)
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

        # --- C. ESPECTRO Y RESPUESTAS ---
        data_espectro = []
        g = 9.81
        
        desplazamientos_modales = np.zeros((n, n))
        fuerzas_modales = np.zeros((n, n))
        
        for i in range(n):
            # Espectro
            T = 2 * np.pi / w[i] if w[i] > 0 else 0
            C = calcular_C(T, Tp, Tl)
            Sa = (Z * U * C * S * g) / R
            
            # Sd = Sa / w^2
            Sd = Sa / (w[i]**2) if w[i]>0 else 0
            
            data_espectro.append({
                "Modo": i+1, "T (s)": T, "C": C, 
                "Sa (m/s²)": Sa, "Sa (g)": Sa/g,
                "Sd (m)": Sd
            })
            
            # Desplazamientos Modales (u_i)
            u_i = Sd * gammas[i] * modos_visual[:, i]
            desplazamientos_modales[:, i] = u_i
            
            # Fuerzas Modales (f_i)
            vector_M_Xi = np.dot(M, modos_visual[:, i])
            f_i = Sa * gammas[i] * vector_M_Xi
            fuerzas_modales[:, i] = f_i
            
        df_esp = pd.DataFrame(data_espectro)

        # --- D. COMBINACIONES Y DERIVAS ---
        u_sva = np.sum(np.abs(desplazamientos_modales), axis=1)
        u_rcsc = np.sqrt(np.sum(desplazamientos_modales**2, axis=1))
        u_final = 0.25 * u_sva + 0.75 * u_rcsc
        
        desp_relativo = np.zeros(n)
        derivas = np.zeros(n)
        
        for k in range(n):
            h_piso = datos_altura[k]
            if k == 0:
                delta = u_final[0]
            else:
                delta = u_final[k] - u_final[k-1]
            
            desp_relativo[k] = delta
            derivas[k] = delta / h_piso if h_piso > 0 else 0

        df_desp = pd.DataFrame({
            "Nivel": [f"Piso {k+1}" for k in range(n)],
            "u Absoluto [m]": u_final,
            "Δ Relativo [m]": desp_relativo,
            "Altura h [m]": datos_altura,
            "Deriva (Δ/h)": derivas
        })

        # --- E. COMBINACIONES DE FUERZAS ---
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
        # 3. PESTAÑAS DE RESULTADOS
        # ==========================================
        tabs = st.tabs(["📊 Dinámica", "🔢 Matrices", "🇵🇪 Espectro", "📉 Derivas", "🏗️ Fuerzas"])

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
            # Aumenté decimales y notación científica para Sd
            st.dataframe(df_esp.style.format({
                "T (s)": "{:.4f}", "C": "{:.2f}", 
                "Sa (m/s²)": "{:.4f}", "Sa (g)": "{:.4f}",
                "Sd (m)": "{:.6f}"
            }), use_container_width=True)
            
            t_plot = np.linspace(0.01, 4.0, 100)
            sa_plot = [(Z * U * calcular_C(t, Tp, Tl) * S * g)/R for t in t_plot]
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(t_plot, sa_plot, color="navy", label="Espectro E.030")
            ax2.scatter(df_esp["T (s)"], df_esp["Sa (m/s²)"], color="red", label="Modos")
            ax2.grid(True, linestyle="--")
            ax2.legend()
            st.pyplot(fig2)

        # 4. DERIVAS (CON NOTACIÓN CIENTÍFICA)
        with tabs[3]:
            st.subheader("A. Desplazamientos Modales (u_i)")
            st.latex(r"u_i = S_{di} \cdot r_i \cdot X_i")
            
            # --- AQUÍ ESTÁ EL CAMBIO: Notación Científica (e) ---
            df_u_modos = pd.DataFrame(desplazamientos_modales, index=rows, columns=cols)
            st.dataframe(df_u_modos.style.format("{:.4e}"), use_container_width=True)
            
            st.divider()
            st.subheader("B. Control de Derivas (25/75)")
            st.markdown(r"$\Delta_i = u_i - u_{i-1}$ (Desp. Relativo)")
            st.markdown(r"$\text{Deriva}_i = \Delta_i / h_i$")
            
            # Formato de 6 decimales para ver derivas pequeñas
            st.dataframe(df_desp.style.format({
                "u Absoluto [m]": "{:.6f}",
                "Δ Relativo [m]": "{:.6f}",
                "Altura h [m]": "{:.2f}",
                "Deriva (Δ/h)": "{:.6f}"
            }).background_gradient(cmap="Oranges", subset=["Deriva (Δ/h)"]), use_container_width=True)

        # 5. FUERZAS
        with tabs[4]:
            st.subheader("A. Fuerzas Modales (f_i)")
            st.latex(r"f_i = S_{ai} \cdot r_i \cdot M \cdot X_i")
            st.dataframe(pd.DataFrame(fuerzas_modales, index=rows, columns=cols).style.format("{:.4f}"), use_container_width=True)
            
            st.divider()
            st.subheader("B. Fuerzas de Diseño (25/75)")
            st.dataframe(df_fuerzas_comb.style.format({
                "F (SVA) [Tn]": "{:.4f}",
                "F (RCSC) [Tn]": "{:.4f}",
                "F (25/75) [Tn]": "{:.4f}"
            }).background_gradient(cmap="Reds", subset=["F (25/75) [Tn]"]), use_container_width=True)







