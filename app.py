import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

st.set_page_config(layout="wide")
st.title("Tafel Analysis App")

uploaded_file = st.file_uploader(
    "Upload polarization file (.xlsx/.csv)", type=['xlsx', 'csv'], accept_multiple_files=False
)

def clean_data(E, I):
    mask = (I != 0) & np.isfinite(E) & np.isfinite(I)
    return E[mask], I[mask]

def fit_region(E, logI):
    slope, intercept, r2, _, _ = linregress(E, logI)
    return slope, intercept, r2**2

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("#### Data preview")
    st.dataframe(df.head())

    cols = df.columns.tolist()
    guess_E = next((c for c in cols if "potential" in c.lower()), cols[0])
    guess_I = next((c for c in cols if "current" in c.lower()), cols[1])
    potential_col = st.selectbox("Select the Potential column:", cols, index=cols.index(guess_E))
    current_col = st.selectbox("Select the Current column:", cols, index=cols.index(guess_I))

    E_raw = df[potential_col].values.astype(float)
    I_raw = df[current_col].values.astype(float)
    E, I = clean_data(E_raw, np.abs(I_raw))

    if len(E) < 10:
        st.warning("Too few valid data points for analysis.")
        st.stop()

    idx_sort = np.argsort(E)
    E = E[idx_sort]
    I = I[idx_sort]
    logI = np.log10(I)

    # ----------- Find zero crossing for Ecorr starting guess -----------
    idx_guess_Ecorr = np.argmin(np.abs(I))
    Ecorr_guess = float(E[idx_guess_Ecorr])
    st.write(f"**Initial Ecorr guess:** {Ecorr_guess:.3f} V")

    # Slider pick
    cath_idx = np.where(E < Ecorr_guess)[0]
    anod_idx = np.where(E > Ecorr_guess)[0]

    if len(cath_idx) < 3 or len(anod_idx) < 3:
        st.error("Not enough data points in one or both regions around Ecorr!")
        st.stop()

    def subsample(ar):
        step = max(1, len(ar)//40)
        return ar[::step] if len(ar) > 40 else ar

    cath_options = [float(f"{E[i]:.5f}") for i in subsample(cath_idx)]
    anod_options = [float(f"{E[i]:.5f}") for i in subsample(anod_idx)]

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Cathodic region:**")
        cath_range = st.select_slider(
            "Pick V range (cathodic, left/blue branch)", cath_options,
            value=(cath_options[min(2, len(cath_options)-2)], cath_options[-2])
        )
    with col2:
        st.write("**Anodic region:**")
        anod_range = st.select_slider(
            "Pick V range (anodic, right/orange branch)", anod_options,
            value=(anod_options[1], anod_options[-2])
        )

    mask_cath = (E >= cath_range[0]) & (E <= cath_range[1])
    mask_anod = (E >= anod_range[0]) & (E <= anod_range[1])

    if mask_cath.sum() < 3:
        st.error("Select a wider cathodic region to allow fitting.")
        st.stop()
    if mask_anod.sum() < 3:
        st.error("Select a wider anodic region to allow fitting.")
        st.stop()

    E_cath, logI_cath, I_cath = E[mask_cath], logI[mask_cath], I[mask_cath]
    E_anod, logI_anod, I_anod = E[mask_anod], logI[mask_anod], I[mask_anod]

    # Fit each region (linear fit in logI)
    slope_c, int_c, fitr2_c = fit_region(E_cath, logI_cath)
    slope_a, int_a, fitr2_a = fit_region(E_anod, logI_anod)

    # Tafel slopes
    beta_c = -2.303/slope_c
    beta_a = 2.303/slope_a

    # Ecorr/Icorr by intersection
    if slope_c == slope_a:
        st.error("Anodic and cathodic Tafel slopes are identical; cannot find intersection!")
        st.stop()
    Ecorr = (int_a - int_c) / (slope_c - slope_a)
    logIcorr = slope_c * Ecorr + int_c
    Icorr = 10 ** logIcorr
    corrosion_rate = 0.00327 * Icorr  # mm/y, placeholder

    # -------- Raw plot with regions highlighted -------
    fig0, ax0 = plt.subplots(figsize=(8,4))
    ax0.plot(E, I, '.', color='lightgray', ms=3, label='All Data')
    ax0.plot(E_cath, I_cath, 'x', color='blue', label='Selected cathodic region')
    ax0.plot(E_anod, I_anod, 'o', color='orange', label='Selected anodic region')
    ax0.axvline(Ecorr, color='purple', ls='--', lw=1.7, label=f'Ecorr (fit) = {Ecorr:.3f} V')
    ax0.set_xlabel('Potential (V)')
    ax0.set_ylabel('Current (A)')
    ax0.legend()
    ax0.grid(True)
    st.pyplot(fig0)
    st.caption("Selected linear Tafel regions highlighted.")

    # ------- Tafel plot -----------
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(E, logI, '.', color="lightgray", markersize=3, label="All log|I| vs E")
    ax.plot(E_cath, logI_cath, 'x', color='blue', label="Cathodic region")
    ax.plot(E_anod, logI_anod, 'o', color='orange', label="Anodic region")
    ax.plot(E_cath, slope_c*E_cath + int_c, 'b-', lw=2, label=f'Cathodic Fit (R²={fitr2_c:.2f})')
    ax.plot(E_anod, slope_a*E_anod + int_a, 'r-', lw=2, label=f'Anodic Fit (R²={fitr2_a:.2f})')
    ax.axvline(Ecorr, color='purple', linestyle='--', lw=1.7, label=f'Ecorr (fit) = {Ecorr:.3f} V')
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("log10(Current / A)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    st.caption("Your selected linear regions and corresponding Tafel fits. Ecorr/Icorr from Tafel intersection.")

    # -------- Table of results --------
    st.markdown("### **Tafel Fit Parameters (from your selected linear regions):**")
    st.write(f"**Ecorr (V, intersection):** `{Ecorr:.5f}`")
    st.write(f"**Icorr (A, intersection):** `{Icorr:.3e}`")
    st.write(f"**Beta_a (V/dec):** `{beta_a:.3e}`")
    st.write(f"**Beta_c (V/dec):** `{beta_c:.3e}`")
    st.write(f"**Corrosion Rate (mm/y):** `{corrosion_rate:.3e}`")
    st.write(f"**R² anodic:** `{fitr2_a:.3f}`")
    st.write(f"**R² cathodic:** `{fitr2_c:.3f}`")
