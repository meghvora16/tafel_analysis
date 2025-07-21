import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

st.set_page_config(layout="wide")
st.title("Tafel Analysis App (Auto Linear Region and Intersection Method)")

uploaded_file = st.file_uploader(
    "Upload polarization file (.xlsx/.csv)", type=['xlsx', 'csv'], accept_multiple_files=False
)

def clean_data(E, I):
    mask = (I != 0) & np.isfinite(E) & np.isfinite(I)
    return E[mask], I[mask]

def find_best_linear_region(E, logI, side='cathodic', Ecorr=None, min_pts=8, max_pts=25):
    best_r2 = -np.inf
    best_start, best_end = 0, 0
    if side == 'cathodic':
        region_mask = (E < Ecorr)
    else:
        region_mask = (E > Ecorr)
    E_side = E[region_mask]
    logI_side = logI[region_mask]
    n = len(E_side)
    if n < min_pts:
        return np.array([], dtype=int), -np.inf
    idx_sort = np.argsort(E_side)
    E_side = E_side[idx_sort]
    logI_side = logI_side[idx_sort]
    for w in range(min_pts, min(max_pts+1, n+1)):
        for i in range(n - w + 1):
            x_win = E_side[i:i+w]
            y_win = logI_side[i:i+w]
            if len(np.unique(x_win)) < 2:
                continue
            slope, intercept, r_value, _, _ = linregress(x_win, y_win)
            r2 = r_value ** 2
            if r2 > best_r2:
                best_r2 = r2
                best_start = i
                best_end = i+w
    if best_end > best_start:
        chosen_Es = E_side[best_start:best_end]
        indices = np.where(np.isin(E, chosen_Es))[0]
        return indices, best_r2
    else:
        return np.array([], dtype=int), -np.inf

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

    # Initial guess for Ecorr (minimum |I|)
    idx_guess_Ecorr = np.argmin(np.abs(I))
    Ecorr_guess = E[idx_guess_Ecorr]
    st.write(f"**Auto-detected Ecorr (initial guess):** {Ecorr_guess:.3f} V")

    cath_indices, r2_c = find_best_linear_region(E, logI, 'cathodic', Ecorr_guess, min_pts=8, max_pts=25)
    anod_indices, r2_a = find_best_linear_region(E, logI, 'anodic', Ecorr_guess, min_pts=8, max_pts=25)

    if len(cath_indices) < 5 or len(anod_indices) < 5:
        st.error("Could not find a wide enough linear region automatically. Try cleaner data or adjust min/max window size.")
        st.stop()

    E_cath, logI_cath, I_cath = E[cath_indices], logI[cath_indices], I[cath_indices]
    E_anod, logI_anod, I_anod = E[anod_indices], logI[anod_indices], I[anod_indices]

    # Fit each region
    slope_c, int_c, fitr2_c = fit_region(E_cath, logI_cath)
    slope_a, int_a, fitr2_a = fit_region(E_anod, logI_anod)

    # Tafel slopes
    beta_c = -2.303/slope_c
    beta_a = 2.303/slope_a

    # Now intersect for Ecorr, Icorr (intersection of Tafel lines)
    if slope_c == slope_a:
        st.error("Anodic and cathodic Tafel slopes are identical; cannot find intersection!")
        st.stop()
    Ecorr = (int_a - int_c) / (slope_c - slope_a)
    logIcorr = slope_c * Ecorr + int_c  # or slope_a * Ecorr + int_a
    Icorr = 10 ** logIcorr
    corrosion_rate = 0.00327 * Icorr  # mm/y, placeholder

    st.write(f"**Intersection Ecorr:** `{Ecorr:.5f}` V")
    st.write(f"**Intersection Icorr:** `{Icorr:.3e}` A")

    # -------- Raw plot with regions highlighted -------
    fig0, ax0 = plt.subplots(figsize=(8,4))
    ax0.plot(E, I, '.', color='lightgray', ms=3, label='All Data')
    ax0.plot(E_cath, I_cath, 'x', color='blue', label='Best cathodic linear region')
    ax0.plot(E_anod, I_anod, 'o', color='orange', label='Best anodic linear region')
    ax0.axvline(Ecorr, color='purple', ls='--', lw=1.7, label=f'Ecorr (fit) = {Ecorr:.3f} V')
    ax0.set_xlabel('Potential (V)')
    ax0.set_ylabel('Current (A)')
    ax0.legend()
    ax0.grid(True)
    st.pyplot(fig0)
    st.caption("Best-fit Tafel region on raw LSV curve (detected automatically).")

    # ------- Tafel plot -----------
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(E, logI, '.', color="lightgray", markersize=3, label="All log|I| vs E")
    ax.plot(E_cath, logI_cath, 'x', color='blue', label="Auto cathodic region")
    ax.plot(E_anod, logI_anod, 'o', color='orange', label="Auto anodic region")
    ax.plot(E_cath, slope_c*E_cath + int_c, 'b-', lw=2, label=f'Cathodic Fit (R²={fitr2_c:.2f})')
    ax.plot(E_anod, slope_a*E_anod + int_a, 'r-', lw=2, label=f'Anodic Fit (R²={fitr2_a:.2f})')
    ax.axvline(Ecorr, color='purple', linestyle='--', lw=1.7, label=f'Ecorr (fit) = {Ecorr:.3f} V')
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("log10(Current / A)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    st.caption("Auto-detected most linear Tafel regions and linear fits. Ecorr/Icorr from Tafel intersection.")

    # -------- Table of results --------
    st.markdown("### **Tafel Fit Parameters (auto linear region):**")
    st.write(f"**Ecorr (V, intersection):** `{Ecorr:.5f}`")
    st.write(f"**Icorr (A, intersection):** `{Icorr:.3e}`")
    st.write(f"**Beta_a (V/dec):** `{beta_a:.3e}`")
    st.write(f"**Beta_c (V/dec):** `{beta_c:.3e}`")
    st.write(f"**Corrosion Rate (mm/y):** `{corrosion_rate:.3e}`")
    st.write(f"**R² anodic:** `{fitr2_a:.3f}`")
    st.write(f"**R² cathodic:** `{fitr2_c:.3f}`")
