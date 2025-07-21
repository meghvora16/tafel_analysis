import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sci_notation(val, precision=3):
    try:
        return f"{float(val):.{precision}e}"
    except (TypeError, ValueError):
        return val

class TafelAnalyzer:
    def __init__(self, material_factor=0.327):
        self.material_factor = material_factor

    def mixed_control_fit(self, E, E_corr, beta_an, beta_cath, i_corr, i_L, gamma):
        anodic = i_corr * np.exp(2.303 * (E - E_corr) / beta_an)
        cathodic_base = (i_corr / i_L) * np.exp(2.303 * (E_corr - E) / beta_cath)
        cathodic = i_L * (cathodic_base * gamma / (1 + cathodic_base * gamma)) * (1 / gamma)
        return anodic - cathodic

    def fit_polarization_data(self, E, i):
        E = np.array(E, dtype=float)
        i = np.array(i, dtype=float)
        # Initial parameter guesses
        idx_Ecorr = np.argmin(np.abs(i))
        E_corr_initial = float(E[idx_Ecorr])
        i_corr_initial = np.abs(i[idx_Ecorr]) + 1e-12
        i_L_guess = np.clip(np.max(np.abs(i)), 1e-9, 1e2)
        gamma_guess = 3.0

        # Set realistic parameter bounds
        E_range = [E.min(), E.max()]
        i_range = [1e-12, np.max(np.abs(i))*100]
        beta_bounds = (1e-3, 0.5)
        gamma_bounds = (1.01, 100)
        bounds_lower = [
            E_range[0], beta_bounds[0], beta_bounds[0], i_range[0], i_range[0], gamma_bounds[0]
        ]
        bounds_upper = [
            E_range[1], beta_bounds[1], beta_bounds[1], i_range[1], i_range[1], gamma_bounds[1]
        ]
        p0 = [
            float(E_corr_initial),   # E_corr
            0.05,                   # beta_an
            0.07,                   # beta_cath
            float(i_corr_initial),
            float(i_L_guess),
            gamma_guess
        ]
        p0 = np.clip(p0, bounds_lower, bounds_upper)

        try:
            params, _ = curve_fit(
                self.mixed_control_fit, E, i,
                p0=p0, bounds=(bounds_lower, bounds_upper),
                method='trf', maxfev=100_000
            )
        except Exception as e:
            st.error(f"Fitting failed: {e}")
            params = [np.nan]*6

        i_fit = self.mixed_control_fit(E, *params)
        residuals = i - i_fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((i - np.mean(i)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        fit_result = {
            'E_corr (V)': float(params[0]),
            'beta_an (V/dec)': float(params[1]),
            'beta_cath (V/dec)': float(params[2]),
            'i_corr (A)': float(params[3]),
            'i_L (A)': float(params[4]),
            'gamma': float(params[5]),
            'Corrosion Rate (mm/y*)': float(params[3]) * self.material_factor,
            'R_squared': float(r_squared),
        }
        return fit_result, params

    def plot_full_fit(self, E, i, params, fit_result, fit_mask=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        E_full = np.linspace(np.min(E), np.max(E), 400)
        i_full_fit = self.mixed_control_fit(E_full, *params)

        if fit_mask is not None:
            ax.semilogy(E[~fit_mask], np.abs(i[~fit_mask]), 'o', color='gray', alpha=0.4, label='Outside fit region')
            ax.semilogy(E[fit_mask], np.abs(i[fit_mask]), 'o', color='C0', label='Fit region data')
        else:
            ax.semilogy(E, np.abs(i), 'o', color='C0', label='Experimental Data')

        ax.semilogy(E_full, np.abs(i_full_fit), 'r-', lw=2, label='Mixed Control Fit')
        ax.axvline(fit_result['E_corr (V)'], color='k', linestyle='--', lw=1.5, label=f'E_corr = {sci_notation(fit_result["E_corr (V)"],3)} V')
        ax.set_xlabel('Potential (V)')
        ax.set_ylabel('|Current| (A)')
        ax.set_title(f'Tafel Mixed Control Fit (R² = {fit_result["R_squared"]:.4f})')
        ax.legend()
        ax.grid(True, which='both', ls='--')
        st.pyplot(fig)

def process_excel(file):
    st.write("Reading data...")
    try:
        df = pd.read_excel(file)
        st.write("Preview of your uploaded data:")
        st.dataframe(df.head())
        col_options = list(df.columns)
        E_col = st.selectbox("Select the column for Potential (V)", col_options, index=0)
        i_col = st.selectbox("Select the column for Current (A)", col_options, index=min(2, len(col_options)-1))
        E = df[E_col].values
        i = df[i_col].values

        idx_Ecorr = np.argmin(np.abs(i))
        E_corr_guess = float(E[idx_Ecorr])
        default_window = 0.25
        Eleft = max(E.min(), E_corr_guess - default_window)
        Eright = min(E.max(), E_corr_guess + default_window)

        region = st.slider("Select Potential Range For Fitting (focus on Tafel/mixed region!)",
                           float(E.min()), float(E.max()), (Eleft, Eright), step=0.001)
        fit_mask = (E >= region[0]) & (E <= region[1])
        E_fit = E[fit_mask]
        i_fit = i[fit_mask]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(E, i, marker='o', label='Raw Data')
        ax.axvspan(region[0], region[1], color='yellow', alpha=0.3, label='Fit region')
        ax.set_title('Raw Data & Fit Region')
        ax.set_xlabel('Potential (V)')
        ax.set_ylabel('Current (A)')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        analyzer = TafelAnalyzer()
        fit_result, params = analyzer.fit_polarization_data(E_fit, i_fit)
        analyzer.plot_full_fit(E, i, params, fit_result, fit_mask=fit_mask)

        st.subheader("Mixed Control Fit Parameters (using fit region):")
        sci_results = {k: sci_notation(v, 3) for k, v in fit_result.items()}
        st.table(sci_results)

        if fit_result["R_squared"] < 0.9:
            st.warning("Low R²: Try adjusting fit region closer to the central Tafel region.")

    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")

def main():
    st.title("Tafel Mixed-Control Fit")
    st.markdown("""
    Upload polarization data, select the correct columns, and use the slider to select the region for fitting (ideally the Tafel/mixed region).
    All fit parameters are shown in scientific format.
    """)

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if uploaded_file is not None:
        process_excel(uploaded_file)

if __name__ == "__main__":
    main()
