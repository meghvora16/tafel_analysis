import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class TafelAnalyzer:
    def __init__(self):
        self.area = None         # To be determined from data if possible/useful
        self.material_factor = 0.327  # Safe generic (mm/y for Fe)

    def mixed_control_fit(self, E, E_corr, beta_an, beta_cath, i_corr, i_L, gamma):
        # Use signed current convention
        anodic = i_corr * np.exp(2.303 * (E - E_corr) / beta_an)
        cathodic_base = (i_corr / i_L) * np.exp(2.303 * (E_corr - E) / beta_cath)
        cathodic = i_L * (cathodic_base * gamma / (1 + cathodic_base * gamma)) * (1 / gamma)
        return anodic - cathodic

    def fit_polarization_data(self, E, i):
        # Data autoconditioning
        E = np.array(E, dtype=float)
        i = np.array(i, dtype=float)
        # If current is positive-negative, leave as is; otherwise, warn for pure anodic/cathodic
        # Guess electrode area (not possible unless user supplies => set to 1.0 so i_density == i)
        self.area = 1.0

        i_density = i / self.area

        # Guess Ecorr as E where |i| is minimized, i_corr as abs(i) at that point
        idx_Ecorr = np.argmin(np.abs(i_density))
        E_corr_initial = float(np.clip(E[idx_Ecorr], np.min(E), np.max(E)))
        i_corr_initial = float(np.clip(np.abs(i_density[idx_Ecorr]) + 1e-10, 1e-12, 1e5))

        # i_L guess as half max(abs(i)), gamma in plausible range
        i_L_guess = np.clip(np.max(np.abs(i_density)), 1e-9, 1e3)
        gamma_guess = 3.0

        # Set simple, physically broad but realistic parameter bounds based on data
        E_range = [np.min(E), np.max(E)]
        i_range = [1e-12, np.max(np.abs(i_density))*100]
        beta_bounds = (0.001, 0.5)  # quite broad (V/dec)
        gamma_bounds = (1.01, 100)
        bounds_lower = [
            E_range[0], beta_bounds[0], beta_bounds[0], i_range[0], i_range[0], gamma_bounds[0]
        ]
        bounds_upper = [
            E_range[1], beta_bounds[1], beta_bounds[1], i_range[1], i_range[1], gamma_bounds[1]
        ]

        # Smart initial guess in bounds
        p0 = [
            E_corr_initial,
            0.05,       # beta_an
            0.07,       # beta_cath
            i_corr_initial,
            i_L_guess,
            gamma_guess
        ]
        # Clip each guess into bounds for safety
        p0 = np.clip(p0, bounds_lower, bounds_upper)

        # Fitting
        try:
            params, _ = curve_fit(
                self.mixed_control_fit, E, i_density,
                p0=p0, bounds=(bounds_lower, bounds_upper), method='trf', maxfev=100_000
            )
        except Exception as e:
            st.error(f"Fitting failed: {e}")
            params = [np.nan]*6

        i_fit = self.mixed_control_fit(E, *params)
        residuals = i_density - i_fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((i_density - np.mean(i_density)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        fit_result = {
            'E_corr': float(params[0]),
            'beta_an': float(params[1]),
            'beta_cath': float(params[2]),
            'i_corr': float(params[3]),
            'i_L': float(params[4]),
            'gamma': float(params[5]),
            'corrosion_rate': float(params[3]) * self.material_factor,
            'R_squared': float(r_squared),
        }
        return fit_result, params

    def plot_full_fit(self, E, i, params, fit_result):
        # Plot experimental and full fit (use abs for log plot)
        fig, ax = plt.subplots(figsize=(10, 6))
        E_full = np.linspace(np.min(E), np.max(E), 500)
        i_full_fit = self.mixed_control_fit(E_full, *params)

        ax.semilogy(E, np.abs(i), 'o', color='navy', label='Experimental Data')
        ax.semilogy(E_full, np.abs(i_full_fit), 'r-', lw=2, label='Mixed Control Fit')
        ax.axvline(fit_result['E_corr'], color='k', linestyle='--', label=f'E_corr = {fit_result["E_corr"]:.3f} V')
        ax.set_xlabel('Potential (V)')
        ax.set_ylabel('|Current| (A)' if self.area == 1.0 else '|Current Density| (A/m²)')
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

        # Pick columns
        col_options = list(df.columns)
        E_col = st.selectbox("Select the column for Potential (V)", col_options, index=0)
        i_col = st.selectbox("Select the column for Current (A)", col_options, index=min(2, len(col_options)-1))
        E = df[E_col].values
        i = df[i_col].values

        # Plot raw data
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(E, i, marker='o', label='Raw Data')
        ax.set_title('Raw Data')
        ax.set_xlabel('Potential (V)')
        ax.set_ylabel('Current (A)')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        analyzer = TafelAnalyzer()
        fit_result, params = analyzer.fit_polarization_data(E, i)
        analyzer.plot_full_fit(E, i, params, fit_result)

        st.subheader("Mixed Control Fit Parameters:")
        results_disp = {
            'E_corr (V)': fit_result['E_corr'],
            'beta_an (V/dec)': fit_result['beta_an'],
            'beta_cath (V/dec)': fit_result['beta_cath'],
            'i_corr (A)': fit_result['i_corr'],
            'i_L (A)': fit_result['i_L'],
            'gamma': fit_result['gamma'],
            'Corrosion Rate (mm/y*)': fit_result['corrosion_rate'],
            'R_squared': fit_result['R_squared'],
        }
        st.table({k: round(v, 4) if isinstance(v, float) else v for k, v in results_disp.items()})

        if fit_result["R_squared"] < 0.9:
            st.warning("Low R² suggests model/data mismatch or noise. Try verifying your data or cleaning if necessary.")

    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")

def main():
    st.title("Tafel Mixed-Control Automated Fitting")
    st.markdown("""
    Upload your polarization data file (Excel).  
    Select the correct columns. No other tuning is required.
    The app will attempt to auto-fit a mixed activation/diffusion model over the data.
    """)

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if uploaded_file is not None:
        process_excel(uploaded_file)

if __name__ == "__main__":
    main()
