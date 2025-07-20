import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class TafelAnalyzer:
    """
    A class for Tafel extrapolation and corrosion analysis of polarization data.
    """
    def __init__(self, area=1e-4, material_factor=0.327):
        self.area = area
        self.material_factor = material_factor

    def _mixed_control_fit(self, E, E_corr, beta_an, beta_cath, i_corr, i_L, gamma):
        # Forward and backward electrode reactions (mixed control)
        anodic = i_corr * np.exp(2.303 * (E - E_corr) / beta_an)
        cathodic_base = (i_corr / i_L) * np.exp(2.303 * (E_corr - E) / beta_cath)
        cathodic = i_L * (cathodic_base * gamma / (1 + cathodic_base * gamma)) * (1 / gamma)
        return anodic - cathodic

    def _calculate_weights(self, E, E_corr, w_ac, W):
        # Weight more data around Ecorr
        weights = np.full_like(E, 100 - W)
        activation_mask = (E >= E_corr - w_ac) & (E <= E_corr + w_ac)
        weights[activation_mask] = W
        return 1 / weights

    def fit_polarization_data(self, E, i, W=95, w_ac=0.05, gamma_bounds=(1.5, 5)):
        """
        Fit polarization curve using mixed control model.
        """
        i_density = np.abs(i) / self.area
        E_corr_initial = np.median(E)

        bounds = (
            [E.min(), 0.05, 0.05, 1e-9, 1e-8, gamma_bounds[0]],  
            [E.max(), 0.3, 0.3, 1e-4, 1e-3, gamma_bounds[1]]
        )
        p0 = [E_corr_initial, 0.12, 0.06, 1e-7, 1e-5, 3]
        sigma = self._calculate_weights(E, E_corr_initial, w_ac, W)

        params, _ = curve_fit(
            self._mixed_control_fit, E, i_density,
            p0=p0, bounds=bounds, sigma=sigma, method='trf', maxfev=30000
        )

        i_corr = params[3]
        corrosion_rate = i_corr * self.material_factor

        # R^2 calculation
        i_fit = self._mixed_control_fit(E, *params)
        residuals = i_density - i_fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((i_density - np.mean(i_density)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return {
            'E_corr': params[0],
            'beta_an': params[1],
            'beta_cath': params[2],
            'i_corr': params[3],
            'i_L': params[4],
            'gamma': params[5],
            'corrosion_rate': corrosion_rate,
            'R_squared': r_squared,
        }

    def plot_fit(self, E, i, fit_result):
        """
        Plot experimental data and fitted Tafel curve.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        E_fit = np.linspace(E.min(), E.max(), 500)
        fit_params = [fit_result[k] for k in ['E_corr', 'beta_an', 'beta_cath', 'i_corr', 'i_L', 'gamma']]
        i_fit = self._mixed_control_fit(E_fit, *fit_params)

        ax.semilogy(E, np.abs(i) / self.area, 'o', label='Experimental')
        ax.semilogy(E_fit, np.abs(i_fit), 'r-', label='Fit')
        ax.axvline(fit_result['E_corr'], color='k', linestyle='--', 
                   label=f'E_corr = {fit_result["E_corr"]:.3f} V')
        ax.set_xlabel('Potential (V)')
        ax.set_ylabel('|Current Density| (A/m²)')
        ax.set_title(f'Tafel Analysis (R² = {fit_result["R_squared"]:.4f})')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

def process_excel(file, area=1e-4, material_factor=0.327):
    """
    Process uploaded Excel file for Tafel analysis.
    """
    st.write("Reading data...")
    try:
        df = pd.read_excel(file)
        st.write("First 5 rows of your data:")
        st.dataframe(df.head())

        # Prompt for potential/current column if not obvious
        col_options = df.columns.tolist()
        E_col = st.selectbox("Select the column for Potential (V)", col_options, index=0)
        i_col = st.selectbox("Select the column for Current (A)", col_options, index=min(2, len(col_options)-1))
        E = df[E_col].values
        i = df[i_col].values

        # Quick plot of raw data
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(E, i, marker='o')
        ax.set_title('Raw Data Analysis')
        ax.set_xlabel('Potential (V)')
        ax.set_ylabel('Current (A)')
        ax.grid(True)
        st.pyplot(fig)

        st.write("Cleaning and fitting data...")

        analyzer = TafelAnalyzer(area=area, material_factor=material_factor)
        fit_result = analyzer.fit_polarization_data(E, i)
        analyzer.plot_fit(E, i, fit_result)

        # Clean up result display:
        st.subheader("Fit Results")
        nice_results = {
            'E_corr (V)': fit_result['E_corr'],
            'beta_an (V/dec)': fit_result['beta_an'],
            'beta_cath (V/dec)': fit_result['beta_cath'],
            'i_corr (A/m²)': fit_result['i_corr'],
            'i_L (A/m²)': fit_result['i_L'],
            'gamma': fit_result['gamma'],
            'Corrosion Rate (mm/y or equiv.)': fit_result['corrosion_rate'],
            'R_squared': fit_result['R_squared'],
        }
        st.json({k: f"{v:.4e}" if isinstance(v, float) else v for k, v in nice_results.items()})

    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")

def main():
    st.title("Tafel Analysis Interface")
    st.write(
        "Upload an Excel file with polarization data to perform Tafel fitting.\n"
        "Columns should include voltage and current (any units, select below)."
    )

    # Optional advanced settings
    with st.expander("Advanced Options"):
        area = st.number_input("Electrode area (m²)", value=1e-4, format="%.1e")
        material_factor = st.number_input("Material factor for corrosion rate", value=0.327)

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if uploaded_file is not None:
        process_excel(uploaded_file, area=area, material_factor=material_factor)

if __name__ == "__main__":
    main()
