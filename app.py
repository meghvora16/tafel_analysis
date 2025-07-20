import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class TafelAnalyzer:
    def __init__(self, area=1e-4, material_factor=0.327):
        self.area = area
        self.material_factor = material_factor

    def mixed_control_fit(self, E, E_corr, beta_an, beta_cath, i_corr, i_L, gamma):
        # Use signed current for better physical fitting!
        anodic = i_corr * np.exp(2.303 * (E - E_corr) / beta_an)
        cathodic_base = (i_corr / i_L) * np.exp(2.303 * (E_corr - E) / beta_cath)
        cathodic = i_L * (cathodic_base * gamma / (1 + cathodic_base * gamma)) * (1 / gamma)
        return anodic - cathodic

    def calculate_weights(self, E, E_corr, w_ac, W):
        weights = np.full_like(E, 100 - W)
        activation_mask = (E >= E_corr - w_ac) & (E <= E_corr + w_ac)
        weights[activation_mask] = W
        return 1 / weights

    def fit_polarization_data(self, E, i, W=95, w_ac=0.05, gamma_bounds=(1.1, 20)):
        # Use signed i for fitting!
        i_density = i / self.area

        # Robust guess for Ecorr: E where |i| is minimized
        idx_Ecorr = np.argmin(np.abs(i_density))
        E_corr_initial = E[idx_Ecorr]
        i_corr_initial = np.abs(i_density[idx_Ecorr]) + 1e-8

        # Wider, physical bounds
        bounds = (
            [E.min(),   0.01,   0.01,   1e-10, 1e-10, gamma_bounds[0]], # Lower
            [E.max(),   0.5,    0.5,    1e-2,  1e-1,  gamma_bounds[1]]  # Upper
        )
        p0 = [
            E_corr_initial,   # E_corr
            0.05,             # beta_an
            0.07,             # beta_cath
            i_corr_initial,   # i_corr
            max(np.abs(i_density))*.5, # i_L: half max current
            np.mean(gamma_bounds)      # gamma
        ]
        sigma = self.calculate_weights(E, E_corr_initial, w_ac, W)

        try:
            params, _ = curve_fit(
                self.mixed_control_fit, E, i_density,
                p0=p0, bounds=bounds, sigma=sigma, method='trf', maxfev=50000
            )
        except Exception as e:
            st.error(f"Curve fitting failed: {str(e)}")
            params = [np.nan]*6

        i_fit = self.mixed_control_fit(E, *params)
        # R^2 for signed current!
        residuals = i_density - i_fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((i_density - np.mean(i_density)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        fit_result = {
            'E_corr': params[0],
            'beta_an': params[1],
            'beta_cath': params[2],
            'i_corr': params[3],
            'i_L': params[4],
            'gamma': params[5],
            'corrosion_rate': params[3] * self.material_factor,
            'R_squared': r_squared,
        }
        return fit_result, params

    def plot_full_fit(self, E, i, params, fit_result, area, region=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        E_full = np.linspace(E.min(), E.max(), 500)
        i_full_fit = self.mixed_control_fit(E_full, *params)

        # Mark which data is fitted (region)
        if region is not None:
            mask = (E >= region[0]) & (E <= region[1])
            ax.semilogy(E[~mask], np.abs(i[~mask]) / area, 'o', color='gray', alpha=0.4, label='Outside fit region')
            ax.semilogy(E[mask], np.abs(i[mask]) / area, 'o', color='C0', label='Fit region data')
        else:
            ax.semilogy(E, np.abs(i)/area, 'o', label='Experimental')

        # Plot fit (using full E range!)
        ax.semilogy(E_full, np.abs(i_full_fit), 'r-', lw=2, label='Mixed Control Fit')
        ax.axvline(fit_result['E_corr'], color='k', linestyle='--', label=f'E_corr = {fit_result["E_corr"]:.3f} V')

        ax.set_xlabel('Potential (V)')
        ax.set_ylabel('|Current Density| (A/m²)')
        ax.set_title(f'Tafel Mixed Control Analysis (R² = {fit_result["R_squared"]:.4f})')
        ax.legend()
        ax.grid(True, which='both', ls='--')
        st.pyplot(fig)

def process_excel(file, area, material_factor):
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

        Emin, Emax = float(np.min(E)), float(np.max(E))
        region = st.slider(
            "Select Potential Range (V) for Fitting (optional, for best R²)",
            min_value=Emin, max_value=Emax,
            value=(Emin, Emax), step=0.001
        )
        mask = (E >= region[0]) & (E <= region[1])
        E_fit, i_fit = E[mask], i[mask]

        # Raw data with fit region shown
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(E, i, marker='o', label='Raw Data')
        ax.axvspan(region[0], region[1], color='yellow', alpha=0.2, label='Fit region')
        ax.set_title('Raw Data & Fit Region')
        ax.set_xlabel('Potential (V)')
        ax.set_ylabel('Current (A)')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        analyzer = TafelAnalyzer(area=area, material_factor=material_factor)
        fit_result, params = analyzer.fit_polarization_data(E_fit, i_fit)
        analyzer.plot_full_fit(E, i, params, fit_result, area, region=region)

        st.subheader("Mixed Control Fit Parameters (using region above):")
        results_disp = {
            'E_corr (V)': fit_result['E_corr'],
            'beta_an (V/dec)': fit_result['beta_an'],
            'beta_cath (V/dec)': fit_result['beta_cath'],
            'i_corr (A/m²)': fit_result['i_corr'],
            'i_L (A/m²)': fit_result['i_L'],
            'gamma': fit_result['gamma'],
            'Corrosion Rate': fit_result['corrosion_rate'],
            'R_squared': fit_result['R_squared'],
        }
        st.table({k: round(v, 4) if isinstance(v, float) else v for k, v in results_disp.items()})

        if fit_result["R_squared"] < 0.9:
            st.warning("Low R² suggests model/data mismatch or inappropriate fit region. Try adjusting region or rechecking parameters. You may get best results if you only fit a region close to E_corr (typically within ±0.2~0.3V).")

    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")

def main():
    st.title("Tafel Analysis – Mixed Activation/Diffusion Fit")
    st.markdown("""
    Upload your polarization data (potential & current columns, `.xlsx`).  
    Select correct columns and choose a region for best fitting.  
    If R² is low, try limiting fit to a region ±0.2V around Ecorr.
    The red curve overlays the *entire plot* for visual check.
    """)

    with st.expander("Advanced Options (optional)"):
        area = st.number_input("Electrode area (m²)", value=1e-4, format="%.1e")
        material_factor = st.number_input("Material factor (default ≈0.327 for mild steel in mm/y)", value=0.327, format="%.3f")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if uploaded_file is not None:
        process_excel(uploaded_file, area, material_factor)

if __name__ == "__main__":
    main()
