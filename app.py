import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class TafelAnalyzer:
    def __init__(self, area=1e-4, material_factor=0.327):
        self.area = area
        self.material_factor = material_factor
        
    def _mixed_control_fit(self, E, E_corr, beta_an, beta_cath, i_corr, i_L, gamma):
        anodic = i_corr * np.exp(2.303 * (E - E_corr) / beta_an)
        cathodic_base = (i_corr / i_L) * np.exp(2.303 * (E_corr - E) / beta_cath)
        cathodic = i_L * (cathodic_base * gamma / (1 + cathodic_base * gamma)) * (1 / gamma)
        return anodic - cathodic

    def _calculate_weights(self, E, E_corr, w_ac, W):
        weights = np.full_like(E, 100 - W)
        activation_mask = (E >= E_corr - w_ac) & (E <= E_corr + w_ac)
        weights[activation_mask] = W
        return 1 / weights
    
    def fit_polarization_data(self, E, i, W=95, w_ac=0.05, gamma_bounds=(1.5, 5)):
        i_density = np.abs(i) / self.area
        E_corr_initial = E[np.argmin(np.abs(i_density))]  # Midpoint of current near zero
        
        # Refine bounds to encompass likely range based on electrochemical theory
        bounds=(
            [E.min(), 0.05, 0.05, 1e-9, 1e-8, gamma_bounds[0]],  
            [E.max(), 0.3, 0.3, 1e-4, 1e-3, gamma_bounds[1]]
        )
        
        # Initial guesses based on empirically typical values
        p0 = [E_corr_initial, 0.12, 0.06, 1e-7, 1e-5, 3]

        sigma = self._calculate_weights(E, E_corr_initial, w_ac, W)
        
        # Curve fitting with higher iterations to ensure convergence
        params, pcov = curve_fit(
            self._mixed_control_fit, E, i_density,
            p0=p0, bounds=bounds, sigma=sigma, method='trf', maxfev=30000
        )

        # Corrosion rate calculation
        i_corr = params[3]
        corrosion_rate = i_corr * self.material_factor
        
        return {
            'E_corr': params[0],
            'beta_an': params[1],
            'beta_cath': params[2],
            'i_corr': params[3],
            'i_L': params[4],
            'gamma': params[5],
            'corrosion_rate': corrosion_rate,
            'covariance': pcov
        }

    def _plot_fit(self, E, i, fit_result):
        fig, ax = plt.subplots(figsize=(10, 6))
        E_fit = np.linspace(E.min(), E.max(), 500)
        i_fit = self._mixed_control_fit(E_fit, *list(fit_result.values())[:6])
        
        ax.semilogy(E, np.abs(i) / self.area, 'o', label='Experimental')
        ax.semilogy(E_fit, np.abs(i_fit), 'r-', label='Fit')
        ax.axvline(fit_result['E_corr'], color='k', linestyle='--', 
                   label=f'E_corr = {fit_result["E_corr"]:.3f} V')
        ax.set_xlabel('Potential (V)')
        ax.set_ylabel('|Current Density| (A/m²)')
        ax.set_title('Tafel Analysis')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

def process_excel(file):
    analyzer = TafelAnalyzer()
    
    try:
        df = pd.read_excel(file)

        # Columns likely refer to potential and current
        E = df.iloc[:, 0].values  
        i = df.iloc[:, 2].values
        
        # Plot raw data for diagnostics
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(E, i, marker='o')
        ax.set_title('Raw Data Analysis')
        ax.set_xlabel('Potential (V)')
        ax.set_ylabel('Current (A)')
        ax.grid(True)
        st.pyplot(fig)
        
        st.write("Cleaning and fitting data...")
        
        fit_result = analyzer.fit_polarization_data(E, i)
        analyzer._plot_fit(E, i, fit_result)
        
        st.write("Fit Results:")
        st.json(fit_result)
        
    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")

st.title("Tafel Analysis Interface")
st.write("Upload an Excel file with polarization data.")

uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")

if uploaded_file is not None:
    process_excel(uploaded_file)
