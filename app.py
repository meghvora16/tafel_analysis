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
        # Enhancement based on shared paper’s equations
        anodic = i_corr * np.exp(2.303 * (E - E_corr) / beta_an)
        cathodic_base = (i_corr / i_L) * np.exp(2.303 * (E_corr - E) / beta_cath)
        cathodic = i_L * (cathodic_base * gamma / (1 + cathodic_base * gamma)) * (1 / gamma)
        return anodic - cathodic

    def _calculate_weights(self, E, E_corr, w_ac, W):
        weights = np.full_like(E, 100 - W)
        activation_mask = (E >= E_corr - w_ac) & (E <= E_corr + w_ac)
        weights[activation_mask] = W
        return 1 / weights

    def fit_polarization_data(self, E, i, W=95, w_ac=0.05, gamma_bounds=(2, 4)):
        i_density = np.abs(i) / self.area
        E_corr_initial = np.median(E)  # Adjustable based on dataset review
        
        bounds = (
            [E.min(), 0.05, 0.05, 1e-9, 1e-7, gamma_bounds[0]],  
            [E.max(), 0.5, 0.5, 1e-3, 1e-2, gamma_bounds[1]]
        )
        
        p0 = [E_corr_initial, 0.3, 0.3, 1e-6, 1e-4, 3]

        sigma = self._calculate_weights(E, E_corr_initial, w_ac, W)
        
        params, pcov = curve_fit(
            self._mixed_control_fit, E, i_density,
            p0=p0, bounds=bounds, sigma=sigma, maxfev=20000
        )
        
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
        # Create fit visualization for deeper insights
        E_fit = np.linspace(E.min(), E.max(), 500)
        i_fit = self._mixed_control_fit(E_fit, *list(fit_result.values())[:6])
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(E, np.abs(i) / self.area, 'o', label='Experimental')
        plt.semilogy(E_fit, np.abs(i_fit), 'r-', label='Fit')
        plt.axvline(fit_result['E_corr'], color='k', linestyle='--', 
                    label=f'E_corr = {fit_result["E_corr"]:.3f} V')
        plt.xlabel('Potential (V)')
        plt.ylabel('|Current Density| (A/m²)')
        plt.title('Tafel Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()

def process_excel(file):
    analyzer = TafelAnalyzer()
    
    try:
        df = pd.read_excel(file)

        # Map the correct columns for Potential and Current
        E = df.iloc[:, 0].values  # Potential applied (V)
        i = df.iloc[:, 2].values  # WE(1).Current (A)
        
        # Initial raw data plot for analysis
        plt.figure(figsize=(10, 5))
        plt.plot(E, i, marker='o')
        plt.title('Raw Data Analysis')
        plt.xlabel('Potential (V)')
        plt.ylabel('Current (A)')
        plt.grid(True)
        st.pyplot()
        
        # Perform fit
        fit_result = analyzer.fit_polarization_data(E, i)
        
        # Plot results
        analyzer._plot_fit(E, i, fit_result)
        
        st.write("Fit Results:")
        st.json(fit_result)
        
    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")

# Setup Streamlit interface
st.title("Tafel Analysis Interface")
st.write("Upload an Excel file with polarization data.")

uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")

if uploaded_file is not None:
    process_excel(uploaded_file)
