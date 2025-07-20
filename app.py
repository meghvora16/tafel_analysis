import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

class TafelAnalyzer:
    def __init__(self, area=1e-4, material_factor=0.327):
        self.area = area
        self.material_factor = material_factor
        
    def _mixed_control_fit(self, E, E_corr, beta_an, beta_cath, i_corr, i_L, gamma):
        anodic = i_corr * np.exp(2.303 * (E - E_corr) / beta_an)
        cathodic_base = (i_corr / i_L) * np.exp(2.303 * (E_corr - E) / beta_cath)
        cathodic = i_L * (cathodic_base*gamma / (1 + cathodic_base*gamma))*(1/gamma)
        return anodic - cathodic

    def _calculate_weights(self, E, E_corr, w_ac, W):
        weights = np.full_like(E, 100 - W)
        activation_mask = (E >= E_corr - w_ac) & (E <= E_corr + w_ac)
        weights[activation_mask] = W
        return 1 / weights

    def fit_polarization_data(self, E, i, W=95, w_ac=0.05, gamma_bounds=(2,4)):
        i_density = np.abs(i) / self.area
        E_corr_initial = E[np.argmin(np.abs(i))]
        
        bounds = (
            [E.min(), 0.01, 0.01, 1e-9, 1e-7, gamma_bounds[0]],
            [E.max(), 1.0, 1.0, 1e-3, 1e-2, gamma_bounds[1]]
        )
        
        p0 = [E_corr_initial, 0.1, 0.1, 1e-6, 1e-4, 3]
        
        sigma = self._calculate_weights(E, E_corr_initial, w_ac, W)
        
        params, pcov = curve_fit(
            self._mixed_control_fit, E, i_density,
            p0=p0, bounds=bounds, sigma=sigma, maxfev=10000
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

    def sensitivity_analysis(self, E, i, w_ac_range=np.linspace(0.02, 0.1, 5), 
                            W_range=np.linspace(50, 90, 5)):
        results = []
        for W in W_range:
            for w_ac in w_ac_range:
                try:
                    fit = self.fit_polarization_data(E, i, W=int(W), w_ac=w_ac)
                    results.append({
                        'W': W,
                        'w_ac': w_ac,
                        'beta_cath': fit['beta_cath'],
                        'i_L': fit['i_L']
                    })
                except:
                    continue
                    
        return pd.DataFrame(results)

    def _plot_sensitivity(self, results):
        df = pd.DataFrame(results)
        
        plt.figure(figsize=(12, 6))
        for W, group in df.groupby('W'):
            plt.plot(group['w_ac'], group['beta_cath'], 'o-', label=f'W={W}%')
            
        plt.xlabel('Activation Window (w_ac, V)')
        plt.ylabel('Cathodic Tafel Slope (V/dec)')
        plt.title('Sensitivity Analysis of Weight Parameters')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

def process_excel(file):
    analyzer = TafelAnalyzer()
    
    try:
        df = pd.read_excel(file, skiprows=6, names=["E/V", "i/A"]).dropna()
        E = df["E/V"].values
        i = df["i/A"].values
        
        fit_result = analyzer.fit_polarization_data(E, i)
        sensitivity_df = analyzer.sensitivity_analysis(E, i)
        
        # Plot Tafel Analysis
        E_fit = np.linspace(E.min(), E.max(), 500)
        i_fit = analyzer._mixed_control_fit(E_fit, *list(fit_result.values())[:6])
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(E, np.abs(i)/analyzer.area, 'o', label='Experimental')
        plt.semilogy(E_fit, np.abs(i_fit), 'r-', label='Fit')
        plt.axvline(fit_result['E_corr'], color='k', linestyle='--', 
                    label=f'E_corr = {fit_result["E_corr"]:.3f} V')
        plt.xlabel('Potential (V)')
        plt.ylabel('|Current Density| (A/m²)')
        plt.title('Tafel Analysis')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        
        # Show sensitivity analysis plot
        st.write("Sensitivity Analysis:")
        analyzer._plot_sensitivity(sensitivity_df)
        
        st.write("Fit Results:")
        st.json(fit_result)
        
    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")

# Setup Streamlit interface
st.title("Tafel Analysis Interface")
st.write("Upload an Excel file with polarization data (columns: E/V, i/A).")

uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")

if uploaded_file is not None:
    process_excel(uploaded_file)
