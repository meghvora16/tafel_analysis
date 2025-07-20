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
