import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class TafelAnalyzer:
    """
    Tafel and mixed-control model fitting for electrochemical corrosion data.
    """
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
        """
        Mixed control fit (all data).
        """
        i_density = np.abs(i) / self.area
        E_corr_initial = float(np.median(E))
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

    def plot_fit(self, E, i, fit_result, title_suffix=""):
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
        ax.set_title(f'Tafel Analysis (R² = {fit_result["R_squared"]:.4f}) {title_suffix}')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    def classic_tafel_fit(self, E, i):
        """
        Fit classic Tafel slopes in log|i| = mE + c format for both branches.
        """
        # Guess Ecorr as median or min(abs(i))
        idx_corr = np.argmin(np.abs(i))
        ecorr_guess = E[idx_corr]
        dE = 0.04  # 40 mV window to avoid crossover
        mask_cath = E < (ecorr_guess - dE)
        mask_an = E > (ecorr_guess + dE)
        # Only fit if enough points
        results = {}

        for branch, mask in [('cathodic', mask_cath), ('anodic', mask_an)]:
            if np.sum(mask) > 8:
                p, cov = np.polyfit(E[mask], np.log10(np.abs(i[mask])), 1, cov=True)
                m, c = p
                y_fit = m * E[mask] + c
                y_obs = np.log10(np.abs(i[mask]))
                r2 = 1 - np.sum((y_obs - y_fit) ** 2) / np.sum((y_obs - y_obs.mean()) ** 2)
                results[branch] = dict(
                    slope=m, intercept=c, R_squared=r2,
                    beta=2.303 / abs(m) if m != 0 else np.nan
                )
            else:
                results[branch] = dict(slope=np.nan, intercept=np.nan, R_squared=np.nan, beta=np.nan)
        results['E_corr_guess'] = ecorr_guess
        return results

    def plot_classic_tafel(self, E, i, fit_result, region):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(E, np.log10(np.abs(i)), 'o', label='Experimental log|i|')

        for branch, color in zip(['cathodic', 'anodic'], ('b', 'r')):
            res = fit_result[branch]
            if np.isfinite(res['slope']):
                mask = ((E < region[0]) if branch == 'cathodic' else (E > region[1]))
                ax.plot(E[mask], res['slope'] * E[mask] + res['intercept'], color=color, lw=2, 
                        label=f"{branch.title()} fit\nβ={res['beta']:.3f} V/dec\nR²={res['R_squared']:.3f}")

        ax.axvline(fit_result['E_corr_guess'], color='k', linestyle='--', label='Ecorr guess')
        ax.set_xlabel('Potential (V)')
        ax.set_ylabel('log10(|Current|/A)')
        ax.set_title('Classic Tafel plot')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

def process_excel(file, area, material_factor, use_classic_tafel):
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

        # Potential region slider
        Emin, Emax = float(np.min(E)), float(np.max(E))
        region = st.slider(
            "Select Potential Range (V) for Fitting", 
            min_value=Emin, max_value=Emax, 
            value=(Emin, Emax), step=0.001
        )
        mask = (E >= region[0]) & (E <= region[1])
        E_fit, i_fit = E[mask], i[mask]

        # Raw data plot
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

        if not use_classic_tafel:
            fit_result = analyzer.fit_polarization_data(E_fit, i_fit)
            analyzer.plot_fit(E_fit, i_fit, fit_result, title_suffix=f"(fit region: {region[0]:.3f} to {region[1]:.3f} V)")
            st.subheader("Mixed Control Fit Results")
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
            st.table({k: round(v,4) if isinstance(v, float) else v for k,v in results_disp.items()})
            if fit_result["R_squared"] < 0.9:
                st.warning("Low R² suggests model/data mismatch or inappropriate region/model. Try adjusting region, or use classic Tafel mode.")
        else:
            fit_res = analyzer.classic_tafel_fit(E_fit, i_fit)
            analyzer.plot_classic_tafel(E_fit, i_fit, fit_res, region)
            st.subheader("Classic (Linear Tafel) Fit Results")
            for branch in ['cathodic', 'anodic']:
                v = fit_res[branch]
                st.markdown(
                    f"**{branch.title()} branch:** β = {v['beta']:.4f} V/dec, R² = {v['R_squared']:.4f}, Slope = {v['slope']:.4f}, Ecorr guess = {fit_res['E_corr_guess']:.3f} V"
                )
            st.info("Classic Tafel fit is basic and may not reflect full mechanisms, but can help when mixed-control model gives poor R².")

    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")

def main():
    st.title("Tafel Analysis Web App")
    st.write("Upload an Excel file with polarization data.\nSelect your voltage/current columns. Choose potential region for fitting. Adjust advanced options if needed!")

    with st.expander("Advanced Options (optional)"):
        area = st.number_input("Electrode area (m²)", value=1e-4, format="%.1e")
        material_factor = st.number_input("Material factor (default ~0.327 for mild steel in mm/y)", value=0.327, format="%.3f")
        use_classic_tafel = st.checkbox("Use classic Tafel fit (for simple activation control)", value=False)
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if uploaded_file is not None:
        process_excel(uploaded_file, area, material_factor, use_classic_tafel)

if __name__ == "__main__":
    main()
