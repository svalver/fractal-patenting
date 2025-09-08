# Predict the peak of spatial inequality using a hyperbolic model
# This code includes data loading, hyperbolic fitting, Taylor's law fitting,
# and peak prediction based on the coefficient of variation (CV).
 
# Evolution of Networks Lab
# Sergi Valverde, 2025
# @svalver


# From the paper:  
# "Fractal clusters and urban scaling shape spatial inequality in U.S. patenting" 
# published in npj Complexity
# https://doi.org/10.1038/s44260-025-00054-y

# Authors:
# Salva Duran-Nebreda, Blai Vidiella,  R. Alexander Bentley and Sergi Valverde

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root_scalar

# -------------------------------
# 1. Data Loader
# -------------------------------
def load_cv_data(csv_path="./cv_tmp.csv", min_d=0.03):
    """
    Load mean and std dev data for different radius values from CSV,
    compute CV = std / mean, filter invalid entries.
    """
    df = pd.read_csv(csv_path)

    R_vals = df['scale'].values
    mean_vals = df['meanpat'].values
    std_vals = df['stdpat'].values

    mask = (mean_vals > 0) & (std_vals > 0) & (R_vals >= min_d)
    R_vals = R_vals[mask]
    mean_vals = mean_vals[mask]
    std_vals = std_vals[mask]
    CV_vals = std_vals / mean_vals

    sort_idx = np.argsort(R_vals)
    return R_vals[sort_idx], CV_vals[sort_idx], mean_vals[sort_idx], std_vals[sort_idx]


# -------------------------------
# 2. μ(R) Hyperbolic Fitter
# -------------------------------
class HyperbolicFitter:
    """
    Fits the hyperbolic form: μ(R) = c * R^α * (R_c - R)^(-β)
    """

    def __init__(self, R, Y, initial_guess=None, Rc_max=100.0):
        self.R = np.asarray(R)
        self.Y = np.asarray(Y)
        self._set_initial_guess(initial_guess)
        self._set_bounds(Rc_max)
        self._fit()
        self._store_results()

    def _model(self, logR, log_c, alpha, beta, log_Rc):
        R = np.exp(logR)
        R_c = np.exp(log_Rc)
        if np.any(R >= R_c):
            return np.full_like(R, np.nan)
        return log_c + alpha * logR - beta * np.log(R_c - R)

    def _set_initial_guess(self, guess):
        if guess is None:
            Rc_init = max(self.R) + 1.0
            guess = [np.log(1), 1.0, 1.0, np.log(Rc_init)]
        self.initial_guess = guess

    def _set_bounds(self, Rc_max):
        Rc_min = max(self.R) + 0.1
        self.bounds = (
            [np.log(1e-6), 0.0, 0.0, np.log(Rc_min)],
            [np.log(1e6), 10.0, 10.0, np.log(Rc_max)],
        )

    def _fit(self):
        self.popt, self.pcov = curve_fit(
            self._model,
            np.log(self.R),
            np.log(self.Y),
            p0=self.initial_guess,
            bounds=self.bounds,
            maxfev=10000
        )

    def _store_results(self):
        log_c, self.alpha, self.beta, log_Rc = self.popt
        self.c = np.exp(log_c)
        self.R_c = np.exp(log_Rc)

    def predict(self, R_eval):
        logR_eval = np.log(np.asarray(R_eval))
        return np.exp(self._model(logR_eval, *self.popt))

    def print_params(self):
        print("μ(R) - Fitted parameters:")
        print(f"  c     = {self.c:.4f}")
        print(f"  alpha = {self.alpha:.4f}")
        print(f"  beta  = {self.beta:.4f}")
        print(f"  R_c   = {self.R_c:.4f}")

    def plot(self, label="Empirical", n_points=300):
        R_fit = np.linspace(min(self.R), max(self.R), n_points)
        Y_fit = self.predict(R_fit)
        plt.loglog(self.R, self.Y, 'o', label=label)
        plt.loglog(R_fit, Y_fit, '-', label='Fitted μ(R)')
        plt.axvline(self.R_c, color='gray', linestyle='--', label=f'$R_c = {self.R_c:.2f}$')
        plt.xlabel('Radius $R$')
        plt.ylabel('μ(R)')
        plt.title('Fitted Mean Cluster Size μ(R)')


# -------------------------------
# 3. Composite Taylor Law Fitter
# -------------------------------

class TaylorLawFit:
    """
    Fits a logistically modulated power-law: σ = c * μ^γ * logistic(μ)
    Sergi Valverde (@svalver)
    Sant Just Desvern
    July 31th, 2025
    """

    def fit(self, mean_vals, std_vals, threshold_taylor=0.7, maxfev=10000):
        """
        Fit the composite Taylor's law model with informed initial guesses
        from tail fitting and empirical modulation estimation.
        """
        mu = np.asarray(mean_vals)
        sigma = np.asarray(std_vals)

        # --- Step 1: Fit power-law in the tail
        def power_law(mu, c, delta):
            return c * mu ** delta

        sorted_idx = np.argsort(mu)
        mu_sorted = mu[sorted_idx]
        sigma_sorted = sigma[sorted_idx]

        tail_cut = int(threshold_taylor * len(mu))
        mu_tail = mu_sorted[tail_cut:]
        sigma_tail = sigma_sorted[tail_cut:]

        popt_tail, _ = curve_fit(power_law, mu_tail, sigma_tail)
        c_tail, delta_tail = popt_tail

        # --- Step 2: Estimate empirical modulation
        mu_threshold = mu_sorted[tail_cut]
        mask_pre_asymptotic = mu < mu_threshold
        mu_mod = mu[mask_pre_asymptotic]
        sigma_mod = sigma[mask_pre_asymptotic]
        M_mu = sigma_mod / (mu_mod ** delta_tail)

        # --- Step 3: Fit logistic modulation to M(μ)
        def logistic_mod(mu, C, k, mu0):
            return C / (1 + np.exp(-k * (mu - mu0)))

        # Use median and max for robust starting point
        try:
            popt_mod, _ = curve_fit(
                logistic_mod, mu_mod, M_mu,
                p0=[np.max(M_mu), 0.1, np.median(mu_mod)],
                bounds=([0, 0.001, 1e-6], [np.inf, 20.0, np.max(mu_mod)]),
                maxfev=maxfev
            )
            C_init, k_init, mu0_init = popt_mod
        except Exception as e:
            print("⚠️ Logistic modulation prefit failed:", e)
            C_init, k_init, mu0_init = 1.0, 0.1, np.median(mu_mod)

        # --- Step 4: Use these as smart initial guesses for composite fit
        log_c_init = np.log10(c_tail)
        gamma_init = delta_tail

        p0 = [log_c_init, gamma_init, k_init, mu0_init]
        bounds = ([-10, 0.1, 0.001, 1e-6], [10, 2.5, 10.0, np.max(mu)])

        # Ensure initial guess is within bounds
        for i in range(len(p0)):
            lower, upper = bounds[0][i], bounds[1][i]
            if not (lower <= p0[i] <= upper):
                p0[i] = np.clip(p0[i], lower + 1e-6, upper - 1e-6)

        print("Initial guess p0:", p0)
        print("Bounds:", bounds)

        # --- Step 5: Full composite fit
        log_sigma = np.log10(sigma)
        self.params, _ = curve_fit(
            self._model, mu, log_sigma,
            p0=p0, bounds=bounds, maxfev=maxfev
        )

        # Unpack parameters
        self.c = 10 ** self.params[0]
        self.gamma = self.params[1]
        self.k = self.params[2]
        self.M0 = self.params[3]

        # Warn if degenerate
        if np.isclose(self.k, 0.0, atol=1e-4):
            print("⚠️ Warning: Fitted k ≈ 0 — Modulation function may be flat.")


    @staticmethod
    def _model(mu, log_c, gamma, k, M0):
        return log_c + gamma * np.log10(mu) + np.log10(1 / (1 + np.exp(-k * (mu - M0))))

    def predict(self, mean_vals):
        logistic = 1 / (1 + np.exp(-self.k * (mean_vals - self.M0)))
        return self.c * (mean_vals ** self.gamma) * logistic

    def print_params(self):
        print("Composite Taylor's Law - Fitted parameters:")
        print(f"  c     = {self.c:.4f}")
        print(f"  γ     = {self.gamma:.4f}")
        print(f"  k     = {self.k:.4f}")
        print(f"  M₀    = {self.M0:.4f}")

    def plot(self, mean_vals, std_vals):
        std_fit = self.predict(mean_vals)
        plt.plot(mean_vals, std_vals, 'o', label='Empirical', alpha=0.7)
        plt.plot(mean_vals, std_fit, '-', color='black',
                #  label=f'Fit: γ={self.gamma:.2f}, M₀={self.M0:.2f}')
                label=f'δ={self.gamma:.2f}, μ0={self.M0:.2f}')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Mean μ')
        plt.ylabel('Std Dev σ')
        plt.title("Fitted Taylor's Law")

    def plot_ax(self, ax,  mean_vals, std_vals, _label ="Simulation"):
        std_fit = self.predict(mean_vals)
        if _label:
            ax.plot(mean_vals, std_vals, 'o', alpha=0.7, label =_label)
        else:
            ax.plot(mean_vals, std_vals, 'o', alpha=0.7)
        ax.plot(mean_vals, std_fit, linestyle='--', color='black',
            # label=f'γ={self.gamma:.2f}, M0={self.M0:.2f}')
            label=f'δ={self.gamma:.2f}, μ0={self.M0:.2f}')
    
    def plot_modulation_ax(self, ax, mu_data, sigma_data, mu_max=5):
        """
        Plot predicted vs empirical modulation function on the given Axes object.

        Parameters:
            ax : matplotlib.axes.Axes
                The axes to plot on.
            mu_data : array-like
                Array of mean (μ) values.
            sigma_data : array-like
                Corresponding standard deviation (σ) values.
            mu_max : float, optional
                Maximum μ value for the modulation plot (default is 5).
        """
        # Convert to numpy arrays
        mu_data = np.asarray(mu_data)
        sigma_data = np.asarray(sigma_data)

        # Filter and sort valid μ values
        valid_mask = mu_data > 0
        mu_data = mu_data[valid_mask]
        sigma_data = sigma_data[valid_mask]
        sorted_indices = np.argsort(mu_data)
        mu_data = mu_data[sorted_indices]
        sigma_data = sigma_data[sorted_indices]

        # Cut to subrange where mu <= mu_max
        cutoff_idx = np.searchsorted(mu_data, mu_max, side='right')
        mu_range = mu_data[:cutoff_idx]
        sigma_range = sigma_data[:cutoff_idx]

        if len(mu_range) == 0:
            raise ValueError("No valid μ values found below mu_max. Adjust mu_max or input data.")

        # Predicted modulation
        M_pred = 1 / (1 + np.exp(-self.k * (mu_range - self.M0)))
        ax.plot(mu_range, M_pred, label='Predicted', color='black', linestyle='--')

        # Empirical modulation with c included
        M_emp = sigma_range / (self.c * (mu_range ** self.gamma))
        ax.plot(mu_range, M_emp, 'o', label='Empirical', alpha=0.6)

        # ax.set_xlabel(r'$\mu$')
        # ax.set_ylabel(r'$\mathcal{M}(\mu)$')
        # ax.set_title('Predicted vs Empirical Modulation')
        # ax.legend()
        # ax.grid(True, linestyle='--', alpha=0.3)


# -------------------------------
# 4. Peak Inequality Estimator
# -------------------------------
class PeakPredictor:
    """
    Predicts the peak of spatial inequality using a simplified model for CV(R)
    """

    def __init__(self, R_vals, CV_vals, fitter):
        self.R_vals = np.asarray(R_vals)
        self.CV_vals = np.asarray(CV_vals)
        self.fitter = fitter
        self.b = fitter.c
        self.rc = fitter.R_c
        self.alpha = fitter.alpha
        self.beta = fitter.beta

    def _cv_model(self, r, k_cv, gamma_cv, c_cv):
        mu = self.b * (r ** self.alpha) * ((self.rc - r + 1e-12) ** -self.beta)
        return (k_cv * (mu ** gamma_cv)) / (mu + c_cv + 1e-12)

    def fit(self, p0=[1.0, 1.0, 1e4], bounds=([0, 0, 0], [np.inf, 5.0, np.inf])):
        popt, _ = curve_fit(self._cv_model, self.R_vals, self.CV_vals, p0=p0, bounds=bounds)
        self.k_cv, self.gamma_cv, self.c_cv = popt
        self.CV_fit = self._cv_model(self.R_vals, *popt)

    def predict_peak(self):
        if self.gamma_cv >= 1:
            self.r_star = None
            return None
        mu_star = (self.gamma_cv * self.c_cv) / (1 - self.gamma_cv)

        def mu_diff(r):
            if r <= 0 or r >= self.rc:
                return np.inf
            mu = self.b * (r ** self.alpha) * ((self.rc - r) ** -self.beta)
            return mu - mu_star

        sol = root_scalar(mu_diff, bracket=[0.01, self.rc - 1e-6], method='brentq')
        self.r_star = sol.root if sol.converged else None
        return self.r_star

    def plot_fit(self, ax):
        ax.plot(self.R_vals, self.CV_fit, ':', label='Approx.', color='black')
        if self.r_star:
            ax.axvline(self.r_star, color='red', linestyle='-', label=f'$R^*$', alpha =0.8)
                       # label=f'$r^* = {self.r_star:.2f}$')




def fit_taylors_law_with_modulation(mu, sigma, threshold_taylor = 0.7):

    # --- Step 1: Fit Taylor's Law in the tail
    def power_law(mu, c, delta):
        return c * mu ** delta

    # Sort data by mu
    sorted_indices = np.argsort(mu)
    mu_sorted = mu[sorted_indices]
    sigma_sorted = sigma[sorted_indices]

    # Select tail region for fitting
    tail_cut = int(threshold_taylor * len(mu))
    mu_tail = mu_sorted[tail_cut:]
    sigma_tail = sigma_sorted[tail_cut:]

    # Fit the tail with a power law
    popt, _ = curve_fit(power_law, mu_tail, sigma_tail)
    c_fit, delta_fit = popt

    # --- Step 2: Compute empirical modulation function
    mu_threshold = mu_sorted[tail_cut]
    mask_pre_asymptotic = mu < mu_threshold
    mu_mod = mu[mask_pre_asymptotic]
    sigma_mod = sigma[mask_pre_asymptotic]
    M_mu = sigma_mod / (mu_mod ** delta_fit)

    # --- Step 3: Fit logistic-like modulation function
    def logistic_mod(mu, C, k, mu0):
        return C / (1 + np.exp(-k * (mu - mu0)))

    popt_mod, _ = curve_fit(logistic_mod, mu_mod, M_mu, p0=[np.max(M_mu), 0.01, np.median(mu_mod)])
    C_fit, k_fit, mu0_fit = popt_mod
    mu_fit = np.logspace(np.log10(min(mu_mod)), np.log10(max(mu_mod)), 200)
    M_fit = logistic_mod(mu_fit, *popt_mod)

    # --- Step 4: Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Taylor’s law fit
    axs[0].loglog(mu, sigma, 'o', label='Data', alpha=0.6)
    axs[0].loglog(mu, power_law(mu, c_fit, delta_fit), '--k', label=f'Fit: δ={delta_fit:.2f}')
    axs[0].set_xlabel(r'$\mu$')
    axs[0].set_ylabel(r'$\sigma$')
    axs[0].set_title("Taylor's Law Fit (Tail)")
    axs[0].legend()

    # Right: Modulation function (empirical + fitted)
    axs[1].plot(mu_mod, M_mu, 'o', alpha=0.7, label='Empirical')
    axs[1].plot(mu_fit, M_fit, 'r--', label='Logistic Fit')
    axs[1].set_xscale('log')
    axs[1].set_xlabel(r'$\mu$')
    axs[1].set_ylabel(r'$\mathcal{M}(\mu)$')
    axs[1].set_title("Fitted Modulation Function")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    # Return the fitted parameters
    return {
        "c_fit": c_fit,
        "delta_fit": delta_fit,
        "C_fit": C_fit,
        "k_fit": k_fit,
        "mu0_fit": mu0_fit
    }

