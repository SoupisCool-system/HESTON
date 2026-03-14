# ==============================================================================
# HESTON MODEL CALIBRATION ENGINE (COS Method + Least Squares)
# ==============================================================================
import numpy as np
from scipy.optimize import least_squares
from HESTON_COS import HestonCOSMethod


class HestonCalibrator:
    """
    Calibration engine for the Heston Stochastic Volatility Model.

    Uses the COS Method (via HestonCOSMethod) for fast analytical pricing
    inside the optimization loop, and scipy.optimize.least_squares with the
    Trust Region Reflective (TRF) method for bounded parameter estimation.

    Parameters to calibrate: kappa, theta, sigma, rho, v0
    """

    # --- Default parameter bounds ---
    DEFAULT_BOUNDS = (
        [1e-4, 1e-4, 1e-4, -0.999, 1e-4],   # Lower bounds: [kappa, theta, sigma, rho, v0]
        [20.0, 2.0,  5.0,   0.999,  2.0 ]    # Upper bounds: [kappa, theta, sigma, rho, v0]
    )

    PARAM_NAMES = ['kappa', 'theta', 'sigma', 'rho', 'v0']

    def __init__(self, cos_engine=None, feller_weight=1.0, N_cos=256):
        """
        Parameters
        ----------
        cos_engine : HestonCOSMethod or None
            An existing COS engine instance. If None, a new one is created
            internally during each objective function evaluation.
        feller_weight : float
            Weight for the Feller condition penalty term. Set to 0 to disable.
            A value of 1.0–5.0 gently pushes the optimizer toward stable regimes.
        N_cos : int
            Number of COS expansion terms used for pricing (default: 256).
        """
        self.cos_engine = cos_engine
        self.feller_weight = feller_weight
        self.N_cos = N_cos

    def _build_engine(self, params):
        """Create a HestonCOSMethod instance from a parameter vector."""
        kappa, theta, sigma, rho, v0 = params
        return HestonCOSMethod(kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0)

    def _objective_function(self, params, market_data):
        """
        Compute the residual vector for least_squares optimization.

        residuals[i] = model_price_i - market_price_i

        An optional Feller condition penalty is appended as an extra residual
        if 2*kappa*theta < sigma^2.

        Parameters
        ----------
        params : array-like
            [kappa, theta, sigma, rho, v0]
        market_data : list of dict
            Each dict must contain: S0, K, r, T, market_price, option_type

        Returns
        -------
        residuals : np.ndarray
            Vector of (model - market) price differences, plus optional penalty.
        """
        kappa, theta, sigma, rho, v0 = params

        # Build a fresh COS engine with the current trial parameters
        engine = self._build_engine(params)

        residuals = []
        for opt in market_data:
            try:
                model_price = engine.price_european(
                    S0=opt['S0'],
                    K=opt['K'],
                    r=opt['r'],
                    T=opt['T'],
                    option_type=opt.get('option_type', 'call'),
                    N=self.N_cos
                )
                # Guard against NaN / Inf from numerical blowup
                if not np.isfinite(model_price):
                    model_price = 1e6  # large penalty
            except Exception:
                model_price = 1e6

            residuals.append(model_price - opt['market_price'])

        # --- Feller Condition Soft Penalty ---
        # Feller condition: 2 * kappa * theta > sigma^2
        # If violated, add a penalty proportional to the violation magnitude.
        if self.feller_weight > 0:
            feller_violation = sigma**2 - 2.0 * kappa * theta
            if feller_violation > 0:
                residuals.append(self.feller_weight * feller_violation)
            else:
                residuals.append(0.0)  # No penalty when satisfied

        return np.array(residuals)

    def calibrate(self, market_data, initial_guess, bounds=None,
                  method='trf', max_nfev=500, verbose=1):
        """
        Calibrate the Heston model to market option prices.

        Parameters
        ----------
        market_data : list of dict
            Each dict must contain keys:
                S0          : float — Spot price
                K           : float — Strike price
                r           : float — Risk-free rate
                T           : float — Time to maturity (in years)
                market_price: float — Observed market option price
                option_type : str   — 'call' or 'put' (default: 'call')
        initial_guess : list or array
            Starting values [kappa, theta, sigma, rho, v0].
        bounds : tuple of (lower, upper) or None
            Parameter bounds. If None, uses DEFAULT_BOUNDS.
        method : str
            Optimizer method: 'trf' (default, supports bounds) or 'lm'.
            Note: 'lm' does not support bounds.
        max_nfev : int
            Maximum number of function evaluations.
        verbose : int
            Verbosity level for the optimizer (0=silent, 1=progress, 2=detailed).

        Returns
        -------
        result : dict
            'kappa', 'theta', 'sigma', 'rho', 'v0': optimized parameters
            'cost'    : final sum of squared residuals
            'nfev'    : number of function evaluations used
            'success' : bool, whether optimizer converged
            'message' : optimizer termination message
            'params'  : array of [kappa, theta, sigma, rho, v0]
        """
        if bounds is None:
            bounds = self.DEFAULT_BOUNDS

        # If 'lm' is chosen, bounds are not supported — warn and remove
        if method == 'lm':
            bounds_arg = (-np.inf, np.inf)
            print("[WARNING] Levenberg-Marquardt ('lm') does not support bounds. Running unbounded.")
        else:
            bounds_arg = bounds

        # Run the optimizer
        opt_result = least_squares(
            fun=self._objective_function,
            x0=np.array(initial_guess, dtype=float),
            args=(market_data,),
            bounds=bounds_arg,
            method=method,
            max_nfev=max_nfev,
            verbose=verbose
        )

        # Unpack optimized parameters
        kappa, theta, sigma, rho, v0 = opt_result.x

        # Check Feller condition on final parameters
        feller_lhs = 2.0 * kappa * theta
        feller_rhs = sigma**2
        feller_satisfied = feller_lhs > feller_rhs

        return {
            'kappa': kappa,
            'theta': theta,
            'sigma': sigma,
            'rho': rho,
            'v0': v0,
            'params': opt_result.x,
            'cost': opt_result.cost,
            'nfev': opt_result.nfev,
            'success': opt_result.success,
            'message': opt_result.message,
            'feller_satisfied': feller_satisfied,
            'feller_lhs': feller_lhs,
            'feller_rhs': feller_rhs,
        }


# ==============================================================================
# DEMO: Calibrate to MU Options (DTE 29)
# ==============================================================================
if __name__ == "__main__":

    # --- 1. Market Data (10 APR 26 Expiry) ---
    S0 = 370.0
    r = 0.05
    T_days = 34
    T = T_days / 365.0

    market_options_raw = [
        {'K': 345.0, 'market_price': 52.750},  # (51.10 + 54.40) / 2
        {'K': 350.0, 'market_price': 49.825},  # (48.30 + 51.35) / 2
        {'K': 355.0, 'market_price': 46.900},  # (45.50 + 48.30) / 2
        {'K': 360.0, 'market_price': 44.025},  # (42.55 + 45.50) / 2
        {'K': 365.0, 'market_price': 41.675},  # (40.40 + 42.95) / 2
        {'K': 370.0, 'market_price': 38.925},  # (37.40 + 40.45) / 2
        {'K': 375.0, 'market_price': 36.525},  # (36.00 + 37.05) / 2
        {'K': 380.0, 'market_price': 33.825},  # (32.85 + 34.80) / 2
        {'K': 385.0, 'market_price': 32.000},  # (31.00 + 33.00) / 2
        {'K': 390.0, 'market_price': 29.675},  # (28.95 + 30.40) / 2
    ]

    # Build market_data in the format the calibrator expects
    market_data = [
        {
            'S0': S0,
            'K': opt['K'],
            'r': r,
            'T': T,
            'market_price': opt['market_price'],
            'option_type': 'call'
        }
        for opt in market_options_raw
    ]

    # --- 2. Initial Guess ---
    # [kappa, theta, sigma, rho, v0]
    initial_guess = [2.0, 0.10, 0.50, -0.50, 0.10]

    # --- 3. Run Calibration ---
    print("=" * 60)
    print("   HESTON CALIBRATION (COS Method + Trust Region Reflective)")
    print("=" * 60)
    print(f"   Spot: ${S0} | DTE: {T_days} days | Options: {len(market_data)}")
    print(f"   Initial guess: kappa={initial_guess[0]}, theta={initial_guess[1]}, "
          f"sigma={initial_guess[2]}, rho={initial_guess[3]}, v0={initial_guess[4]}")
    print("-" * 60)

    calibrator = HestonCalibrator(feller_weight=2.0, N_cos=256)
    result = calibrator.calibrate(
        market_data=market_data,
        initial_guess=initial_guess,
        method='trf',
        max_nfev=1000,
        verbose=1
    )

    # --- 4. Print Results ---
    print("\n" + "=" * 60)
    print("   CALIBRATION RESULTS")
    print("=" * 60)
    print(f"  Toc do hoi quy   (kappa) : {result['kappa']:.6f}")
    print(f"  Phuong sai dai han(theta): {result['theta']:.6f}")
    print(f"  Vol of Vol        (sigma): {result['sigma']:.6f}")
    print(f"  Tuong quan         (rho) : {result['rho']:.6f}")
    print(f"  Phuong sai ban dau  (v0) : {result['v0']:.6f}")
    print("-" * 60)
    print(f"  IV hien tai ~ sqrt(v0)   : {np.sqrt(result['v0'])*100:.2f}%")
    print(f"  IV dai han  ~ sqrt(theta): {np.sqrt(result['theta'])*100:.2f}%")
    print("-" * 60)
    print(f"  Convergence: {'[OK] YES' if result['success'] else '[X] NO'}")
    print(f"  Function evals: {result['nfev']}")
    print(f"  Total cost (Sum residual^2): {result['cost']:.6f}")
    feller_str = "[OK] SATISFIED" if result['feller_satisfied'] else "[X] VIOLATED"
    print(f"  Feller condition (2*kappa*theta > sigma^2): {feller_str}  "
          f"({result['feller_lhs']:.4f} vs {result['feller_rhs']:.4f})")

    # --- 5. Show Fit Quality ---
    print("\n" + "=" * 60)
    print("   FIT QUALITY (Model vs Market)")
    print("=" * 60)
    print(f"{'Strike':>8s}  {'Market':>10s}  {'Model':>10s}  {'Error':>10s}  {'Error%':>8s}")
    print("-" * 52)

    fitted_engine = HestonCOSMethod(
        kappa=result['kappa'], theta=result['theta'],
        sigma=result['sigma'], rho=result['rho'], v0=result['v0']
    )

    total_abs_error = 0.0
    for opt in market_data:
        model_price = fitted_engine.price_european(
            S0=opt['S0'], K=opt['K'], r=opt['r'], T=opt['T'],
            option_type='call', N=256
        )
        error = model_price - opt['market_price']
        pct_error = error / opt['market_price'] * 100
        total_abs_error += abs(error)
        print(f"{opt['K']:8.1f}  {opt['market_price']:10.3f}  {model_price:10.3f}  "
              f"{error:+10.3f}  {pct_error:+7.2f}%")

    print("-" * 52)
    print(f"Mean absolute error: ${total_abs_error / len(market_data):.3f}")
    print("=" * 60)