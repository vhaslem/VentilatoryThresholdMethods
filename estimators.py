from scipy.optimize import minimize
import numpy as np
from scipy.stats import t
from scipy.stats import truncnorm

# guarantees noise is only positive
def truncated_normal(mu, sigma, size, lower=0, upper=np.inf):
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)


class breakpointFitter:
    """
    A Python reimplementation of strucchange::breakpoints (1 breakpoint version).
    Always fits one breakpoint and two linear segments.
    """
    def __init__(self, min_size=5):
        self.min_size = min_size
        self.models_ = None          # [(beta, start, end), (beta, start, end)]
        self.breakpoints_ = None     # [bp_index]
        self.params_ = None          # [beta1, beta2]
        self.ssr_ = None
        self.aic_ = None
        self.bic_ = None

    # ---------- FIT ----------
    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)

        best_ssr = np.inf
        best_bp = None
        best_models = None

        # Always scan for at least one breakpoint
        for bp in range(self.min_size, n - self.min_size):
            # Segment 1
            X1 = np.column_stack((np.ones(bp), x[:bp]))
            beta1, *_ = np.linalg.lstsq(X1, y[:bp], rcond=None)
            y1_hat = X1 @ beta1
            ssr1 = np.sum((y[:bp] - y1_hat)**2)

            # Segment 2
            X2 = np.column_stack((np.ones(n - bp), x[bp:]))
            beta2, *_ = np.linalg.lstsq(X2, y[bp:], rcond=None)
            y2_hat = X2 @ beta2
            ssr2 = np.sum((y[bp:] - y2_hat)**2)

            total_ssr = ssr1 + ssr2
            if total_ssr < best_ssr:
                best_ssr = total_ssr
                best_bp = bp
                best_models = [(beta1, 0, bp), (beta2, bp, n)]

        # Always build a model — even if no improvement
        if best_models is None:
            # Fallback: no valid breakpoint found
            X = np.column_stack((np.ones(n), x))
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            best_models = [(beta, 0, n)]
            best_bp = x[n // 2]  # arbitrary midpoint

        # Save parameters
        self.models_ = best_models
        self.params_ = [m[0] for m in best_models]
        self.breakpoints_ = [best_bp]
        self.ssr_total_ = best_ssr
        self.x = x
        self.y = y

        # Compute AIC/BIC for model selection info
        k = sum(len(m[0]) for m in best_models)
        self.aic_ = n * np.log(best_ssr / n) + 2 * k
        self.bic_ = n * np.log(best_ssr / n) + np.log(n) * k
        return self

    # ---------- PREDICT ----------
    def predict(self, x_new):
        x_new = np.atleast_1d(x_new).astype(float)
        preds = []
        for x0 in x_new:
            segment = 0
            if self.breakpoints_ is not None:
                for bp in self.breakpoints_:
                    if x0 > bp:
                        segment += 1
                    else:
                        break
            segment = min(segment, len(self.models_) - 1)
            beta, start, end = self.models_[segment]
            preds.append(beta[0] + beta[1] * x0)
        return np.array(preds)

    # ---------- PREDICT INTERVAL ----------
    def predict_interval(self, x_new, alpha=0.05):
        x_new = np.atleast_1d(x_new).astype(float)
        preds, ci_lower, ci_upper, pi_lower, pi_upper = [], [], [], [], []

        for x0 in x_new:
            segment = 0
            if self.breakpoints_ is not None:
                for bp in self.breakpoints_:
                    if x0 > bp:
                        segment += 1
                    else:
                        break
            segment = min(segment, len(self.models_) - 1)
            beta, start, end = self.models_[segment]

            # Compute stats for this segment
            x_seg = self.x[start:end]
            X_seg = np.column_stack((np.ones_like(x_seg), x_seg))
            y_seg = X_seg @ beta

            n, p = X_seg.shape
            residuals = self.y[start:end] - y_seg
            ssr = np.sum(residuals**2)
            sigma2 = ssr / (n - p)
            XtX_inv = np.linalg.pinv(X_seg.T @ X_seg)

            x_vec = np.array([1.0, x0])
            var_mean = x_vec @ XtX_inv @ x_vec.T
            se_mean = np.sqrt(sigma2 * var_mean)
            se_pred = np.sqrt(sigma2 * (1 + var_mean))

            tcrit = t.ppf(1 - alpha/2, df=n - p)
            y_pred = float(x_vec @ beta)

            preds.append(y_pred)
            ci_lower.append(y_pred - tcrit * se_mean)
            ci_upper.append(y_pred + tcrit * se_mean)
            pi_lower.append(y_pred - tcrit * se_pred)
            pi_upper.append(y_pred + tcrit * se_pred)

        return [preds[0], ci_lower[0], ci_upper[0], pi_lower[0], pi_upper[0]]

    # ---------- DATA GENERATOR ----------
    
    def generate_data(self, n=200, break_x=80, slopes=(0.5, 1.5), intercepts=(0, 10), noise=0.5, seed=None, range = 30, scale = 30):
        if seed is not None:
            np.random.seed(seed)
        x = np.linspace(0, scale, n)
        y = np.empty_like(x)
        bp = int(np.searchsorted(x, break_x))

        y[:bp] = intercepts[0] + slopes[0] * x[:bp]
        y[bp:] = intercepts[1] + slopes[1] * x[bp:]
        y += truncated_normal(mu = 0, sigma = noise, size=n)
        self.x = x
        self.y = y
        return self


class vslopeModel:
    def __init__(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.params_ = None  # fitted parameters: m1, b1, m2, b2
        self.result_ = None

    # ---------------------------------------------------------------------
    # Piecewise linear model: breakpoint is determined by intersection
    # ---------------------------------------------------------------------
    def piecewise_linear(self, x, m1, b1, m2, b2):
        x_break = (b2 - b1) / (m1 - m2 + 1e-12)  # avoid divide-by-zero
        y1 = m1 * x + b1
        y2 = m2 * x + b2
        return np.where(x < x_break, y1, y2)

    # ---------------------------------------------------------------------
    # Sum of squared residuals for optimization
    # ---------------------------------------------------------------------
    def _loss(self, params):
        m1, b1, m2, b2 = params
        y_pred = self.piecewise_linear(self.x, m1, b1, m2, b2)
        return np.sum((self.y - y_pred) ** 2)

    # ---------------------------------------------------------------------
    # Fit model to data
    # ---------------------------------------------------------------------
    def fit(self, init_params=None, bounds=None):
        if init_params is None:
            # initial guess: mild slopes and intercepts
            init_params = [1.0, 0.0, 2.0, 0.0]

        if bounds is None:
            # reasonable slope/intercept range
            bounds = [(0, 10), (-20, 10), (0, 10), (-50, 10)]

        self.result_ = minimize(self._loss, init_params, bounds=bounds)
        self.params_ = self.result_.x
        return self

    # ---------------------------------------------------------------------
    # Predict y values for new x
    # ---------------------------------------------------------------------
    def predict(self, x_new):
        if self.params_ is None:
            raise ValueError("Model must be fitted before prediction.")
        m1, b1, m2, b2 = self.params_
        return self.piecewise_linear(np.asarray(x_new), m1, b1, m2, b2)

    # ---------------------------------------------------------------------
    # Intersection point (breakpoint)
    # ---------------------------------------------------------------------
    def intersection(self):
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        m1, b1, m2, b2 = self.params_
        x_star = (b2 - b1) / (m1 - m2 + 1e-12)
        y_star = m1 * x_star + b1
        return x_star, y_star

    # ---------------------------------------------------------------------
    # Generate synthetic data from a two-segment model
    # ---------------------------------------------------------------------
    def generate_sample_data(self, n=100, x_range=(0, 5),x_break = 3,
                             m1=1.0, b1=2.0, m2=4.0, b2=-5.0,
                             noise_std=0.5, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        x = np.linspace(x_range[0], x_range[1], n)
        # x_break = (b2 - b1) / (m1 - m2 + 1e-12)
        y_true = np.where(x < x_break, m1 * x + b1, m2 * x + b2)
        noise = np.random.normal(0, noise_std, size=n)
        self.x, self.y = x, y_true + noise
        return self

    # ---------------------------------------------------------------------
    # Return model summary
    # ---------------------------------------------------------------------
    def summary(self):
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        m1, b1, m2, b2 = self.params_
        x_star, y_star = self.intersection()
        return {
            "slope_1": m1,
            "intercept_1": b1,
            "slope_2": m2,
            "intercept_2": b2,
            "intersection_x": x_star,
            "intersection_y": y_star,
            "SSR": self._loss(self.params_)
        }

    def predict_interval(self, x_new, alpha=0.05):
        """
        Compute prediction and confidence intervals for the given x value.
        Uses OLS theory for each segment.
        """
        if self.params_ is None:
            raise ValueError("Model must be fitted before computing intervals.")
        if self.x is None or self.y is None:
            raise ValueError("No data found. Fit or generate data first.")

        m1, b1, m2, b2 = self.params_
        x_break = (b2 - b1) / (m1 - m2 + 1e-12)

        x_new = np.asarray(x_new)
        if x_new.ndim == 0:
            x_new = np.array([x_new])

        results = []

        for xi in x_new:
            if xi < x_break:
                mask = self.x < x_break
            else:
                mask = self.x >= x_break

            X = np.column_stack((np.ones_like(self.x[mask]), self.x[mask]))
            y = self.y[mask]
            beta = np.linalg.lstsq(X, y, rcond=None)[0]

            n, p = X.shape
            residuals = y - X @ beta
            s2 = np.sum(residuals**2) / (n - p)
            XtX_inv = np.linalg.pinv(X.T @ X)


            x_vec = np.array([1, xi])
            y_pred = float(x_vec @ beta)

            # Standard errors
            var_mean = x_vec @ XtX_inv @ x_vec.T
            se_mean = np.sqrt(s2 * var_mean)
            se_pred = np.sqrt(s2 * (1 + var_mean))

            tcrit = t.ppf(1 - alpha / 2, df=n - p)

            ci_lower = y_pred - tcrit * se_mean
            ci_upper = y_pred + tcrit * se_mean
            pi_lower = y_pred - tcrit * se_pred
            pi_upper = y_pred + tcrit * se_pred

            results.append([
                
                 y_pred,
                 ci_lower,
                ci_upper,
                 pi_lower,
                pi_upper
            ])

        return results if len(results) > 1 else results[0]



class exponentialFitter:
    """
    Fits an exponential model y = a * exp(b * x) + c
    by minimizing the sum of squared residuals (SSR).
    """

    def __init__(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.params_ = None
        self.result_ = None

    def model(self, x, a, b):
        """Exponential function (safe version)."""
        try:
            return a * np.exp(b * x)
        except RuntimeWarning:
            print(f"Overflow detected in model() for params: a={a}, b={b}")
            return a * np.exp(np.clip(b * x, -700, 700))


    def _loss(self, params):
        """Sum of squared residuals."""
        a, b = params
        y_pred = self.model(self.x, a, b)
        return np.sum((self.y - y_pred) ** 2)

    def fit(self, init_params=None, bounds=None):
        """Fit parameters by minimizing SSR."""
        if init_params is None:
            # Rough initial guess from data
            a0 = max(self.y) - min(self.y)
            b0 = 0.1
            
            init_params = [a0, b0]

        if bounds is None:
            bounds = [(0, 1e3), (0, 1)]

        self.result_ = minimize(self._loss, init_params, bounds=bounds)
        self.params_ = self.result_.x
        return self

    def predict(self, x_new):
        """Predict y for new x."""
        if self.params_ is None:
            raise ValueError("Model must be fitted before prediction.")
        return self.model(np.asarray(x_new), *self.params_)

    def summary(self):
        """Return fitted parameters and model fit statistics."""
        if self.params_ is None:
            raise ValueError("Model must be fitted first.")
        a, b= self.params_
        y_pred = self.predict(self.x)
        ssr = np.sum((self.y - y_pred) ** 2)
        sst = np.sum((self.y - np.mean(self.y)) ** 2)
        r2 = 1 - ssr / sst
        return {
            "a": a,
            "b": b,
            
            "SSR": ssr,
            "R2": r2
        }

      
    def generate_sample_data(self, n=100, x_range=(0, 5),
                             a=2.5, b=0.8,
                             noise_std=1.0, random_state=None):
        """
        Generate synthetic data from an exponential model with noise.

        Parameters
        ----------
        n : int
            Number of samples.
        x_range : tuple
            Range (min, max) of x values.
        a, b, c : floats
            True exponential parameters (y = a * exp(bx) + c).
        noise_std : float
            Standard deviation of Gaussian noise added to y.
        random_state : int or None
            Random seed for reproducibility.

        Returns
        -------
        x : np.ndarray
            Input x values.
        y : np.ndarray
            Output y values with noise.
        """
        if random_state is not None:
            np.random.seed(random_state)
        x = np.linspace(x_range[0], x_range[1], n)
        y_true = a * np.exp(np.clip(b * x, -700, 700))

        noise = truncated_normal(mu = 0, sigma = noise_std, size=n)
        self.x = x
        self.y = y_true + noise
        return self

    
    def compute_t_max(self):
        '''returns value at which t is maximized'''
        a, b = self.params_
        model_derivatives = a * b * np.exp(np.clip(b * self.x, -700, 700))

        derivative_diffs = []
        for i, vals in enumerate(model_derivatives[1:-1]):
            
            ## need to replace x[i + 1] - x[i] with derivative
            abs_diff = np.abs(model_derivatives[i + 1] -model_derivatives[i])
            derivative_diffs.append((i, abs_diff))
        t_max_tup = sorted(derivative_diffs, key=lambda x: x[1])[-1]
        t_max = t_max_tup[0]
        return np.log(self.x[int(t_max)])
    

    def prediction_interval(self, x_new, X=None, y=None, beta=None, alpha=0.05):
        """
        Compute 95% confidence and prediction intervals for a linear model prediction.

        Parameters
        ----------
        x_new : float or array-like
            New x value (scalar or array). Must include intercept term if model does.
        X : array-like, shape (n, p)
            Design matrix used for fitting.
        y : array-like, shape (n,)
            Response vector.
        beta : array-like, shape (p,)
            Estimated coefficients.
        alpha : float, optional
            Significance level (default = 0.05).

        Returns
        -------
        y_pred : float
            Predicted mean value.
        ci_lower, ci_upper : float
            Confidence interval bounds.
        pi_lower, pi_upper : float
            Prediction interval bounds.
        """

        # Ensure numpy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        beta = np.asarray(beta, dtype=float)

        # Prepare new x (column vector with intercept)
        x_new = np.atleast_1d(x_new)
        if x_new.ndim == 1:
            x_new = np.column_stack((np.ones_like(x_new), x_new))
        elif x_new.shape[1] == 1:
            x_new = np.column_stack((np.ones(x_new.shape[0]), x_new))

        n, p = X.shape

        # Predicted mean
        y_pred = x_new @ beta  # shape (m,1)

        # Residuals and variance estimate
        residuals = y - X @ beta
        ssr = np.sum(residuals**2)
        sigma2 = ssr / (n - p)

        # Compute covariance of beta
        XtX_inv = np.linalg.pinv(X.T @ X)

        # Variance of predicted mean
        var_mean = np.array([x @ XtX_inv @ x.T for x in x_new])
        se_mean = float(np.sqrt(sigma2 * var_mean))

        # Variance of prediction (includes +σ²)
        se_pred = float(np.sqrt(sigma2 * (1 + var_mean)))

        # t critical value
        t_crit = float(t.ppf(1 - alpha / 2, df=n - p))

        # Confidence intervals (mean)
        ci_lower = y_pred - t_crit * se_mean
        ci_upper = y_pred + t_crit * se_mean

        # Prediction intervals (new obs)
        pi_lower = y_pred - t_crit * se_pred
        pi_upper = y_pred + t_crit * se_pred

        return float(y_pred), float(ci_lower), float(ci_upper), float(pi_lower), float(pi_upper)


    def predict_interval(self, x_new, alpha=0.05):
        """
        Wrapper for exponential regression — computes prediction interval in log-space,
        then exponentiates the results for correct asymmetry.
        """
        # Construct design matrix with intercept
        X = np.column_stack((np.ones_like(self.x), self.x))
        y_log = np.log(np.maximum(self.y, 1e-6))
        beta = np.linalg.lstsq(X, y_log, rcond=None)[0]

        # Get prediction intervals in log-space
        y_pred_log, ci_l_log, ci_u_log, pi_l_log, pi_u_log = self.prediction_interval(
            x_new, X, y_log, beta, alpha
        )

        # Exponentiate to get asymmetric bounds in real space
        return [
            np.exp(y_pred_log),
            np.exp(ci_l_log),
            np.exp(ci_u_log),
            np.exp(pi_l_log),
            np.exp(pi_u_log)
        ]
    
    def approx_interval_prediction(self, x_new, alpha = 0.05):
        # call predict interval, get the low and high ci back
        conf_bounds = self.predict_interval(x_new = x_new, alpha = alpha)[1:3]
        abs_diff = (conf_bounds[1] - conf_bounds[0])/2
        return [x_new, x_new - abs_diff, x_new + abs_diff]



