import numpy as np
import pandas as pd


class KalmanHedgeRatio:
    # state [beta, alpha], obs y_t = [x_t, 1] @ state + noise
    # Q = delta/(1-delta)*I (Chan's parameterization)

    def __init__(self, delta: float = 1e-4, R: float | None = None):
        self.delta = delta
        self._R_init = R
        self.R = R if R is not None else 0.001
        self.Q = delta / (1 - delta) * np.eye(2)
        self.P = np.eye(2) * 1.0
        self.x = np.array([1.0, 0.0])  # [beta, alpha]
        self.last_S: float = 1.0  # innovation variance from last update

    def update(self, y_t: float, x_t: float) -> tuple[float, float, float]:
        H = np.array([x_t, 1.0])

        # Predict
        P_pred = self.P + self.Q

        # Innovation variance S_t = H @ P_pred @ H.T + R
        # Used for z-score normalization: e_t / sqrt(S_t) ~ N(0,1) under model
        S = float(H @ P_pred @ H.T + self.R)
        self.last_S = S

        # Innovation (uses prior state — look-ahead-free)
        innovation = y_t - H @ self.x

        # Kalman gain and state update
        K = P_pred @ H.T / S
        self.x = self.x + K * innovation
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred

        return float(self.x[0]), float(self.x[1]), float(innovation)

    def reset(self) -> None:
        self.P = np.eye(2) * 1.0
        self.x = np.array([1.0, 0.0])
        self.last_S = 1.0


def calibrate_R(log_y: pd.Series, log_x: pd.Series) -> float:
    # R must match daily-change variance of OLS spread, not the level variance.
    # Level variance (~0.016) inflates sqrt(S_t) ~8x and kills all z-scores.
    from numpy.linalg import lstsq
    X = np.column_stack([log_x.values, np.ones(len(log_x))])
    coeffs = lstsq(X, log_y.values, rcond=None)[0]
    resid = log_y.values - X @ coeffs
    return float(np.var(np.diff(resid)))


def kalman_spread(
    log_y: pd.Series,
    log_x: pd.Series,
    delta: float = 1e-4,
    R: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if R is None:
        R = calibrate_R(log_y, log_x)

    kf = KalmanHedgeRatio(delta=delta, R=R)
    betas, alphas, innovations, innov_stds = [], [], [], []

    for y_t, x_t in zip(log_y.values, log_x.values):
        beta, alpha, innov = kf.update(float(y_t), float(x_t))
        betas.append(beta)
        alphas.append(alpha)
        innovations.append(innov)
        innov_stds.append(np.sqrt(kf.last_S))

    return np.array(betas), np.array(alphas), np.array(innovations), np.array(innov_stds)
