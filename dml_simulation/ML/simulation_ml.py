"""
Monte Carlo Simulation Study with Machine Learning
Predicting Y from high-dimensional discrete X using cross-validation

Demonstrates:
1) Out-of-sample prediction via CV
2) Bias-variance tradeoff
3) Performance comparison of ML learners
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ----------------------------
# Data-generating process
# ----------------------------
def sample_X(n, p, rng):
    """High-dimensional binary covariates"""
    return rng.binomial(1, 0.5, size=(n, p))


def sample_Y(X, beta, sigma, rng):
    """Linear signal + noise"""
    eps = rng.normal(0, sigma, size=X.shape[0])
    return X @ beta + eps


# ----------------------------
# Machine learning with CV
# ----------------------------
def ml_predict_cv_fast(X, Y, learner="lasso", n_folds=5, seed=42):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    if learner == "lasso":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(max_iter=5000))
        ])
        param_grid = {"model__alpha": np.logspace(-3, 1, 10)}

    elif learner == "elasticnet":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=5000))
        ])
        param_grid = {
            "model__alpha": np.logspace(-3, 1, 10),
            "model__l1_ratio": [0.1, 0.5, 0.9]
        }

    elif learner == "rf":
        pipe = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=seed
        )
        param_grid = {}

    elif learner == "gboost":
        pipe = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=3,
            random_state=seed
        )
        param_grid = {}

    else:
        raise ValueError("Unknown learner")

    # Hyperparameter tuning ONCE
    if param_grid:
        grid = GridSearchCV(
            pipe,
            param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        grid.fit(X, Y)
        best_model = grid.best_estimator_
    else:
        best_model = pipe.fit(X, Y)

    # Out-of-fold predictions with FIXED hyperparameters
    y_hat = cross_val_predict(
        best_model,
        X,
        Y,
        cv=kf,
        n_jobs=-1
    )

    return y_hat

def ols_predict_cv(X, Y, n_folds=5, seed=42):
    """Out-of-sample predictions from OLS using CV"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    model = LinearRegression()
    y_hat = cross_val_predict(model, X, Y, cv=kf)
    return y_hat


# ----------------------------
# One Monte Carlo run
# ----------------------------
def run_once(n, p, beta, sigma, learner, rng):
    X = sample_X(n, p, rng)
    Y = sample_Y(X, beta, sigma, rng)
    if learner == "ols":
        Y_hat = ols_predict_cv(X, Y)
    else:
        Y_hat = ml_predict_cv_fast(X, Y, learner=learner)
    mse = np.mean((Y - Y_hat) ** 2)
    return mse


# ----------------------------
# Monte Carlo simulation
# ----------------------------
def monte_carlo(n, p, beta, sigma, learner, n_rep, seed=42):
    rng = np.random.default_rng(seed)
    mses = []

    for _ in range(n_rep):
        mse = run_once(n, p, beta, sigma, learner, rng)
        mses.append(mse)

    return np.array(mses)


# ----------------------------
# Plots
# ----------------------------
def plot_mse_distribution(mses, learner):
    plt.figure(figsize=(8, 5))
    plt.hist(mses, bins=30, density=True, alpha=0.7)
    plt.title(f"MSE Distribution ({learner})")
    plt.xlabel("Out-of-sample MSE")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"mse_distribution_{learner}.png", dpi=300)
    plt.close()


def plot_learner_comparison(results):
    learners = list(results.keys())
    avg_mse = [results[l].mean() for l in learners]

    plt.figure(figsize=(8, 5))
    plt.bar(learners, avg_mse)
    plt.ylabel("Average out-of-sample MSE")
    plt.title("Machine Learner Comparison")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "learner_comparison.png", dpi=300)
    plt.close()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Parameters
    n = 200          # sample size
    p = 500          # number of covariates (p >> n)
    sigma = 1.0
    n_rep = 50     # Monte Carlo repetitions

    rng = np.random.default_rng(123)
    beta = rng.normal(0, 1, size=p)
    beta[50:] = 0    # sparse signal

    learners = ["ols", "lasso", "elasticnet", "rf", "gboost"]
    results = {}

    print("Monte Carlo ML Simulation")
    print("--------------------------")

    for learner in learners:
        print(f"Running learner: {learner}")
        mses = monte_carlo(
            n=n,
            p=p,
            beta=beta,
            sigma=sigma,
            learner=learner,
            n_rep=n_rep
        )
        results[learner] = mses

        print(f"  Avg MSE: {mses.mean():.4f}")
        print(f"  Std MSE: {mses.std():.4f}")

        plot_mse_distribution(mses, learner)

    plot_learner_comparison(results)

    print("\nFigures saved to:", FIGURES_DIR)