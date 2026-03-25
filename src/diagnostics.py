import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import learning_curve


def plot_residuals(y_true, y_pred, name="Model"):
    """4-panel residual analysis: pred vs actual, distribution, homoscedasticity, Q-Q."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(y_pred, y_true, alpha=0.4, s=15, c="#1E90FF")
    lims = [min(min(y_pred), min(y_true)), max(max(y_pred), max(y_true))]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Predicted ($)")
    ax.set_ylabel("Actual ($)")
    ax.set_title(f"{name}: Predicted vs Actual")
    ax.legend()

    # 2. Residual distribution
    ax = axes[0, 1]
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="#2ecc71")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual ($)")
    ax.set_ylabel("Count")
    ax.set_title(f"{name}: Residual Distribution")
    ax.text(
        0.95,
        0.95,
        f"Mean: ${np.mean(residuals):,.0f}\nStd: ${np.std(residuals):,.0f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # 3. Residuals vs Predicted (homoscedasticity check)
    ax = axes[1, 0]
    ax.scatter(y_pred, residuals, alpha=0.4, s=15, c="#e74c3c")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted ($)")
    ax.set_ylabel("Residual ($)")
    ax.set_title(f"{name}: Residuals vs Predicted (Homoscedasticity)")

    # 4. Q-Q plot (normality of residuals)
    ax = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f"{name}: Q-Q Plot (Normality of Residuals)")
    ax.get_lines()[0].set(marker="o", markersize=3, color="#9b59b6", alpha=0.5)
    ax.get_lines()[1].set(color="red", linewidth=1.5)

    plt.tight_layout()
    return fig


def plot_learning_curves(model, X_train, y_train, name="Model"):
    """Plot learning curves showing bias-variance tradeoff."""
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="blue",
    )
    ax.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.1,
        color="orange",
    )
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training R2")
    ax.plot(train_sizes, val_mean, "o-", color="orange", label="Validation R2")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R2 Score")
    ax.set_title(f"{name}: Learning Curve")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_importance(imp_df, name="Model"):
    """Horizontal bar chart of permutation importance."""
    imp_df = imp_df.sort_values("importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(imp_df) * 0.4)))

    def _bar_color(v):
        if v > 0.05:
            return "#e74c3c"
        if v > 0.01:
            return "#3498db"
        return "#bdc3c7"

    colors = [_bar_color(v) for v in imp_df["importance_mean"]]
    ax.barh(
        imp_df["feature"],
        imp_df["importance_mean"],
        xerr=imp_df["importance_std"],
        color=colors,
        edgecolor="black",
        alpha=0.8,
        capsize=3,
    )
    ax.set_xlabel("Permutation Importance (R2 decrease)")
    ax.set_title(f"{name}: Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return fig


def plot_model_comparison(all_results):
    """Bar chart comparing test R² and Adjusted R² across all models."""
    names = []
    r2_scores = []
    adj_r2_scores = []
    for name, res in all_results.items():
        display_name = name.replace("_", "\n")
        names.append(display_name)
        r2_scores.append(res.loc["R2", "Test"])
        adj_r2_scores.append(res.loc["Adj_R2", "Test"])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, r2_scores, width, label="R2", color="#3498db", edgecolor="black", alpha=0.8
    )
    bars2 = ax.bar(
        x + width / 2,
        adj_r2_scores,
        width,
        label="Adjusted R2",
        color="#2ecc71",
        edgecolor="black",
        alpha=0.8,
    )

    for bars, scores in [(bars1, r2_scores), (bars2, adj_r2_scores)]:
        for bar, score in zip(bars, scores, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: R2 vs Adjusted R2 (Test Set)")
    ax.set_ylim(0.6, max(r2_scores) + 0.06)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig
