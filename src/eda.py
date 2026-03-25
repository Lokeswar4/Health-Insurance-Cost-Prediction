import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.outliers_influence import variance_inflation_factor


def summarize(df):
    """Print basic dataset info and statistics."""
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nNumeric summary:\n{df.describe().round(2)}")
    print(f"\nCategorical summary:\n{df.describe(include='object')}")


def check_outliers(series):
    """Return outlier count and IQR bounds for a numeric series."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = ((series < lower) | (series > upper)).sum()
    return n_outliers, lower, upper


def analyze_target(df):
    """Analyze the target variable distribution and sub-population structure."""
    charges = df["charges"]

    print("\n--- Target Variable (charges) ---")
    print(f"Mean:     ${charges.mean():>10,.2f}")
    print(f"Median:   ${charges.median():>10,.2f}")
    print(f"Std:      ${charges.std():>10,.2f}")
    print(f"CV:       {charges.std() / charges.mean():.3f}")
    print(f"Skew:     {charges.skew():.3f}")
    print(f"Kurtosis: {charges.kurtosis():.3f}")
    print(f"log(charges) skew: {np.log1p(charges).skew():.3f}")

    print("\nPercentiles:")
    for p in [5, 25, 50, 75, 95]:
        print(f"  P{p:02d}: ${charges.quantile(p / 100):>10,.2f}")

    print("\n--- Sub-population Analysis ---")
    for smoker_status, group in df.groupby("smoker"):
        c = group["charges"]
        print(
            f"  smoker={smoker_status} (n={len(group)}): "
            f"mean=${c.mean():,.0f}, median=${c.median():,.0f}, std=${c.std():,.0f}"
        )

    smoker_mean = df.loc[df.smoker == "yes", "charges"].mean()
    nonsmoker_mean = df.loc[df.smoker == "no", "charges"].mean()
    ratio = smoker_mean / nonsmoker_mean
    print(f"  Smoker/Non-smoker cost ratio: {ratio:.1f}x")


def analyze_interactions(df):
    """Analyze key feature interactions that drive model performance."""
    print("\n--- Smoker x BMI Interaction (mean charges) ---")
    df = df.copy()
    df["bmi_group"] = pd.cut(
        df["bmi"], bins=[0, 30, float("inf")], labels=["BMI<30", "BMI>=30"], right=True
    )
    pivot = df.pivot_table(
        values="charges", index="bmi_group", columns="smoker", aggfunc=["mean", "count"]
    )
    print(pivot.round(0).to_string())
    mean_pivot = df.pivot_table(
        values="charges", index="bmi_group", columns="smoker", aggfunc="mean"
    )
    print(
        f"\n  Key insight: Smokers with BMI>=30 pay ${mean_pivot.loc['BMI>=30', 'yes']:,.0f} "
        f"vs ${mean_pivot.loc['BMI<30', 'yes']:,.0f} for BMI<30 smokers "
        f"({mean_pivot.loc['BMI>=30', 'yes'] / mean_pivot.loc['BMI<30', 'yes']:.1f}x)"
    )

    print("\n--- Smoker x Age Group Interaction (mean charges) ---")
    df["age_group"] = pd.cut(df["age"], bins=[17, 30, 45, 65], labels=["18-30", "31-45", "46-64"])
    pivot2 = df.pivot_table(
        values="charges", index="age_group", columns="smoker", aggfunc=["mean", "count"]
    )
    print(pivot2.round(0).to_string())

    print("\n--- Children x Smoker Interaction ---")
    pivot3 = df.pivot_table(
        values="charges",
        index=pd.cut(df["children"], bins=[-1, 0, 2, 5], labels=["0", "1-2", "3-5"]),
        columns="smoker",
        aggfunc="mean",
    )
    print(pivot3.round(0).to_string())


def compute_vif(X):
    """Compute Variance Inflation Factor for multicollinearity detection."""
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data = vif_data.sort_values("VIF", ascending=False)

    print("\n--- Variance Inflation Factor (VIF) ---")
    print("  (VIF > 5 = moderate, VIF > 10 = severe multicollinearity)")
    for _, row in vif_data.iterrows():
        if row["VIF"] > 10:
            flag = " !!!"
        elif row["VIF"] > 5:
            flag = " !"
        else:
            flag = ""
        print(f"  {row['feature']:<16} VIF = {row['VIF']:>8.2f}{flag}")
    return vif_data


def smoker_hypothesis_test(df):
    """T-test and Mann-Whitney U with effect sizes."""
    smoker_charges = df.loc[df["smoker"] == "yes", "charges"]
    nonsmoker_charges = df.loc[df["smoker"] == "no", "charges"]

    s_mean, ns_mean = smoker_charges.mean(), nonsmoker_charges.mean()
    s_med, ns_med = smoker_charges.median(), nonsmoker_charges.median()
    print(f"Smoker mean: ${s_mean:,.2f}  |  Non-smoker mean: ${ns_mean:,.2f}")
    print(f"Smoker median: ${s_med:,.2f}  |  Non-smoker median: ${ns_med:,.2f}")

    # Cohen's d effect size
    pooled_std = np.sqrt((smoker_charges.std() ** 2 + nonsmoker_charges.std() ** 2) / 2)
    cohens_d = (s_mean - ns_mean) / pooled_std
    if abs(cohens_d) > 0.8:
        size = "large"
    elif abs(cohens_d) > 0.5:
        size = "medium"
    else:
        size = "small"
    print(f"\nCohen's d: {cohens_d:.3f} ({size})")

    alpha = 0.05

    t_stat, t_pval = ttest_ind(smoker_charges, nonsmoker_charges, equal_var=False)
    print(f"\nWelch's t-test: t={t_stat:.4f}, p={t_pval:.2e}")
    print(f"  -> {'Reject' if t_pval < alpha else 'Fail to reject'} H0 (alpha={alpha})")

    u_stat, u_pval = mannwhitneyu(smoker_charges, nonsmoker_charges, alternative="two-sided")
    # rank-biserial correlation as effect size for Mann-Whitney
    n1, n2 = len(smoker_charges), len(nonsmoker_charges)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)
    print(f"\nMann-Whitney U: U={u_stat:.4f}, p={u_pval:.2e}")
    print(f"  Rank-biserial r: {r_rb:.3f}")
    print(f"  -> {'Reject' if u_pval < alpha else 'Fail to reject'} H0 (alpha={alpha})")

    return {
        "t_stat": t_stat,
        "t_pval": t_pval,
        "u_stat": u_stat,
        "u_pval": u_pval,
        "cohens_d": cohens_d,
        "rank_biserial_r": r_rb,
    }


# --- Plotting functions ---


def _subplots_row(n_cols):
    """Create a single-row figure with n_cols subplots, always returning a list of axes."""
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    return fig, axes


def plot_distributions(df, numerical_cols):
    """Histograms with KDE for each numerical column."""
    fig, axes = _subplots_row(len(numerical_cols))
    for ax, col in zip(axes, numerical_cols, strict=True):
        sns.histplot(data=df, x=col, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
    plt.tight_layout()
    return fig


def plot_boxplots(df, numerical_cols):
    """Box plots showing outlier counts."""
    fig, axes = _subplots_row(len(numerical_cols))
    for ax, col in zip(axes, numerical_cols, strict=True):
        sns.boxplot(data=df, y=col, ax=ax)
        n_out, _, _ = check_outliers(df[col])
        ax.set_title(f"{col}\nOutliers: {n_out}, Median: {df[col].median():.2f}")
    plt.tight_layout()
    return fig


def plot_smoker_scatter(df):
    """Scatter plots of charges vs BMI and age, colored by smoker status."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"yes": "#DC143C", "no": "#1E90FF"}

    for ax, x_col in zip(axes, ["bmi", "age"], strict=True):
        for status, color in colors.items():
            mask = df["smoker"] == status
            ax.scatter(
                df.loc[mask, x_col],
                df.loc[mask, "charges"],
                alpha=0.5,
                c=color,
                label=f"smoker={status}",
                s=15,
            )
        ax.set_xlabel(x_col)
        ax.set_ylabel("charges ($)")
        ax.set_title(f"Charges vs {x_col} by Smoker Status")
        ax.legend()

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df, method="spearman"):
    """Lower-triangle correlation heatmap."""
    corr = df.corr(method=method, numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))  # mask upper triangle, show lower
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="bwr", mask=mask, ax=ax)
    ax.set_title(f"{method.title()} Correlation")
    plt.tight_layout()
    return fig
