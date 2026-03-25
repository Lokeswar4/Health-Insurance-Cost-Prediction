"""
Health Insurance Cost Prediction
================================
Predicts medical insurance costs from patient demographics using the
Kaggle Medical Cost Personal Dataset.

Usage:
    python main.py              # full pipeline (EDA + modeling)
    python main.py --model-only # skip EDA, run modeling only
    python main.py --save-plots # save all plots to outputs/
"""

import argparse
import logging

import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from src.config import NUMERICAL_FEATURES, OUTPUT_DIR
from src.data_loader import load_data
from src.diagnostics import (
    plot_feature_importance,
    plot_learning_curves,
    plot_model_comparison,
    plot_residuals,
)
from src.eda import (
    analyze_interactions,
    analyze_target,
    compute_vif,
    plot_boxplots,
    plot_correlation_heatmap,
    plot_distributions,
    plot_smoker_scatter,
    smoker_hypothesis_test,
    summarize,
)
from src.model import run_all_models
from src.persistence import FullPipeline, save_model
from src.preprocessing import add_interaction_features, preprocess, split_data

logger = logging.getLogger(__name__)


def _init_plot_saving():
    """Switch to non-interactive backend and ensure output directory exists."""
    matplotlib.use("Agg")
    OUTPUT_DIR.mkdir(exist_ok=True)


def _save_fig(fig, name: str) -> None:
    """Save a figure to the output directory and close it to free memory."""
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug("plot.saved name=%s dir=%s", path.name, OUTPUT_DIR)


def run_eda(df, save_plots: bool = False) -> None:
    print("\n" + "=" * 60)
    print(" EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    summarize(df)
    analyze_target(df)
    analyze_interactions(df)

    print("\n--- Hypothesis Test: Smoker vs Non-Smoker Charges ---")
    smoker_hypothesis_test(df)

    if save_plots:
        _init_plot_saving()

    fig_dist = plot_distributions(df, NUMERICAL_FEATURES + ["charges"])
    fig_box = plot_boxplots(df, NUMERICAL_FEATURES + ["charges"])
    fig_corr_s = plot_correlation_heatmap(df, method="spearman")
    fig_corr_p = plot_correlation_heatmap(df, method="pearson")
    fig_scatter = plot_smoker_scatter(df)

    if save_plots:
        for fig, name in [
            (fig_dist, "distributions"),
            (fig_box, "boxplots"),
            (fig_corr_s, "correlation_spearman"),
            (fig_corr_p, "correlation_pearson"),
            (fig_scatter, "smoker_scatter"),
        ]:
            _save_fig(fig, name)
        logger.info("plots.saved phase=eda count=5 dir=%s", OUTPUT_DIR)


def run_modeling(df, save_plots: bool = False):
    print("\n" + "=" * 60)
    print(" MODELING")
    print("=" * 60)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_proc, X_test_proc, preprocessor = preprocess(X_train, X_test)
    X_train_eng, X_test_eng = add_interaction_features(X_train_proc, X_test_proc)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    compute_vif(X_train_eng)

    results, fitted_models, imp_dfs = run_all_models(
        X_train_proc,
        y_train,
        X_test_proc,
        y_test,
        X_train_eng,
        X_test_eng,
    )

    # Save best model for deployment (retrained on full dataset for maximum data use)
    # Note: reported metrics are from the 80/20 split; the saved model uses all 1,337 rows
    print("\n" + "-" * 55)
    print(" MODEL PERSISTENCE")
    print("-" * 55)

    full_pipeline = FullPipeline()
    X_full = df.drop(columns=["charges"])
    full_pipeline.fit(X_full)
    X_full_transformed = full_pipeline.transform(X_full)

    deploy_lr = LinearRegression()
    deploy_lr.fit(X_full_transformed, df["charges"])
    save_model(deploy_lr, full_pipeline, "best_model")

    if save_plots:
        _init_plot_saving()

        lr_model, lr_Xtest = fitted_models["LR_interactions"]
        _save_fig(
            plot_residuals(y_test, lr_model.predict(lr_Xtest), "LR + interactions"),
            "residuals_lr",
        )

        gb_model, gb_Xtest = fitted_models["GradientBoosting_tuned"]
        _save_fig(
            plot_residuals(y_test, gb_model.predict(gb_Xtest), "Gradient Boosting"),
            "residuals_gb",
        )

        xgb_model, xgb_Xtest = fitted_models["XGBoost_tuned"]
        _save_fig(
            plot_residuals(y_test, xgb_model.predict(xgb_Xtest), "XGBoost"),
            "residuals_xgb",
        )

        lgbm_model, lgbm_Xtest = fitted_models["LightGBM_tuned"]
        _save_fig(
            plot_residuals(y_test, lgbm_model.predict(lgbm_Xtest), "LightGBM"),
            "residuals_lgbm",
        )

        _save_fig(
            plot_learning_curves(LinearRegression(), X_train_eng, y_train, "LR + interactions"),
            "learning_curve_lr",
        )
        _save_fig(
            plot_learning_curves(gb_model, X_train_proc, y_train, "Gradient Boosting"),
            "learning_curve_gb",
        )

        _save_fig(
            plot_feature_importance(imp_dfs["LR"], "LR + interactions"),
            "feature_importance_lr",
        )
        _save_fig(
            plot_feature_importance(imp_dfs["GB"], "Gradient Boosting"),
            "feature_importance_gb",
        )
        _save_fig(
            plot_feature_importance(imp_dfs["XGB"], "XGBoost"),
            "feature_importance_xgb",
        )
        _save_fig(
            plot_feature_importance(imp_dfs["LGBM"], "LightGBM"),
            "feature_importance_lgbm",
        )

        _save_fig(plot_model_comparison(results), "model_comparison")

        logger.info("plots.saved phase=diagnostics count=11 dir=%s", OUTPUT_DIR)

    return results


def main():
    parser = argparse.ArgumentParser(description="Health Insurance Cost Prediction")
    parser.add_argument("--model-only", action="store_true", help="Skip EDA")
    parser.add_argument("--save-plots", action="store_true", help="Save all plots to outputs/")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s level=%(levelname)-7s module=%(name)-24s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    df = load_data()

    if not args.model_only:
        run_eda(df, save_plots=args.save_plots)

    run_modeling(df, save_plots=args.save_plots)


if __name__ == "__main__":
    main()
