"""
Prediction interface for the Insurance Cost model.

Usage:
    # Single prediction from CLI arguments
    python predict.py --age 35 --sex male --bmi 28.5 --children 2 --smoker yes --region northeast

    # Batch prediction from CSV
    python predict.py --csv input.csv --output predictions.csv
"""

import argparse
import logging
import sys

import joblib
import pandas as pd

from src.config import MODEL_DIR

logger = logging.getLogger(__name__)


def load_model():
    """Load the saved model pipeline."""
    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.exists():
        logger.error("model.missing path=%s", model_path)
        print("Run 'python main.py' first to train and save the model.")
        sys.exit(1)
    logger.info("model.loaded path=%s", model_path)
    return joblib.load(model_path)


def predict_single(
    age: int, sex: str, bmi: float, children: int, smoker: str, region: str
) -> float:
    """Predict insurance cost for a single person."""
    pipeline = load_model()
    df = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }
        ]
    )
    return pipeline.predict(df)[0]


def predict_batch(csv_path: str, output_path: str | None = None) -> pd.DataFrame:
    """Predict insurance costs for a CSV file."""
    pipeline = load_model()
    df = pd.read_csv(csv_path)

    required = {"age", "sex", "bmi", "children", "smoker", "region"}
    missing = required - set(df.columns)
    if missing:
        logger.error("csv.schema_error missing_columns=%s", missing)
        sys.exit(1)

    df["predicted_charges"] = pipeline.predict(df[list(required)])
    logger.info("predict.batch records=%d", len(df))

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info("predict.export path=%s records=%d", output_path, len(df))
    else:
        print(df.to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(description="Predict insurance costs")
    parser.add_argument("--csv", help="Path to input CSV for batch prediction")
    parser.add_argument("--output", help="Path to save predictions CSV")
    parser.add_argument("--age", type=int)
    parser.add_argument("--sex", choices=["male", "female"])
    parser.add_argument("--bmi", type=float)
    parser.add_argument("--children", type=int)
    parser.add_argument("--smoker", choices=["yes", "no"])
    parser.add_argument("--region", choices=["northeast", "northwest", "southeast", "southwest"])

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.csv:
        predict_batch(args.csv, args.output)
    elif all(
        v is not None
        for v in [args.age, args.sex, args.bmi, args.children, args.smoker, args.region]
    ):
        cost = predict_single(args.age, args.sex, args.bmi, args.children, args.smoker, args.region)
        print(f"\nPredicted insurance cost: ${cost:,.2f}")
        print(
            f"\nInput: age={args.age}, sex={args.sex}, bmi={args.bmi}, "
            f"children={args.children}, smoker={args.smoker}, region={args.region}"
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print(
            "  python predict.py --age 35 --sex male --bmi 28.5 "
            "--children 2 --smoker yes --region northeast"
        )
        print("  python predict.py --csv data/insurance.csv --output predictions.csv")


if __name__ == "__main__":
    main()
