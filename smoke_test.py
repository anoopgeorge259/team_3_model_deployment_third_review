"""Run this after copying the model/metadata files into this folder and before
pushing to GitHub / deploying. Confirms every required artifact is present and
that a prediction can actually be made end to end.

    python smoke_test.py
"""
import os
import sys

REQUIRED_FILES = [
    "kmeans_segmentation.joblib",
    "kmeans_scaler.joblib",
    "segment_names.json",
    "churn_gb_tuned.joblib",
    "churn_feature_cols.json",
    "repeat_purchase_svm.joblib",
    "manifest.json",
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(here, f))]
    if missing:
        print("Missing artifact(s):")
        for f in missing:
            print(" -", f)
        print(
            "\nRun Team3_Deployment_NextSteps.ipynb on the real dataset, then copy "
            "everything from its deployment_models/ output into this folder."
        )
        sys.exit(1)

    print("All required files present. Loading models...")
    from inference import OlistCustomerIntelligence

    engine = OlistCustomerIntelligence()

    print("\nSegment  :", engine.predict_segment(recency=25, frequency=3, monetary=450.0))
    print("Churn    :", engine.predict_churn_risk(
        frequency=2, monetary=300.0, avg_review_score=4.2,
        avg_delivery_days=9, on_time_rate=0.9, avg_payment_installments=2,
    ))
    print("Repeat   :", engine.predict_repeat_purchase(
        price=120.0, freight_value=18.0, payment_installments=3,
        payment_type="credit_card", product_category_name="bed_bath_table",
        customer_state="SP", on_time=True,
    ))
    print("\nSmoke test passed - safe to deploy.")


if __name__ == "__main__":
    main()
