from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
try:
    import mlflow
    import mlflow.xgboost as mlflow_xgboost
except ModuleNotFoundError:
    mlflow = None
    mlflow_xgboost = None

try:
    import shap
except ModuleNotFoundError:
    shap = None
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {
    "Pickup Time",
    "Vehicle Type",
    "Payment Method",
    "Ride Distance (km)",
    "Booking Value (INR)",
    "Customer ID",
    "Booking Status",
}


def _risk_level(probability_cancelled: float) -> str:
    """Map cancellation probability to business risk buckets."""
    if probability_cancelled < 0.3:
        return "low"
    if probability_cancelled > 0.6:
        return "high"
    return "medium"


def _extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from pickup timestamp."""
    pickup = pd.to_datetime(df["Pickup Time"], errors="coerce")
    hour = pickup.dt.hour.fillna(0).astype(int)
    day = pickup.dt.dayofweek.fillna(0).astype(int)
    is_weekend = day.isin([5, 6]).astype(int)
    is_peak = (((hour >= 7) & (hour <= 10)) | ((hour >= 17) & (hour <= 21))).astype(int)

    return pd.DataFrame(
        {
            "hour_of_day": hour,
            "day_of_week": day,
            "is_weekend": is_weekend,
            "is_peak_hour": is_peak,
        },
        index=df.index,
    )


def _build_customer_experience(df: pd.DataFrame) -> pd.Series:
    """Compute customer ride history count from available records.

    The count is cumulative and represents prior rides before each booking.
    """
    work = df.copy()
    pickup = pd.to_datetime(work["Pickup Time"], errors="coerce")
    work = work.assign(_pickup_ts=pickup)
    work = work.sort_values(["Customer ID", "_pickup_ts"], kind="stable")
    experience = work.groupby("Customer ID").cumcount()
    experience = experience.reindex(df.index).fillna(0).astype(int)
    return experience


def _encode_ordinal(series: pd.Series, categories: list[str] | None = None) -> tuple[pd.Series, list[str]]:
    """Encode categorical values with sklearn OrdinalEncoder.

    Unknown categories are mapped to -1.
    """
    cleaned = series.fillna("unknown").astype(str).to_frame(name="value")

    if categories is None:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoded = encoder.fit_transform(cleaned).ravel().astype(int)
        learned_categories = [str(value) for value in encoder.categories_[0].tolist()]
        return pd.Series(encoded, index=series.index), learned_categories

    encoder = OrdinalEncoder(
        categories=[categories],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoded = encoder.fit_transform(cleaned).ravel().astype(int)
    return pd.Series(encoded, index=series.index), categories


def _build_features_internal(
    df: pd.DataFrame,
    vehicle_categories: list[str] | None = None,
    payment_categories: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build engineered features with deterministic category encoding."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns for cancellation features: {missing_str}")

    if df.empty:
        feature_columns = [
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_peak_hour",
            "ride_distance_km",
            "booking_value_inr",
            "vehicle_type_encoded",
            "payment_method_encoded",
            "customer_ride_count",
            "booking_label",
        ]
        return pd.DataFrame(columns=feature_columns), vehicle_categories or [], payment_categories or []

    time_features = _extract_time_features(df)

    ride_distance = pd.to_numeric(df["Ride Distance (km)"], errors="coerce")
    ride_distance = ride_distance.fillna(ride_distance.median() if not ride_distance.dropna().empty else 0.0)

    booking_value = pd.to_numeric(df["Booking Value (INR)"], errors="coerce")
    booking_value = booking_value.fillna(booking_value.median() if not booking_value.dropna().empty else 0.0)

    vehicle_encoded, vehicle_categories_out = _encode_ordinal(df["Vehicle Type"], vehicle_categories)
    payment_encoded, payment_categories_out = _encode_ordinal(df["Payment Method"], payment_categories)

    customer_ride_count = _build_customer_experience(df)

    booking_status = df["Booking Status"].fillna("").astype(str)
    label = booking_status.str.contains("Cancelled", case=False, na=False).astype(int)

    features = pd.DataFrame(
        {
            "hour_of_day": time_features["hour_of_day"],
            "day_of_week": time_features["day_of_week"],
            "is_weekend": time_features["is_weekend"],
            "is_peak_hour": time_features["is_peak_hour"],
            "ride_distance_km": ride_distance.astype(float),
            "booking_value_inr": booking_value.astype(float),
            "vehicle_type_encoded": vehicle_encoded.astype(int),
            "payment_method_encoded": payment_encoded.astype(int),
            "customer_ride_count": customer_ride_count.astype(int),
            "booking_label": label.astype(int),
        },
        index=df.index,
    )

    return features, vehicle_categories_out, payment_categories_out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model-ready cancellation features from raw booking data."""
    features, _, _ = _build_features_internal(df)
    return features


class CancellationPredictor:
    """Train, explain, persist, and serve a binary cancellation predictor."""

    def __init__(self, experiment_name: str = "cancellation_prediction") -> None:
        """Initialize predictor state and configure MLflow experiment."""
        self.experiment_name = experiment_name
        self.model: XGBClassifier | Any | None = None
        self.feature_names: list[str] = []
        self.threshold: float = 0.5
        self.vehicle_categories: list[str] = []
        self.payment_categories: list[str] = []
        self._fitted: bool = False

        if mlflow is not None:
            mlflow.set_experiment(experiment_name)
        else:
            logger.warning("MLflow is not installed. Training logs will be unavailable.")

    def train(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict[str, Any]:
        """Train XGBoost cancellation model and log metrics/artifacts to MLflow."""
        if XGBClassifier is None:
            raise ImportError("xgboost is required for training but is not installed in this environment.")

        features, vehicle_categories, payment_categories = _build_features_internal(df)
        if features.empty:
            raise ValueError("No rows available for training after feature engineering.")

        X = features.drop(columns=["booking_label"])
        y = features["booking_label"].astype(int)

        if y.nunique() < 2:
            raise ValueError("Training requires at least one cancelled and one completed sample.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

        params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "scale_pos_weight": scale_pos_weight,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)

        auc_roc = float(roc_auc_score(y_test, y_proba))
        avg_precision = float(average_precision_score(y_test, y_proba))
        f1 = float(f1_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        if mlflow is not None and mlflow_xgboost is not None:
            with mlflow.start_run(run_name="xgb_cancellation_v1"):
                mlflow.log_params(params)
                mlflow.log_metric("auc_roc", auc_roc)
                mlflow.log_metric("avg_precision", avg_precision)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("test_size", float(len(X_test)))
                mlflow_xgboost.log_model(model, "cancellation_xgb")

        self.model = model
        self.feature_names = list(X.columns)
        self.vehicle_categories = vehicle_categories
        self.payment_categories = payment_categories
        self._fitted = True

        return {
            "auc_roc": auc_roc,
            "avg_precision": avg_precision,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report_str": report,
        }

    def get_shap_values(self, X: pd.DataFrame) -> tuple[shap.Explainer, np.ndarray]:
        """Compute SHAP values for the provided feature matrix."""
        if not self._fitted or self.model is None:
            raise ValueError("Model is not trained. Call train() before SHAP analysis.")
        if shap is None:
            raise ImportError("shap is required for feature attributions but is not installed.")

        feature_frame = X[self.feature_names].copy() if self.feature_names else X.copy()
        explainer = shap.Explainer(self.model)
        shap_result = explainer(feature_frame)
        values = shap_result.values
        if isinstance(values, list):
            values = np.asarray(values)
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        return explainer, values

    def plot_shap_summary(self, X: pd.DataFrame) -> go.Figure:
        """Plot mean absolute SHAP value per feature as a horizontal bar chart."""
        _, shap_values = self.get_shap_values(X)
        feature_frame = X[self.feature_names].copy() if self.feature_names else X.copy()

        if shap_values.size == 0 or feature_frame.empty:
            fig = go.Figure()
            fig.update_layout(template="plotly_white", title="Feature importance (mean |SHAP|)")
            return fig

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        importance_df = pd.DataFrame(
            {
                "feature": feature_frame.columns,
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=True)

        colors = ["#9CA3AF"] * len(importance_df)
        if colors:
            colors[-1] = "coral"

        fig = go.Figure(
            go.Bar(
                x=importance_df["mean_abs_shap"],
                y=importance_df["feature"],
                orientation="h",
                marker=dict(color=colors),
            )
        )
        fig.update_layout(
            template="plotly_white",
            title="Feature importance (mean |SHAP|)",
            xaxis_title="mean |SHAP value|",
            yaxis_title="Feature",
        )
        return fig

    def plot_roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series) -> go.Figure:
        """Plot ROC curve with AUC annotation and random baseline."""
        if not self._fitted or self.model is None:
            raise ValueError("Model is not trained. Call train() before plotting ROC.")

        X_eval = X_test[self.feature_names].copy() if self.feature_names else X_test.copy()
        y_proba = self.model.predict_proba(X_eval)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_roc = roc_auc_score(y_test, y_proba)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                name="Random baseline",
            )
        )
        fig.update_layout(
            template="plotly_white",
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            annotations=[
                dict(
                    x=0.6,
                    y=0.1,
                    text=f"AUC: {auc_roc:.3f}",
                    showarrow=False,
                    bgcolor="white",
                )
            ],
        )
        return fig

    def _build_single_feature_row(self, feature_dict: dict[str, Any]) -> pd.DataFrame:
        """Transform one raw booking dict into model feature row."""
        required_raw = {
            "Pickup Time",
            "Vehicle Type",
            "Payment Method",
            "Ride Distance (km)",
            "Booking Value (INR)",
            "Customer ID",
            "Booking Status",
        }
        missing = [col for col in required_raw if col not in feature_dict]
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Missing raw fields for prediction: {missing_str}")

        row_df = pd.DataFrame([feature_dict])
        transformed, _, _ = _build_features_internal(
            row_df,
            vehicle_categories=self.vehicle_categories if self.vehicle_categories else None,
            payment_categories=self.payment_categories if self.payment_categories else None,
        )

        transformed = transformed.drop(columns=["booking_label"], errors="ignore")
        if "customer_ride_count" in feature_dict:
            transformed.loc[:, "customer_ride_count"] = int(feature_dict["customer_ride_count"])

        for col in self.feature_names:
            if col not in transformed.columns:
                transformed.loc[:, col] = 0.0

        return transformed[self.feature_names].copy()

    def predict_proba(self, feature_dict: dict[str, Any]) -> dict[str, Any]:
        """Predict cancellation probability and return risk explanation payload."""
        if not self._fitted or self.model is None:
            raise ValueError("Model is not trained. Call train() or load() before prediction.")

        X_one = self._build_single_feature_row(feature_dict)
        proba = np.asarray(self.model.predict_proba(X_one))
        classes = np.asarray(getattr(self.model, "classes_", [0, 1]))

        if proba.ndim != 2 or proba.shape[0] == 0:
            raise ValueError("Model returned invalid probability output.")

        if proba.shape[1] == 1:
            probability_cancelled = float(proba[0, 0]) if classes[0] == 1 else 0.0
        else:
            if 1 in classes:
                cancelled_index = int(np.where(classes == 1)[0][0])
                probability_cancelled = float(proba[0, cancelled_index])
            else:
                probability_cancelled = 0.0

        probability_cancelled = float(np.clip(probability_cancelled, 0.0, 1.0))
        probability_completed = float(1.0 - probability_cancelled)
        risk_level = _risk_level(probability_cancelled)

        top_risk_factors: list[str] = []
        try:
            _, shap_values = self.get_shap_values(X_one)
            if shap_values.ndim == 2 and shap_values.shape[0] > 0:
                abs_vals = np.abs(shap_values[0])
                top_idx = np.argsort(abs_vals)[::-1][:3]
                top_risk_factors = [self.feature_names[idx] for idx in top_idx]
        except Exception as exc:
            logger.warning("SHAP explanation unavailable for prediction payload: %s", exc)

        return {
            "probability_cancelled": probability_cancelled,
            "probability_completed": probability_completed,
            "risk_level": risk_level,
            "top_risk_factors": top_risk_factors,
        }

    def save(self, path: str) -> None:
        """Save model artifacts and metadata to a directory."""
        if not self._fitted or self.model is None:
            raise ValueError("Model is not trained. Nothing to save.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "model.joblib"
        metadata_path = save_dir / "metadata.json"

        joblib.dump(self.model, model_path)

        metadata = {
            "experiment_name": self.experiment_name,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "vehicle_categories": self.vehicle_categories,
            "payment_categories": self.payment_categories,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "CancellationPredictor":
        """Load a persisted predictor from disk."""
        load_dir = Path(path)
        model_path = load_dir / "model.joblib"
        metadata_path = load_dir / "metadata.json"

        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Expected model.joblib and metadata.json in the load path.")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        predictor = cls(experiment_name=metadata.get("experiment_name", "cancellation_prediction"))
        predictor.model = joblib.load(model_path)
        predictor.feature_names = list(metadata.get("feature_names", []))
        predictor.threshold = float(metadata.get("threshold", 0.5))
        predictor.vehicle_categories = list(metadata.get("vehicle_categories", []))
        predictor.payment_categories = list(metadata.get("payment_categories", []))
        predictor._fitted = True
        return predictor
