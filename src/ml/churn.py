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
    import shap
except ModuleNotFoundError:
    shap = None
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    import mlflow
    import mlflow.xgboost as mlflow_xgboost
except ModuleNotFoundError:
    mlflow = None
    mlflow_xgboost = None

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Predict customer churn risk from ride-history aggregates."""

    VEHICLE_CATEGORIES = ["Auto", "Bike", "Sedan", "SUV", "Prime Sedan"]
    PAYMENT_CATEGORIES = ["Cash", "UPI", "Card", "Wallet"]

    def __init__(self, churn_days: int = 30, experiment_name: str = "churn_prediction") -> None:
        """Initialize churn predictor and configure MLflow experiment if available."""
        self.churn_days = int(churn_days)
        self.experiment_name = experiment_name
        self.model: XGBClassifier | Any | None = None
        self.feature_names: list[str] = []
        self.threshold: float = 0.5
        self._fitted = False
        self._scores_cache_key: tuple[int, str] | None = None
        self._scores_cache_base: pd.DataFrame | None = None
        self._scores_cache_X: pd.DataFrame | None = None

        if mlflow is not None:
            mlflow.set_experiment(experiment_name)
        else:
            logger.warning("MLflow is not installed. Training logs will be unavailable.")

    @staticmethod
    def _encode_mode(series: pd.Series, categories: list[str]) -> int:
        """Encode modal categorical value to fixed integer index."""
        cleaned = series.dropna().astype(str)
        if cleaned.empty:
            return 0
        mode_value = cleaned.mode().iloc[0]
        mapping = {value: idx for idx, value in enumerate(categories)}
        return int(mapping.get(mode_value, 0))

    def build_churn_features(
        self,
        df: pd.DataFrame,
        reference_date: pd.Timestamp | None = None,
        min_rides: int = 3,
    ) -> pd.DataFrame:
        """Aggregate customer-level churn features and labels."""
        required = {
            "Customer ID",
            "Pickup Time",
            "Booking Status",
            "Booking Value (INR)",
            "Customer Rating",
            "Vehicle Type",
            "Payment Method",
            "Ride Distance (km)",
        }
        missing = required.difference(df.columns)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns for churn features: {missing_str}")

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "Customer ID",
                    "days_since_last_ride",
                    "total_rides",
                    "rides_last_30d",
                    "rides_last_60d",
                    "avg_booking_value",
                    "avg_rating_given",
                    "preferred_vehicle",
                    "preferred_payment",
                    "completion_rate",
                    "avg_ride_distance",
                    "is_churned",
                ]
            )

        work = df.copy()
        work = work.assign(**{"Pickup Time": pd.to_datetime(work["Pickup Time"], errors="coerce")})
        work = work.dropna(subset=["Customer ID", "Pickup Time"]).copy()
        if work.empty:
            return pd.DataFrame(columns=[
                "Customer ID",
                "days_since_last_ride",
                "total_rides",
                "rides_last_30d",
                "rides_last_60d",
                "avg_booking_value",
                "avg_rating_given",
                "preferred_vehicle",
                "preferred_payment",
                "completion_rate",
                "avg_ride_distance",
                "is_churned",
            ])

        work.loc[:, "Booking Value (INR)"] = pd.to_numeric(work["Booking Value (INR)"], errors="coerce")
        work.loc[:, "Customer Rating"] = pd.to_numeric(work["Customer Rating"], errors="coerce")
        work.loc[:, "Ride Distance (km)"] = pd.to_numeric(work["Ride Distance (km)"], errors="coerce")
        work.loc[:, "is_completed"] = work["Booking Status"].astype(str).eq("Success").astype(int)

        ref_date = (
            pd.Timestamp(reference_date)
            if reference_date is not None
            else pd.to_datetime(work["Pickup Time"]).max().normalize() + pd.Timedelta(days=1)
        )

        rows: list[dict[str, Any]] = []
        for customer_id, grp in work.groupby("Customer ID"):
            total_rides = int(len(grp))
            if total_rides < int(min_rides):
                continue

            grp = grp.sort_values("Pickup Time")
            last_ride = pd.to_datetime(grp["Pickup Time"]).max()
            days_since_last_ride = int(max((ref_date - last_ride).days, 0))

            rides_last_30d = int((grp["Pickup Time"] >= (ref_date - pd.Timedelta(days=30))).sum())
            rides_last_60d = int((grp["Pickup Time"] >= (ref_date - pd.Timedelta(days=60))).sum())

            avg_booking_value = float(pd.to_numeric(grp["Booking Value (INR)"], errors="coerce").mean())
            avg_rating_given = float(pd.to_numeric(grp["Customer Rating"], errors="coerce").mean())
            completion_rate = float(grp["is_completed"].mean())
            avg_ride_distance = float(pd.to_numeric(grp["Ride Distance (km)"], errors="coerce").mean())

            preferred_vehicle = self._encode_mode(grp["Vehicle Type"], self.VEHICLE_CATEGORIES)
            preferred_payment = self._encode_mode(grp["Payment Method"], self.PAYMENT_CATEGORIES)

            rows.append(
                {
                    "Customer ID": customer_id,
                    "days_since_last_ride": days_since_last_ride,
                    "total_rides": total_rides,
                    "rides_last_30d": rides_last_30d,
                    "rides_last_60d": rides_last_60d,
                    "avg_booking_value": 0.0 if np.isnan(avg_booking_value) else avg_booking_value,
                    "avg_rating_given": 0.0 if np.isnan(avg_rating_given) else avg_rating_given,
                    "preferred_vehicle": preferred_vehicle,
                    "preferred_payment": preferred_payment,
                    "completion_rate": 0.0 if np.isnan(completion_rate) else completion_rate,
                    "avg_ride_distance": 0.0 if np.isnan(avg_ride_distance) else avg_ride_distance,
                    "is_churned": int(days_since_last_ride > self.churn_days),
                }
            )

        features = pd.DataFrame(rows)
        if features.empty:
            return pd.DataFrame(columns=[
                "Customer ID",
                "days_since_last_ride",
                "total_rides",
                "rides_last_30d",
                "rides_last_60d",
                "avg_booking_value",
                "avg_rating_given",
                "preferred_vehicle",
                "preferred_payment",
                "completion_rate",
                "avg_ride_distance",
                "is_churned",
            ])

        return features

    def train(self, df: pd.DataFrame) -> dict[str, Any]:
        """Train an XGBoost churn model and log experiment metrics."""
        features = self.build_churn_features(df, min_rides=1)
        if features.empty:
            raise ValueError("No eligible customers available for churn training.")

        X = features.drop(columns=["Customer ID", "is_churned"])
        y = features["is_churned"].astype(int)
        if y.nunique() < 2:
            raise ValueError("Training requires both churned and active classes.")

        class_counts = y.value_counts()
        min_class_count = int(class_counts.min()) if not class_counts.empty else 0
        stratify_target = y if min_class_count >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify_target,
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
            "random_state": 42,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)

        try:
            auc_roc = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            auc_roc = 0.0
        try:
            avg_precision = float(average_precision_score(y_test, y_proba))
        except ValueError:
            avg_precision = 0.0
        f1 = float(f1_score(y_test, y_pred))

        if mlflow is not None and mlflow_xgboost is not None:
            with mlflow.start_run(run_name="xgb_churn_v1"):
                mlflow.log_params(params)
                mlflow.log_metric("auc_roc", auc_roc)
                mlflow.log_metric("avg_precision", avg_precision)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("test_size", float(len(X_test)))
                mlflow_xgboost.log_model(model, "churn_xgb")

        self.model = model
        self.feature_names = list(X.columns)
        self._fitted = True

        return {
            "auc_roc": auc_roc,
            "avg_precision": avg_precision,
            "f1": f1,
            "n_churned": int(y.sum()),
            "n_active": int((y == 0).sum()),
        }

    def _get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for current model and feature matrix."""
        if self.model is None:
            raise ValueError("Model is not trained.")
        if shap is None:
            raise RuntimeError("SHAP is not installed in this runtime.")

        explainer = shap.Explainer(self.model)
        values = explainer(X).values
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        return values

    @staticmethod
    def _build_scores_cache_key(df: pd.DataFrame) -> tuple[int, str]:
        """Build a lightweight cache key for churn scoring inputs."""
        if df is None or df.empty:
            return (0, "")
        pickup = pd.to_datetime(df.get("Pickup Time"), errors="coerce")
        last_ts = pickup.max()
        last_ts_str = "" if pd.isna(last_ts) else str(last_ts)
        return (int(len(df)), last_ts_str)

    @staticmethod
    def _minmax(series: pd.Series) -> pd.Series:
        """Min-max normalize a numeric series to [0, 1] with safe zero-variance handling."""
        s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
        s_min = float(s.min()) if not s.empty else 0.0
        s_max = float(s.max()) if not s.empty else 0.0
        denom = s_max - s_min
        if denom <= 0:
            return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
        return (s - s_min) / denom

    def _heuristic_churn_probability(self, features: pd.DataFrame) -> pd.Series:
        """Compute fallback churn risk using business heuristics when model outputs collapse."""
        recency = self._minmax(features.get("days_since_last_ride", pd.Series(dtype=float)))
        low_freq = 1.0 - self._minmax(features.get("rides_last_60d", pd.Series(dtype=float)))
        low_completion = 1.0 - pd.to_numeric(
            features.get("completion_rate", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0)
        low_value = 1.0 - self._minmax(features.get("avg_booking_value", pd.Series(dtype=float)))

        # Weighted blend keeps scores in [0,1] and introduces useful spread.
        score = (0.50 * recency) + (0.25 * low_freq) + (0.20 * low_completion) + (0.05 * low_value)
        return pd.to_numeric(score, errors="coerce").fillna(0.0).clip(0.0, 1.0)

    def predict_churn_scores(self, df: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
        """Predict churn probability and per-customer top SHAP risk factors.

        If top_n is provided, SHAP explanations are computed only for top-risk rows.
        """
        if not self._fitted or self.model is None:
            raise ValueError("Model is not trained. Call train() or load() first.")

        cache_key = self._build_scores_cache_key(df)

        if self._scores_cache_key != cache_key or self._scores_cache_base is None or self._scores_cache_X is None:
            # Use >=2 rides for scoring to keep enough population while avoiding very sparse/noisy rows.
            features = self.build_churn_features(df, min_rides=2)
            if features.empty:
                return pd.DataFrame(columns=["Customer ID", "churn_probability", "churn_risk", "top_3_risk_factors"])

            X = features[self.feature_names].copy()
            y_proba = self.model.predict_proba(X)[:, 1]

            # Some trained artifacts can collapse to a near-constant probability for all rows.
            # In that case, fall back to feature-driven heuristic scoring for a meaningful ranking.
            y_proba_arr = np.asarray(y_proba, dtype=float)
            unique_probs = np.unique(np.round(y_proba_arr, 8))
            proba_std = float(np.nanstd(y_proba_arr)) if y_proba_arr.size else 0.0
            if unique_probs.size <= 1 or proba_std < 1e-4:
                logger.warning("Detected degenerate churn probabilities; using heuristic fallback scoring.")
                y_proba = self._heuristic_churn_probability(features).to_numpy(dtype=float)

            base_scores = pd.DataFrame(
                {
                    "Customer ID": features["Customer ID"].astype(str).values,
                    "churn_probability": y_proba,
                    "days_since_last_ride": features["days_since_last_ride"].values,
                    "total_rides": features["total_rides"].values,
                }
            )

            base_scores.loc[:, "churn_risk"] = np.where(
                base_scores["churn_probability"] > 0.6,
                "high",
                np.where(base_scores["churn_probability"] < 0.3, "low", "medium"),
            )

            base_scores = base_scores.sort_values("churn_probability", ascending=False).reset_index(drop=True)

            X_cache = X.copy()
            X_cache.index = features["Customer ID"].astype(str).values

            self._scores_cache_key = cache_key
            self._scores_cache_base = base_scores
            self._scores_cache_X = X_cache

        scores = self._scores_cache_base.copy()

        if top_n is not None:
            top_n = max(int(top_n), 1)
            scores = scores.head(top_n).reset_index(drop=True)

        # Align the feature matrix with selected customers for SHAP explanations.
        X_selected = self._scores_cache_X.reindex(scores["Customer ID"].astype(str).tolist()).copy()

        top_risk_factors: list[str] = [""] * len(scores)
        try:
            shap_values = self._get_shap_values(X_selected)
            for idx in range(len(scores)):
                abs_vals = np.abs(shap_values[idx])
                top_idx = np.argsort(abs_vals)[::-1][:3]
                factors = [self.feature_names[i] for i in top_idx]
                top_risk_factors[idx] = ", ".join(factors)
        except Exception as exc:
            logger.warning("Unable to compute SHAP risk factors: %s", exc)

        scores.loc[:, "top_3_risk_factors"] = top_risk_factors
        return scores

    def plot_churn_risk_distribution(self, scores_df: pd.DataFrame) -> go.Figure:
        """Plot churn risk histogram with low/medium/high risk background zones."""
        fig = go.Figure()

        if scores_df.empty or "churn_probability" not in scores_df.columns:
            fig.update_layout(template="plotly_white", title="Customer Churn Risk Distribution")
            return fig

        probabilities = pd.to_numeric(scores_df["churn_probability"], errors="coerce").fillna(0.0)

        fig.add_shape(type="rect", x0=0.0, x1=0.3, y0=0, y1=1, xref="x", yref="paper", fillcolor="rgba(0,200,0,0.12)", line_width=0)
        fig.add_shape(type="rect", x0=0.3, x1=0.6, y0=0, y1=1, xref="x", yref="paper", fillcolor="rgba(255,215,0,0.12)", line_width=0)
        fig.add_shape(type="rect", x0=0.6, x1=1.0, y0=0, y1=1, xref="x", yref="paper", fillcolor="rgba(255,0,0,0.12)", line_width=0)

        fig.add_trace(
            go.Histogram(x=probabilities, nbinsx=20, marker_color="steelblue", opacity=0.8, name="Customers")
        )

        fig.update_layout(
            template="plotly_white",
            title="Customer Churn Risk Distribution",
            xaxis_title="Churn Probability",
            yaxis_title="Customer Count",
            bargap=0.05,
        )
        return fig

    def plot_feature_importance(self, X: pd.DataFrame) -> go.Figure:
        """Plot mean absolute SHAP feature importance."""
        if not self._fitted or self.model is None:
            raise ValueError("Model is not trained. Call train() or load() first.")

        X_eval = X[self.feature_names].copy()

        try:
            shap_values = self._get_shap_values(X_eval)
            mean_abs = np.mean(np.abs(shap_values), axis=0)
        except Exception:
            mean_abs = np.zeros(X_eval.shape[1], dtype=float)

        imp = pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=True
        )
        colors = ["#9CA3AF"] * len(imp)
        if colors:
            colors[-1] = "coral"

        fig = go.Figure(
            go.Bar(
                x=imp["mean_abs_shap"],
                y=imp["feature"],
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

    def save(self, path: str) -> None:
        """Persist model and metadata to disk."""
        if not self._fitted or self.model is None:
            raise ValueError("Model is not trained. Nothing to save.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, save_dir / "model.joblib")
        metadata = {
            "churn_days": self.churn_days,
            "experiment_name": self.experiment_name,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
        }
        (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "ChurnPredictor":
        """Load predictor from persisted artifacts."""
        load_dir = Path(path)
        model_path = load_dir / "model.joblib"
        metadata_path = load_dir / "metadata.json"

        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Expected model.joblib and metadata.json in load path.")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        predictor = cls(
            churn_days=int(metadata.get("churn_days", 30)),
            experiment_name=metadata.get("experiment_name", "churn_prediction"),
        )
        predictor.model = joblib.load(model_path)
        predictor.feature_names = list(metadata.get("feature_names", []))
        predictor.threshold = float(metadata.get("threshold", 0.5))
        predictor._fitted = True
        return predictor
