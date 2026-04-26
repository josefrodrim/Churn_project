"""
Tests unitarios — feature engineering y validación de datos.

Cubre los casos borde que más afectan la calidad de las predicciones:
  - Usuario sin historial de logs
  - Membresía expirada hace mucho tiempo
  - Usuario en su primer mes
  - Validación de rangos y columnas binarias
  - FeatureRecord rechaza campos inválidos
"""

import numpy as np
import pandas as pd
import pytest

from src.api.schemas import FeatureRecord
from src.models.retrain_submit_14 import FEATURE_COLS_V5
from src.pipeline.compute_features import validate_features, BINARY_COLS


# ── FEATURE RECORD — validación Pydantic ─────────────────────────────────────

class TestFeatureRecordValidation:

    def test_valid_record_creates_ok(self, dummy_features):
        assert dummy_features.msno == "test_user_active"
        assert dummy_features.completion_ratio == 0.75

    def test_completion_ratio_above_one_rejected(self):
        from tests.conftest import BASE_FEATURES
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FeatureRecord(msno="x", **{**BASE_FEATURES, "completion_ratio": 1.5})

    def test_completion_ratio_below_zero_rejected(self):
        from tests.conftest import BASE_FEATURES
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FeatureRecord(msno="x", **{**BASE_FEATURES, "completion_ratio": -0.3})

    def test_invalid_age_rejected(self):
        from tests.conftest import BASE_FEATURES
        with pytest.raises(Exception):
            FeatureRecord(msno="x", **{**BASE_FEATURES, "age": 200.0})

    def test_invalid_gender_enc_rejected(self):
        from tests.conftest import BASE_FEATURES
        with pytest.raises(Exception):
            FeatureRecord(msno="x", **{**BASE_FEATURES, "gender_enc": 5})

    def test_extra_fields_rejected(self):
        from tests.conftest import BASE_FEATURES
        with pytest.raises(Exception):
            FeatureRecord(msno="x", **{**BASE_FEATURES, "unknown_field": 99})

    def test_all_48_feature_cols_present(self, dummy_features):
        record_keys = set(dummy_features.model_fields.keys()) - {"msno"}
        feature_set = set(FEATURE_COLS_V5)
        assert record_keys == feature_set, (
            f"Diferencia: {record_keys.symmetric_difference(feature_set)}"
        )


# ── EDGE CASES ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_no_logs_user_is_valid(self, edge_no_logs):
        assert edge_no_logs.has_log_record == 0
        assert edge_no_logs.n_days == 0.0
        assert edge_no_logs.days_since_last == 999.0

    def test_expired_user_flags(self, edge_expired):
        assert edge_expired.is_expired == 1
        assert edge_expired.cancel_at_expire == 1
        assert edge_expired.cancel_before_expire == 1
        assert edge_expired.days_until_expire == -30.0

    def test_first_month_user_no_renewals(self, edge_first_month):
        assert edge_first_month.n_renewals == 0.0
        assert edge_first_month.prev_churn == 0
        assert edge_first_month.tenure_days == 15.0


# ── VALIDATE_FEATURES ─────────────────────────────────────────────────────────

class TestValidateFeatures:

    def _make_df(self, overrides: dict = None) -> pd.DataFrame:
        from tests.conftest import BASE_FEATURES
        row = {k: BASE_FEATURES[k] for k in FEATURE_COLS_V5 if k in BASE_FEATURES}
        if overrides:
            row.update(overrides)
        return pd.DataFrame([row])[FEATURE_COLS_V5]

    def test_clean_row_passes(self):
        df = self._make_df()
        valid, invalid = validate_features(df)
        assert len(valid) == 1
        assert len(invalid) == 0

    def test_null_feature_is_invalid(self):
        df = self._make_df({"days_since_last": np.nan})
        valid, invalid = validate_features(df)
        assert len(valid) == 0
        assert len(invalid) == 1

    def test_age_out_of_range_is_invalid(self):
        df = self._make_df({"age": 200.0})
        valid, invalid = validate_features(df)
        assert len(invalid) == 1

    def test_binary_col_invalid_value(self):
        df = self._make_df({"is_expired": 2})
        valid, invalid = validate_features(df)
        assert len(invalid) == 1

    def test_multiple_rows_mixed(self):
        from tests.conftest import BASE_FEATURES as BF
        good_row = {k: BF[k] for k in FEATURE_COLS_V5 if k in BF}
        bad_row  = {**good_row, "age": -5.0}
        df = pd.DataFrame([good_row, bad_row])[FEATURE_COLS_V5]
        valid, invalid = validate_features(df)
        assert len(valid) == 1
        assert len(invalid) == 1

    def test_all_binary_cols_covered(self):
        for col in BINARY_COLS:
            assert col in FEATURE_COLS_V5, f"{col} no está en FEATURE_COLS_V5"
