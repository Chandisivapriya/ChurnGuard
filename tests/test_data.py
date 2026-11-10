import pytest

from churnguard.data import SchemaError, clean_features, prepare_training_data


def test_blank_total_charges_for_new_customers_is_zero(synthetic_df):
    cleaned = clean_features(synthetic_df)
    assert cleaned["TotalCharges"].notna().all()
    assert (cleaned.loc[synthetic_df["tenure"] == 0, "TotalCharges"] == 0).all()


def test_no_service_values_collapsed(synthetic_df):
    cleaned = clean_features(synthetic_df)
    assert "No phone service" not in set(cleaned["MultipleLines"])
    assert "No internet service" not in set(cleaned["OnlineSecurity"])


def test_target_is_binary(synthetic_df):
    _, y = prepare_training_data(synthetic_df)
    assert set(y.unique()) <= {0, 1}


def test_missing_column_raises(synthetic_df):
    with pytest.raises(SchemaError):
        clean_features(synthetic_df.drop(columns=["Contract"]))
