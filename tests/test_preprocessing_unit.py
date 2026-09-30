"""Dataset-independent checks for the NIDS split/preprocessing contract."""

import pandas as pd

from ml.preprocessing.pipeline import NIDSPreprocessor, prepare_data


def sample_flows() -> pd.DataFrame:
    rows = []
    for label in ("BENIGN", "DoS Hulk", "PortScan", "Unknown"):
        for index in range(40):
            rows.append(
                {
                    "Label": label,
                    "Flow Duration": index + 1,
                    "Total Fwd Packets": index % 7,
                    "Protocol": "tcp" if index % 2 else "udp",
                    "Flow ID": f"{label}-{index}",
                    "Source IP": f"192.0.2.{index + 1}",
                    "Destination IP": f"198.51.100.{index + 1}",
                    "Timestamp": f"2026-01-{index % 28 + 1:02d} 00:00:00",
                    "SimillarHTTP": "metadata",
                }
            )
    return pd.DataFrame(rows)


def test_prepare_data_is_deterministic_stratified_and_holds_out_attack():
    frame = sample_flows()
    first = prepare_data(frame, holdout_attack="Unknown")
    second = prepare_data(frame, holdout_attack="Unknown")

    for actual, repeated in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(actual, repeated) if isinstance(actual, pd.DataFrame) else pd.testing.assert_series_equal(actual, repeated)

    x_train, x_val, x_test, y_train, y_val, y_test = first
    assert set(y_train.unique()) == {"BENIGN", "DoS Hulk", "PortScan"}
    assert set(y_val.unique()) == {"BENIGN", "DoS Hulk", "PortScan"}
    assert "Unknown" not in set(y_train) | set(y_val)
    assert (y_test == "Unknown").sum() == 40
    assert set(x_train.index).isdisjoint(x_val.index)
    assert set(x_train.index).isdisjoint(x_test.index)
    assert set(x_val.index).isdisjoint(x_test.index)


def test_preprocessor_drops_metadata_and_fits_only_on_training_data():
    x_train, x_val, x_test, *_ = prepare_data(sample_flows(), holdout_attack="Unknown")
    preprocessor = NIDSPreprocessor()
    train_features = preprocessor.fit_transform(x_train)
    validation_features = preprocessor.transform(x_val)
    test_features = preprocessor.transform(x_test)

    assert preprocessor.features == ["Flow Duration", "Total Fwd Packets", "Protocol"]
    assert train_features.shape[1] == validation_features.shape[1] == test_features.shape[1]
    assert validation_features.shape[0] == len(x_val)
    assert test_features.shape[0] == len(x_test)
