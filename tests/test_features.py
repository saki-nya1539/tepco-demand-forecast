"""features.py の単体テスト。外部通信は行わず、合成データのみで検証する。"""
import numpy as np
import pandas as pd

from features import (
    FEATURE_COLUMNS,
    add_calendar_features,
    add_lag_features,
    add_weather_features,
    make_supervised,
)


def make_synthetic_df(n_hours=24 * 10):
    """1時間刻みの合成需要データを作る(需要=時刻依存のわかりやすい値)。"""
    start = pd.Timestamp("2026-01-05")  # 月曜始まり
    dt = pd.date_range(start, periods=n_hours, freq="h")
    demand = 1000 + dt.hour * 10  # 時刻だけに依存する値にしておくと検算しやすい
    temperature = 15 + np.sin(np.arange(n_hours))
    humidity = 60 + np.cos(np.arange(n_hours))
    return pd.DataFrame(
        {
            "datetime": dt,
            "demand_10000kW": demand,
            "temperature": temperature,
            "humidity": humidity,
        }
    )


def test_add_calendar_features_values():
    df = make_synthetic_df(n_hours=48)
    out = add_calendar_features(df)

    assert out.loc[0, "hour"] == 0
    assert out.loc[0, "dow"] == 0  # 2026-01-05 は月曜日
    assert out.loc[0, "month"] == 1
    assert out.loc[0, "is_weekend"] == 0

    # 土曜日(dow=5)を含む行がweekendフラグ1になっているか
    saturday_rows = out[out["dow"] == 5]
    if len(saturday_rows):
        assert (saturday_rows["is_weekend"] == 1).all()


def test_add_weather_features_squares_temperature():
    df = make_synthetic_df(n_hours=5)
    out = add_weather_features(df)
    assert np.allclose(out["temperature_sq"], out["temperature"] ** 2)


def test_add_lag_features_shift_correctness():
    df = make_synthetic_df(n_hours=24 * 10)
    out = add_lag_features(df)

    # 24時間前の値と一致するはず(先頭24行を除く)
    shifted = out["demand_10000kW"].shift(24)
    assert np.allclose(
        out["lag_24h"].dropna(), shifted.dropna(), equal_nan=True
    )

    # 168時間(7日)前の値と一致するはず
    shifted_week = out["demand_10000kW"].shift(168)
    assert np.allclose(
        out["lag_168h"].dropna(), shifted_week.dropna(), equal_nan=True
    )

    # 先頭24行はlag_24hがNaNのはず
    assert out["lag_24h"].iloc[:24].isna().all()


def test_make_supervised_drops_nan_rows_and_returns_expected_columns():
    df = make_synthetic_df(n_hours=24 * 10)
    X, y, feat = make_supervised(df)

    assert list(X.columns) == FEATURE_COLUMNS
    assert not X.isna().any().any()
    assert len(X) == len(y) == len(feat)
    # 168時間分のラグが必要なので、先頭168行は学習データから落ちているはず
    assert len(X) <= len(df) - 168


def test_make_supervised_raises_no_error_on_minimal_valid_data():
    # 169時間分あればちょうど1行だけ学習データが作れる
    df = make_synthetic_df(n_hours=169)
    X, _y, _feat = make_supervised(df)
    assert len(X) == 1
