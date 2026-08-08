"""特徴量エンジニアリング。

学習時(train.py)と予測時(predict.py)の両方から呼び出す共通ロジック。
1行 = 1時刻(年月日+時)の需要データを想定し、以下を特徴量として付与する:
  - カレンダー特徴量: 時, 曜日, 月, 週末フラグ, 祝日フラグ
  - 気象特徴量: 気温, 気温の2乗(冷房/暖房需要の非線形性を捉える), 湿度
  - ラグ特徴量: 24時間前(前日同時刻), 168時間前(前週同時刻)の実績需要
  - 移動平均: 直近3日間・同時刻の平均需要
"""
import pandas as pd

try:
    import jpholiday

    HAS_JPHOLIDAY = True
except ImportError:
    HAS_JPHOLIDAY = False

FEATURE_COLUMNS = [
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "temperature",
    "temperature_sq",
    "humidity",
    "lag_24h",
    "lag_168h",
    "rolling_mean_3d",
]
TARGET_COLUMN = "demand_10000kW"


def _is_holiday(dt: pd.Timestamp) -> bool:
    if HAS_JPHOLIDAY:
        return jpholiday.is_holiday(dt.date())
    return False  # jpholiday未インストール時は平日扱い(精度がやや落ちる)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = df["datetime"].apply(lambda d: int(_is_holiday(d)))
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temperature_sq"] = df["temperature"] ** 2
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """datetimeで昇順ソート済み・1時間刻みで欠損のない系列を前提とする。"""
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    df["lag_24h"] = df[target_col].shift(24)
    df["lag_168h"] = df[target_col].shift(168)
    # 直近3日間・同時刻(24h, 48h, 72h前)の平均
    df["rolling_mean_3d"] = (
        df[target_col].shift(24) + df[target_col].shift(48) + df[target_col].shift(72)
    ) / 3
    return df


def build_features(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """生データ(datetime, demand_10000kW, temperature, humidity)から特徴量を作成。"""
    df = add_calendar_features(df)
    df = add_weather_features(df)
    df = add_lag_features(df, target_col=target_col)
    return df


def make_supervised(df: pd.DataFrame, target_col: str = TARGET_COLUMN):
    """学習用に (X, y) を返す。ラグ特徴量がNaNになる先頭行は除外する。"""
    feat = build_features(df, target_col=target_col)
    feat = feat.dropna(subset=FEATURE_COLUMNS + [target_col])
    X = feat[FEATURE_COLUMNS]
    y = feat[target_col]
    return X, y, feat
