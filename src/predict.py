"""翌日24時間の需要予測スクリプト。

前提:
  - data/processed/dataset.csv に「今日」までの実績需要が入っていること
    (最新化するには先に `python src/fetch_data.py` を再実行してください)
  - models/demand_model.joblib が学習済みであること (`python src/train.py`)

使い方:
    python src/predict.py                # 翌日を予測
    python src/predict.py --date 2026-08-10
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_calendar_features,
    add_weather_features,
)
from fetch_data import fetch_weather_forecast

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


def load_history() -> pd.DataFrame:
    path = PROCESSED_DIR / "dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} が見つかりません。先に `python src/fetch_data.py` を実行してください。"
        )
    return pd.read_csv(path, parse_dates=["datetime"])


def lookup_demand(history: pd.DataFrame, dt: pd.Timestamp) -> float | None:
    row = history.loc[history["datetime"] == dt, TARGET_COLUMN]
    return float(row.iloc[0]) if len(row) else None


def build_target_rows(target_date: pd.Timestamp, history: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    hours = pd.date_range(target_date, periods=24, freq="h")
    rows = pd.DataFrame({"datetime": hours})
    rows = add_calendar_features(rows)

    rows = rows.merge(weather, on="datetime", how="left")
    if rows["temperature"].isna().any():
        raise ValueError(
            "対象日の気象予報が取得できませんでした。Open-Meteoの forecast_days 範囲内の日付を指定してください。"
        )
    rows = add_weather_features(rows)

    rows["lag_24h"] = rows["datetime"].apply(lambda d: lookup_demand(history, d - pd.Timedelta(hours=24)))
    rows["lag_168h"] = rows["datetime"].apply(lambda d: lookup_demand(history, d - pd.Timedelta(hours=168)))
    rows["rolling_mean_3d"] = rows["datetime"].apply(
        lambda d: pd.Series(
            [
                lookup_demand(history, d - pd.Timedelta(hours=24)),
                lookup_demand(history, d - pd.Timedelta(hours=48)),
                lookup_demand(history, d - pd.Timedelta(hours=72)),
            ]
        ).mean()
    )

    missing = rows[FEATURE_COLUMNS].isna().any(axis=1)
    if missing.any():
        raise ValueError(
            "実績データが不足しているため、一部時刻のラグ特徴量を計算できませんでした。\n"
            "data/processed/dataset.csv を最新化(直近1週間以上のデータを含む)してください。\n"
            f"不足時刻: {rows.loc[missing, 'datetime'].tolist()}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="翌日24時間の需要予測")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="予測対象日 (YYYY-MM-DD)。省略時は翌日。",
    )
    args = parser.parse_args()

    if args.date:
        target_date = pd.Timestamp(args.date)
    else:
        # 実行サーバーのタイムゾーンに依存しないよう、日本時間で「明日」を決める
        # (例: GitHub Actions等UTCのCI環境で実行しても日付がずれない)
        now_jst = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None)
        target_date = now_jst.normalize() + pd.Timedelta(days=1)

    model_path = MODELS_DIR / "demand_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} が見つかりません。先に `python src/train.py` を実行してください。")
    model = joblib.load(model_path)

    history = load_history()
    weather = fetch_weather_forecast(days=3)

    rows = build_target_rows(target_date, history, weather)
    rows["predicted_demand_10000kW"] = model.predict(rows[FEATURE_COLUMNS])

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"forecast_{target_date.strftime('%Y%m%d')}.csv"
    result = rows[["datetime", "temperature", "predicted_demand_10000kW"]]
    result.to_csv(out_path, index=False, encoding="utf-8")

    print(f"=== {target_date.strftime('%Y-%m-%d')} の需要予測(万kW) ===")
    for _, r in result.iterrows():
        print(f"  {r['datetime'].strftime('%H:%M')}  {r['predicted_demand_10000kW']:.0f}  (気温 {r['temperature']:.1f}C)")
    print(f"\n[OK] {out_path} に保存しました")
    print(f"[ピーク] {result.loc[result['predicted_demand_10000kW'].idxmax(), 'datetime'].strftime('%H:%M')} 時点で "
          f"{result['predicted_demand_10000kW'].max():.0f} 万kW")


if __name__ == "__main__":
    main()
