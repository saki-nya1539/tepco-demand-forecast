"""モデル学習スクリプト。

data/processed/dataset.csv (fetch_data.py の出力) を読み込み、
翌日24時間の需要(万kW)を予測する回帰モデルを学習・評価・保存する。

使い方:
    python src/train.py
    python src/train.py --test-days 30
"""
import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 環境によって使える日本語フォントが異なる(Windows: Meiryo, Linux CI: Noto Sans CJK JP)。
# 見つかったものを使い、どれも無ければmatplotlibの既定フォントにフォールバックする。
_JP_FONT_CANDIDATES = ["Meiryo", "Yu Gothic", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _JP_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

from features import FEATURE_COLUMNS, TARGET_COLUMN, make_supervised

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


def load_dataset() -> pd.DataFrame:
    path = PROCESSED_DIR / "dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} が見つかりません。先に `python src/fetch_data.py` を実行してください。"
        )
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df


def time_based_split(feat: pd.DataFrame, test_days: int):
    cutoff = feat["datetime"].max() - pd.Timedelta(days=test_days)
    train = feat[feat["datetime"] <= cutoff]
    test = feat[feat["datetime"] > cutoff]
    return train, test


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE(%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def plot_predictions(test: pd.DataFrame, y_pred: np.ndarray, out_path: Path, n_hours: int = 24 * 7):
    plot_df = test.tail(n_hours)
    pred_tail = y_pred[-len(plot_df):]
    plt.figure(figsize=(12, 4))
    plt.plot(plot_df["datetime"], plot_df[TARGET_COLUMN], label="実績")
    plt.plot(plot_df["datetime"], pred_tail, label="予測")
    plt.legend()
    plt.title("需要実績 vs 予測(テスト期間 直近1週間)")
    plt.ylabel("需要 (万kW)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[OK] {out_path} を保存しました")


def main():
    parser = argparse.ArgumentParser(description="需要予測モデルの学習")
    parser.add_argument("--test-days", type=int, default=30, help="末尾何日分をテストに使うか")
    args = parser.parse_args()

    df = load_dataset()
    X, _y, feat = make_supervised(df)
    feat = feat.loc[X.index]

    train, test = time_based_split(feat, args.test_days)
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    if len(X_train) < 168 or len(X_test) == 0:
        raise ValueError(
            "学習/テストデータが不足しています。fetch_data.py で複数年分のデータを取得してください。"
        )

    model = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.05,
        max_iter=500,
        early_stopping=True,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    print("=== テスト期間の精度 ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "demand_model.joblib"
    joblib.dump(model, model_path)
    print(f"[OK] モデルを {model_path} に保存しました")

    metrics_path = OUTPUTS_DIR / "metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"[OK] 精度指標を {metrics_path} に保存しました")

    plot_predictions(test, y_pred, OUTPUTS_DIR / "test_predictions.png")

    # 特徴量重要度(permutation importanceの簡易版として学習データでの寄与を確認したい場合は
    # sklearn.inspection.permutation_importance を別途利用してください)


if __name__ == "__main__":
    main()
