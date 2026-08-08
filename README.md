# 電力需要予測モデル(TEPCOエリア・翌日24時間)

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
[![Daily Forecast](https://github.com/OWNER/REPO/actions/workflows/forecast.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/forecast.yml)
[![Weekly Retrain](https://github.com/OWNER/REPO/actions/workflows/retrain.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/retrain.yml)

東京電力パワーグリッド(TEPCO)エリアの電力需要を、翌日の時間別(24時間分)で予測するモデルです。

## 仕組み

1. **データ取得** (`src/fetch_data.py`)
   - TEPCOが公開する「エリア需給実績データ」(30分間隔・月次CSV)から実績需要を取得
   - [Open-Meteo](https://open-meteo.com/) API から東京の気温・湿度を取得(APIキー不要)
   - 両者を1時間単位で結合し `data/processed/dataset.csv` を作成
2. **学習** (`src/train.py`)
   - 時刻・曜日・月・週末/祝日フラグ、気温(および気温の2乗)、湿度、
     前日同時刻・前週同時刻の実績需要などを特徴量として学習
   - scikit-learn の `HistGradientBoostingRegressor` で回帰
   - 直近30日をテスト期間として精度(MAE/RMSE/MAPE)を評価
3. **予測** (`src/predict.py`)
   - 学習済みモデルと、翌日の気象予報・直近の実績需要(ラグ特徴量用)から
     翌日24時間分の需要を予測し `outputs/forecast_YYYYMMDD.csv` に出力

## セットアップ

```powershell
cd demand_forecast_tepco
pip install -r requirements.txt
```

## 使い方

```powershell
# 1. データ取得(初回は直近14ヶ月分。以降は毎日/毎週再実行して最新化する)
python src/fetch_data.py

# 2. モデル学習(直近30日をテストに)
python src/train.py

# 3. 翌日の需要予測(24時間分)
python src/predict.py
# 特定の日を指定する場合
python src/predict.py --date 2026-08-10
```

出力:
- `outputs/metrics.csv` : テスト期間の精度指標
- `outputs/test_predictions.png` : 実績 vs 予測の比較グラフ(直近1週間)
- `outputs/forecast_YYYYMMDD.csv` : 翌日24時間分の予測結果
- `models/demand_model.joblib` : 学習済みモデル

## 実行時の精度目安(参考)

直近1年強のデータで学習した場合、テスト期間(直近30日)で **MAPE 約3%** 程度でした
(季節・年によって変動します)。実運用に入れる前に、対象期間・季節を変えて
バックテストし、業務要件に対して十分な精度か必ず確認してください。

## CI/CD (GitHub Actions)

`.github/workflows/` に3つのワークフローを用意しています。

| ワークフロー | トリガー | 内容 |
|---|---|---|
| `ci.yml` | push / PR (mainブランチ) | `ruff` でlint、`pytest` で単体テスト(`tests/test_features.py`)を実行。外部API通信は行わないので毎回安定して動く |
| `forecast.yml` | 毎日 16:30 UTC(01:30 JST)+ 手動実行 | 直近データ取得 → 当日分の需要予測を実行し、`outputs/forecast_*.csv` をコミット・push |
| `retrain.yml` | 毎週日曜 18:00 UTC(月曜03:00 JST)+ 手動実行 | 直近14ヶ月分を再取得 → モデル再学習し、`models/demand_model.joblib` と精度指標をコミット・push |

- `forecast.yml` / `retrain.yml` が `git push` するため、リポジトリの
  Settings → Actions → General → Workflow permissions を
  **"Read and write permissions"** にしておく必要があります(未設定だとpushが失敗します)。
- 手動実行はGitHubの Actions タブから対象ワークフローを選び「Run workflow」で可能です。
- ローカルのタスクスケジューラ(`scripts/run_daily_forecast.ps1` など)と役割が重複するため、
  GitHub Actions側で運用する場合はローカルのタスクは無効化/削除しても構いません。

## 運用上の注意

- **予測を実行するタイミング**: `predict.py` は「前日・前週同時刻の実績需要」を
  特徴量に使うため、対象日の前日分の実績データが `dataset.csv` に含まれている
  必要があります。TEPCOのエリア需給実績データは当日〜前日分がほぼ遅延なく
  更新されるので、予測を実行する直前に `python src/fetch_data.py` を
  再実行してデータを最新化してください。深夜0時台など当日データがまだ
  存在しない時間帯に実行すると、ラグ特徴量が計算できずエラーになります。
- **TEPCOのデータ形式・URLは変更される可能性があります**。自動ダウンロードが
  失敗した場合は `src/fetch_data.py` 冒頭のコメントに記載した手動ダウンロード
  手順に従ってください。
- **祝日判定**には `jpholiday` パッケージを使用しています(`pip install jpholiday`)。
  未インストールの場合は祝日を平日として扱うため、精度がやや落ちます。
- 実務投入する場合は、定期実行(タスクスケジューラ等で `fetch_data.py` →
  `predict.py` を毎日自動実行)や、モデルの定期再学習(季節変化・電源構成の
  変化に追従するため月次程度での再学習を推奨)を検討してください。

## ディレクトリ構成

```
demand_forecast_tepco/
  .github/workflows/
    ci.yml          # lint + 単体テスト
    forecast.yml     # 日次: データ更新 + 当日予測
    retrain.yml      # 週次: 全データ再取得 + モデル再学習
  data/
    raw/            # TEPCOからダウンロードした生CSV(git管理外)
    processed/       # 結合済みデータセット dataset.csv(git管理外)
  src/
    fetch_data.py    # データ取得・結合
    features.py      # 特徴量エンジニアリング(train/predict共通)
    train.py         # 学習・評価
    predict.py       # 翌日予測
  tests/
    test_features.py # features.pyの単体テスト(外部通信なし)
  scripts/            # ローカル(Windowsタスクスケジューラ)用の実行スクリプト
  models/             # 学習済みモデル
  outputs/            # 精度指標・グラフ・予測結果
```
