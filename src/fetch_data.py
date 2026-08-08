"""
データ取得スクリプト
1. 東京電力パワーグリッド「エリア需給実績データ」(月次CSV・30分間隔)をダウンロード
2. Open-Meteo API から東京(大手町)の気象データ(気温・湿度)を取得
3. 両者を時刻(1時間単位)で結合して data/processed/dataset.csv に保存

データソースについて:
  https://www.tepco.co.jp/forecast/html/area_jukyu-j.html の月別リンクから、
  "eria_jukyu_YYYYMM_03.csv" が月ごとにダウンロードできる。
  列: DATE, TIME, エリア需要(MW平均), 原子力, 火力(LNG)... 供給力の内訳。
  今回使うのは「エリア需要」列のみ(単位はMW。万kW換算は /10)。
  このデータは前日分までほぼ遅延なく更新されるため、翌日予測に必要な
  「直近の実績値」を得るのに適している(月次実績CSVより新しい)。

  この形式・URLは東京電力側の都合で変更される可能性があります。自動取得が失敗した場合は
  上記ページから手動でCSVをダウンロードし、data/raw/ に配置してから
  `python src/fetch_data.py --skip-download` を実行してください。
"""
import argparse
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# 東京(大手町)付近の座標。TEPCOエリアの気温代表点として利用。
TOKYO_LAT, TOKYO_LON = 35.6895, 139.6917

TEPCO_ERIA_URL_TEMPLATE = "https://www.tepco.co.jp/forecast/html/images/eria_jukyu_{yyyymm}_03.csv"


def download_tepco_month(year: int, month: int) -> Path | None:
    """指定年月のエリア需給実績CSVをダウンロードして data/raw に保存する。"""
    yyyymm = f"{year}{month:02d}"
    url = TEPCO_ERIA_URL_TEMPLATE.format(yyyymm=yyyymm)
    dest = RAW_DIR / f"eria_jukyu_{yyyymm}.csv"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[WARN] {year}年{month}月分のダウンロードに失敗しました: {e}")
        return None

    dest.write_bytes(resp.content)
    print(f"[OK] {dest} を保存しました ({len(resp.content):,} bytes)")
    return dest


def _find_table_start(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        if line.startswith("DATE,TIME"):
            return i
    return None


def load_eria_csv(path: Path) -> pd.DataFrame | None:
    """エリア需給実績CSVを読み込み、datetime / demand_10000kW の2列に正規化する。"""
    for encoding in ("utf-8", "cp932", "shift_jis"):
        try:
            lines = path.read_text(encoding=encoding).splitlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"[WARN] {path} を読み込めませんでした(エンコーディング不明)")
        return None

    start = _find_table_start(lines)
    if start is None:
        print(f"[WARN] {path} に実績テーブルが見つかりませんでした")
        return None

    df = pd.read_csv(pd.io.common.StringIO("\n".join(lines[start:])))
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if "DATE" in c.upper()), None)
    time_col = next((c for c in df.columns if "TIME" in c.upper()), None)
    demand_col = next((c for c in df.columns if "エリア需要" in c), None)
    if not (date_col and time_col and demand_col):
        print(f"[WARN] {path} の列名を認識できませんでした。列一覧: {list(df.columns)}")
        return None

    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(
        df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce"
    )
    demand_mw = pd.to_numeric(df[demand_col], errors="coerce")
    out["demand_10000kW"] = demand_mw / 10.0  # MW -> 万kW
    out = out.dropna(subset=["datetime", "demand_10000kW"])
    return out


def load_all_raw_csv() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("eria_jukyu_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"{RAW_DIR} に eria_jukyu_*.csv が見つかりません。先に download_tepco_month() を実行するか、"
            "手動でCSVを配置してください。"
        )
    frames = [load_eria_csv(f) for f in files]
    frames = [f for f in frames if f is not None and len(f)]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)

    # 30分間隔 -> 1時間ごとの平均値に集約(気象データと粒度を揃える)
    hourly = (
        df.set_index("datetime")["demand_10000kW"]
        .resample("h")
        .mean()
        .dropna()
        .reset_index()
    )
    return hourly


def fetch_weather_history(start_date: str, end_date: str) -> pd.DataFrame:
    """Open-Meteo Archive API から過去の実測気温・湿度を取得(キー不要)。"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": TOKYO_LAT,
        "longitude": TOKYO_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "Asia/Tokyo",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    js = resp.json()["hourly"]
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(js["time"]),
            "temperature": js["temperature_2m"],
            "humidity": js["relative_humidity_2m"],
        }
    )
    return df


def fetch_weather_forecast(days: int = 3, past_days: int = 2) -> pd.DataFrame:
    """Open-Meteo Forecast API から気温・湿度予報を取得(キー不要)。

    past_days>0 を指定すると直近の実測に近い値も遡って取得できるため、
    dataset.csv の更新が数日遅れていても当日分の気象データを補完しやすくなる。
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": TOKYO_LAT,
        "longitude": TOKYO_LON,
        "hourly": "temperature_2m,relative_humidity_2m",
        "forecast_days": days,
        "past_days": past_days,
        "timezone": "Asia/Tokyo",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    js = resp.json()["hourly"]
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(js["time"]),
            "temperature": js["temperature_2m"],
            "humidity": js["relative_humidity_2m"],
        }
    )
    return df


def build_dataset() -> pd.DataFrame:
    demand = load_all_raw_csv()
    start = demand["datetime"].min().strftime("%Y-%m-%d")
    end = demand["datetime"].max().strftime("%Y-%m-%d")
    print(f"[INFO] 需要データ期間: {start} 〜 {end} ({len(demand)}行)")

    weather = fetch_weather_history(start, end)
    df = pd.merge(demand, weather, on="datetime", how="left")
    df["temperature"] = df["temperature"].interpolate(limit_direction="both")
    df["humidity"] = df["humidity"].interpolate(limit_direction="both")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "dataset.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] {out_path} を保存しました ({len(df)}行)")
    return df


def month_range(months_back: int):
    end = pd.Timestamp.now().normalize().replace(day=1)
    months = pd.date_range(end=end, periods=months_back, freq="MS")
    return [(d.year, d.month) for d in months]


def main():
    parser = argparse.ArgumentParser(description="TEPCO需要データ + 気象データ取得")
    parser.add_argument(
        "--months-back",
        type=int,
        default=14,
        help="今月から遡って何ヶ月分の実績データを取得するか(既定14ヶ月=季節性1年分+直近トレンド)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="既にdata/rawにCSVがある場合はダウンロードをスキップして結合のみ行う",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        for year, month in month_range(args.months_back):
            download_tepco_month(year, month)

    build_dataset()


if __name__ == "__main__":
    main()
