# 毎日実行: 直近データを更新し、当日分の需要予測(24時間)を作成する。
#
# 実行タイミングについて:
#   このタスクは深夜〜早朝(既定 01:30)に実行する前提。
#   予測モデルは「前日・前週同時刻の実績需要」を特徴量に使うため、
#   対象日の"前日"が完全に終わっている必要がある。
#   深夜明け直後に実行すれば、前日分のデータは出そろっているため、
#   「今日」を対象日として予測できる(=前日夜に判明する翌日予報と同等の情報)。

$ErrorActionPreference = "Stop"
$python = "C:\Users\Saki TSUNODA\AppData\Local\Programs\Python\Python314\python.exe"
$srcDir = "C:\Users\Saki TSUNODA\demand_forecast_tepco\src"
$logDir = "C:\Users\Saki TSUNODA\demand_forecast_tepco\logs"
$logFile = Join-Path $logDir "daily_forecast.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $srcDir

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"===== START $timestamp =====" | Out-File -FilePath $logFile -Append -Encoding utf8

try {
    & $python fetch_data.py --months-back 2 2>&1 | Out-File -FilePath $logFile -Append -Encoding utf8

    $targetDate = Get-Date -Format "yyyy-MM-dd"
    & $python predict.py --date $targetDate 2>&1 | Out-File -FilePath $logFile -Append -Encoding utf8

    "===== SUCCESS =====" | Out-File -FilePath $logFile -Append -Encoding utf8
}
catch {
    "===== ERROR: $_ =====" | Out-File -FilePath $logFile -Append -Encoding utf8
}
