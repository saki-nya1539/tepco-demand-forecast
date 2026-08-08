# 毎週実行: 直近14ヶ月分のデータをフル再取得し、モデルを再学習する。
# 季節変化・電源構成の変化に追従するため、日次予測とは別に定期的な再学習を行う。

$ErrorActionPreference = "Stop"
$python = "C:\Users\Saki TSUNODA\AppData\Local\Programs\Python\Python314\python.exe"
$srcDir = "C:\Users\Saki TSUNODA\demand_forecast_tepco\src"
$logDir = "C:\Users\Saki TSUNODA\demand_forecast_tepco\logs"
$logFile = Join-Path $logDir "weekly_retrain.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $srcDir

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"===== START $timestamp =====" | Out-File -FilePath $logFile -Append -Encoding utf8

try {
    & $python fetch_data.py --months-back 14 2>&1 | Out-File -FilePath $logFile -Append -Encoding utf8
    & $python train.py --test-days 30 2>&1 | Out-File -FilePath $logFile -Append -Encoding utf8

    "===== SUCCESS =====" | Out-File -FilePath $logFile -Append -Encoding utf8
}
catch {
    "===== ERROR: $_ =====" | Out-File -FilePath $logFile -Append -Encoding utf8
}
