param(
  [int]$Epochs = 8,
  [switch]$SkipTrain,
  [switch]$SkipWeb
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$env:PYTHONPATH = $root
$python = "D:\python\python.exe"
$train = "data/real/idrid/processed/splits/train.csv"
$val = "data/real/idrid/processed/splits/val.csv"
$test = "data/real/idrid/processed/labels_test.csv"
$artifact = "artifacts/idrid_resnet_model.json"

if (-not $SkipTrain) {
  & $python -m src.train_resnet --train $train --val $val --artifact $artifact --checkpoint artifacts/idrid_resnet_best.pt --history artifacts/idrid_resnet_training_history.json --epochs $Epochs --batch-size 16 --learning-rate 1e-4 --weight-decay 1e-4 --device cuda
}
& $python -m src.evaluate --model $artifact --test $test --output artifacts/idrid_resnet_evaluation

if (-not $SkipWeb) {
  & $python web/manage.py migrate
  Write-Host "Start Django with: $python web/manage.py runserver 127.0.0.1:8000"
}
