param(
  [int]$SamplesPerClass = 20,
  [switch]$SkipWeb
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$env:PYTHONPATH = $root

python -m src.generate_demo_data --output-dir data/demo --samples-per-class $SamplesPerClass --seed 42
python -m src.check_data data/demo/labels.csv
python -m src.make_splits data/demo/labels.csv --output-dir data/demo/splits --seed 42
python -m src.train --train data/demo/splits/train.csv --val data/demo/splits/val.csv --test data/demo/splits/test.csv --artifact artifacts/model.json --epochs 5 --seed 42
python -m src.evaluate --model artifacts/model.json --test data/demo/splits/test.csv --output artifacts/evaluation
python -m src.predict --model artifacts/model.json --image data/demo/images/normal_0000.png

if (-not $SkipWeb) {
  python web/manage.py migrate
  Write-Host "Start Django with: python web/manage.py runserver 127.0.0.1:8000"
}
