$ErrorActionPreference = 'Stop'
$outDir = Join-Path $PSScriptRoot 'data\real\idrid'
$url = 'https://zenodo.org/records/17219542/files/B.%20Disease%20Grading.zip?download=1'
$total = 212405123
$parts = 8
$partDir = Join-Path $outDir '_parts'
New-Item -ItemType Directory -Force -Path $partDir | Out-Null
$jobs = @()
for ($i = 0; $i -lt $parts; $i++) {
    $start = [int64][math]::Floor($i * $total / $parts)
    $end = [int64][math]::Floor(($i + 1) * $total / $parts) - 1
    $part = Join-Path $partDir ('part_{0:D2}.bin' -f $i)
    if ((Test-Path $part) -and ((Get-Item $part).Length -eq ($end - $start + 1))) { continue }
    $range = "bytes=$start-$end"
    $jobs += Start-Job -ScriptBlock {
        param($u, $r, $p)
        & curl.exe -L --fail --retry 5 --retry-delay 2 --max-time 3600 -H "Range: $r" -o $p $u
        if ($LASTEXITCODE -ne 0) { throw "curl failed: $LASTEXITCODE" }
    } -ArgumentList $url, $range, $part
}
if ($jobs.Count -gt 0) { $jobs | Wait-Job | Receive-Job; $jobs | Remove-Job }
$target = Join-Path $outDir 'B. Disease Grading.zip'
$stream = [System.IO.File]::Open($target, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
try {
    for ($i = 0; $i -lt $parts; $i++) {
        $part = Join-Path $partDir ('part_{0:D2}.bin' -f $i)
        $expected = [int64][math]::Floor(($i + 1) * $total / $parts) - [int64][math]::Floor($i * $total / $parts)
        if (!(Test-Path $part) -or (Get-Item $part).Length -ne $expected) { throw "Missing or incomplete part $i" }
        $bytes = [System.IO.File]::ReadAllBytes($part)
        $stream.Write($bytes, 0, $bytes.Length)
    }
} finally { $stream.Dispose() }
$hash = (Get-FileHash -Algorithm MD5 $target).Hash.ToLowerInvariant()
if ($hash -ne 'b9239a4b956021a1cf0225522f11f58f') { throw "MD5 mismatch: $hash" }
Write-Host "Downloaded and verified: $target"
