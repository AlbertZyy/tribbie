# Run the HaloPlan.exchange vs EntityMPI.sync comparison across MPI sizes.
#
# PREREQUISITE: multi-rank launch needs a running MPI process manager. Either:
#   (a) the MS-MPI Launch Service (admin):
#         Start-Service MsMpiLaunchSvc
#   (b) the smpd daemon as a normal user (no admin):
#         & "C:\Program Files\Microsoft MPI\Bin\smpd.exe" -d   # keep this window open
# Then run:
#     powershell -ExecutionPolicy Bypass -File run_comparison.ps1
param(
    [int[]]$Sizes = @(2, 4, 8, 16),
    [string]$RunId = (Get-Date -Format "yyyy-MM-dd-HHmmss"),
    [int]$Warmup = 5,
    [int]$Repeats = 20
)

$ErrorActionPreference = "Stop"
$py = "D:\repo\tribbie\.venv\Scripts\python.exe"
$bench = "D:\repo\tribbie\benchmarks\halo\comparison\compare_sync.py"
$outDir = "D:\repo\tribbie\reports\halo\comparison\$RunId"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function Invoke-Bench([int]$size, [string]$entitiesList, [string]$haloList, [string]$tag) {
    $out = Join-Path $outDir "size${size}-${tag}.json"
    $cmdArgs = @("-n", "$size", $py, $bench,
                 "--entities-list", $entitiesList,
                 "--components-list", "1",
                 "--warmup", "$Warmup", "--repeats", "$Repeats",
                 "--output", $out)
    if ($haloList) { $cmdArgs += @("--halo-list", $haloList) }
    Write-Host "== mpiexec -n $size ($tag) -> $out"
    & mpiexec @cmdArgs
    if ($LASTEXITCODE -ne 0) { throw "mpiexec failed for size=$size tag=$tag" }
}

# Weak scaling: constant per-rank work (E=1e5, H=1024) across all sizes.
foreach ($s in $Sizes) {
    Invoke-Bench $s "100000" "1024" "weak"
}

# Strong scaling: constant global problem (1e6 entities split across ranks).
foreach ($s in $Sizes) {
    $e = [int](1000000 / $s)
    $h = [int]([math]::Max(1, $e / 16))
    Invoke-Bench $s "$e" "$h" "strong"
}

# Data scale: fixed size=4, sweep per-rank entity count (H = E/16).
Invoke-Bench 4 "10000,100000,1000000" "" "data"

Write-Host "Done. Outputs in $outDir"
