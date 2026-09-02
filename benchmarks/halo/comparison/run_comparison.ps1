# Run the reduce-and-broadcast vs sync_add comparison across topologies and sizes.
#
# PREREQUISITE: multi-rank launch needs a running MPI process manager. Either:
#   (a) the MS-MPI Launch Service (admin):
#         Start-Service MsMpiLaunchSvc
#   (b) the smpd daemon as a normal user (no admin):
#         & "C:\Program Files\Microsoft MPI\Bin\smpd.exe" -d   # keep this window open
# Then run:
#     powershell -ExecutionPolicy Bypass -File run_comparison.ps1
param(
    [int[]]$RingSizes = @(2, 4, 8, 16),
    [int[]]$GridSizes = @(4, 8, 16),
    [string]$RunId = (Get-Date -Format "yyyy-MM-dd-HHmmss"),
    [int]$Warmup = 3,
    [int]$Repeats = 10
)

$ErrorActionPreference = "Stop"
$py = "D:\repo\tribbie\.venv\Scripts\python.exe"
$bench = "D:\repo\tribbie\benchmarks\halo\comparison\compare_sync.py"
$outDir = "D:\repo\tribbie\reports\halo\comparison\$RunId"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# Payload targets (bytes).  The 100MB case is only run at small sizes (<=4 ranks)
# because the per-rank construction memory (gid/owner arrays + discovery) plus the
# payload does not fit 16 concurrent ranks on this host.
$smallPayloads = @(100000, 1000000, 10000000)
$largePayloads = @(100000, 1000000, 10000000, 100000000)

function Get-Params([string]$topology, [int[]]$payloads) {
    $entities = @(); $halos = @()
    foreach ($p in $payloads) {
        $N = [int]($p / 8)
        if ($topology -eq "ring") {
            $H = [math]::Max(1, [int]($N / 32))
            $entities += ($N - 2 * $H); $halos += $H
        } else {
            $n = [int][math]::Round([math]::Sqrt($N))
            $H = [math]::Max(1, [int]($n / 16))
            $entities += $n; $halos += $H
        }
    }
    return @{ entities = $entities; halos = $halos }
}

function Invoke-Bench([int]$size, [string]$topology, [string]$entitiesList, [string]$haloList) {
    $out = Join-Path $outDir "${topology}-size${size}.json"
    $cmdArgs = @("-n", "$size", $py, $bench,
                 "--topology", $topology,
                 "--entities-list", $entitiesList,
                 "--halo-list", $haloList,
                 "--warmup", "$Warmup", "--repeats", "$Repeats",
                 "--output", $out)
    Write-Host "== mpiexec -n $size ($topology) -> $out"
    & mpiexec @cmdArgs
    if ($LASTEXITCODE -ne 0) { throw "mpiexec failed for $topology size=$size" }
}

foreach ($s in $RingSizes) {
    $payloads = if ($s -le 4) { $largePayloads } else { $smallPayloads }
    $params = Get-Params "ring" $payloads
    Invoke-Bench $s "ring" ($params.entities -join ",") ($params.halos -join ",")
}
foreach ($s in $GridSizes) {
    $payloads = if ($s -le 4) { $largePayloads } else { $smallPayloads }
    $params = Get-Params "grid" $payloads
    Invoke-Bench $s "grid" ($params.entities -join ",") ($params.halos -join ",")
}

Write-Host "Done. Outputs in $outDir"
