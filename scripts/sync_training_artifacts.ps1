param(
    [string]$LogsRoot = "logs/rsl_rl"
)

$ErrorActionPreference = "Stop"

$repoCandidate = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$repoRoot = (& git -C $repoCandidate rev-parse --show-toplevel).Trim()
if (-not $repoRoot) {
    throw "This script must be run inside a Git repository."
}

$logsPath = Join-Path $repoRoot $LogsRoot
if (-not (Test-Path -LiteralPath $logsPath -PathType Container)) {
    throw "Training logs directory not found: $logsPath"
}

# Stage every non-checkpoint artifact, including exported policies, configs,
# TensorBoard events, source diffs, and videos. model_*.pt stays ignored here.
& git -C $repoRoot add -- $LogsRoot
if ($LASTEXITCODE -ne 0) {
    throw "Failed to stage training artifacts."
}

$latestModels = @(Get-ChildItem -LiteralPath $logsPath -Recurse -File -Filter "model_*.pt" |
    ForEach-Object {
        if ($_.BaseName -match '^model_(\d+)$') {
            [PSCustomObject]@{
                Directory = $_.DirectoryName
                Iteration = [long]$Matches[1]
                FullName  = $_.FullName
            }
        }
    } |
    Group-Object Directory |
    ForEach-Object {
        $_.Group | Sort-Object Iteration -Descending | Select-Object -First 1
    })

foreach ($model in $latestModels) {
    $runRelative = $model.Directory.Substring($repoRoot.Length + 1).Replace("\", "/")
    $latestRelative = $model.FullName.Substring($repoRoot.Length + 1).Replace("\", "/")
    $trackedModels = @(& git -C $repoRoot ls-files -- "$runRelative/model_*.pt")

    foreach ($trackedModel in $trackedModels) {
        if ($trackedModel -ne $latestRelative) {
            & git -C $repoRoot rm --cached --quiet --ignore-unmatch -- $trackedModel
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to untrack intermediate checkpoint: $trackedModel"
            }
        }
    }

    & git -C $repoRoot add --force -- $model.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to stage latest checkpoint: $($model.FullName)"
    }
}

Write-Host "Staged training artifacts and $($latestModels.Count) latest checkpoints."
Write-Host "Review with: git status --short"
