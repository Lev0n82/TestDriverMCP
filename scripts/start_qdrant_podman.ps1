<#
Starts Qdrant using Podman (podman-compose or podman compose) or falls back to podman run.

Usage:
  .\start_qdrant_podman.ps1 [-ComposeFile <path>] [-VolumeName <name>] [-Port <port>] [-TimeoutSeconds <n>]

This script is safe to re-run; it will attempt to start an existing container if present.
#>

param(
    [string]$ComposeFile = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'podman-compose.yml'),
    [string]$VolumeName = 'qdrant_data',
    [int]$Port = 6333,
    [int]$TimeoutSeconds = 60
)

Set-StrictMode -Version Latest

function Write-Info($msg){ Write-Host "[info] $msg" -ForegroundColor Cyan }
function Write-Warn($msg){ Write-Host "[warn] $msg" -ForegroundColor Yellow }
function Write-Err($msg){ Write-Host "[error] $msg" -ForegroundColor Red }

Write-Info "Script started. Compose file: $ComposeFile"

# Helper to run a command and return $true on success
function Try-Run($exe, $args){
    try{
        $proc = Start-Process -FilePath $exe -ArgumentList $args -NoNewWindow -PassThru -Wait -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# 1) Try podman-compose
if (Try-Run 'podman-compose' '--version'){
    Write-Info 'podman-compose detected; using podman-compose to bring up services.'
    $cmd = "podman-compose -f `"$ComposeFile`" up -d"
    Write-Info "Running: $cmd"
    & podman-compose -f $ComposeFile up -d
}
elseif (Try-Run 'podman' 'compose version'){
    Write-Info 'podman compose detected; using `podman compose` to bring up services.'
    Write-Info "Running: podman compose -f '$ComposeFile' up -d"
    & podman compose -f $ComposeFile up -d
}
else{
    Write-Warn 'No podman-compose or `podman compose` detected. Falling back to direct podman commands.'

    # Ensure named volume exists
    Write-Info "Creating volume '$VolumeName' (idempotent)"
    & podman volume create $VolumeName | Out-Null

    # If a qdrant container exists, try to start/ensure it's running
    $existing = & podman ps -a --filter name=qdrant --format "{{.ID}} {{.Names}} {{.Status}}" 2>$null
    if ($existing){
        Write-Info 'Found existing qdrant container; attempting to start it.'
        & podman start qdrant
    } else {
        Write-Info 'Running new qdrant container.'
        & podman run -d --name qdrant -p ${Port}:6333 -v ${VolumeName}:/qdrant/storage qdrant/qdrant:latest
    }
}

# Wait for Qdrant HTTP endpoint
Write-Info "Waiting up to $TimeoutSeconds seconds for Qdrant to respond on http://localhost:$Port/collections"
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    try{
        $resp = Invoke-RestMethod -Uri "http://localhost:$Port/collections" -UseBasicParsing -TimeoutSec 5
        if ($null -ne $resp -and $resp.status -eq 'ok'){
            Write-Info 'Qdrant is healthy and responding.'
            exit 0
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

Write-Err "Timed out waiting for Qdrant to become healthy after $TimeoutSeconds seconds. Check container logs with: podman logs qdrant"
exit 2
