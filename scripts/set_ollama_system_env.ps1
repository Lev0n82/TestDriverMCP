<#
Set the `OLLAMA_API_KEY` as a system-wide environment variable.

Usage (recommended - run as Administrator):
  .\set_ollama_system_env.ps1            # Prompts for the API key securely
  .\set_ollama_system_env.ps1 -ApiKey 'sk-...'

This script uses `setx -m` to persist the variable to the machine environment.
It must be run from an elevated PowerShell prompt (Administrator) to succeed.
#>

param(
    [string]$ApiKey
)

Set-StrictMode -Version Latest

function Write-Info($m){ Write-Host "[info] $m" -ForegroundColor Cyan }
function Write-Err($m){ Write-Host "[error] $m" -ForegroundColor Red }

if (-not $ApiKey) {
    # Prompt for secure input to avoid leaving the key in shell history
    $secure = Read-Host -AsSecureString "Enter OLLAMA_API_KEY (will not echo)"
    $ApiKey = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
}

if (-not $ApiKey) {
    Write-Err "No API key provided. Exiting."
    exit 2
}

try {
    Write-Info "Setting system environment variable 'OLLAMA_API_KEY' (machine scope)."
    # Use setx to persist the environment variable for the machine
    & setx OLLAMA_API_KEY "$ApiKey" -m | Out-Null
    Write-Info "Command executed. Note: new processes will see the variable after logoff/login or a new elevated shell."
    Write-Info "To verify (without printing the key), open a new PowerShell and run:`n  [Environment]::GetEnvironmentVariable('OLLAMA_API_KEY','Machine') -ne $null"
} catch {
    Write-Err "Failed to set system environment variable. Ensure you are running PowerShell as Administrator. Error: $_"
    exit 1
}
