<#
.\secure_retrieve_and_apply_autologon.ps1

This script reads the encrypted password stored by `secure_store_password.ps1`,
decrypts it using DPAPI (machine scope), and optionally invokes Autologon.exe to
apply the autologon settings. This script must run as the SYSTEM account (or a
process that can read the registry key — the key is ACL'd for SYSTEM only).

Usage:
- Run this as a scheduled task configured to run as `NT AUTHORITY\SYSTEM`.
- Or run interactively as SYSTEM (PsExec -s) for testing.
#>

param(
    [switch]$ApplyWithAutologonExe
)

function Read-EncryptedCredential {
    $baseRegPath = 'SOFTWARE\\TestDriverMCP\\Credentials'
    $key = [Microsoft.Win32.Registry]::LocalMachine.OpenSubKey($baseRegPath)
    if (-not $key) { throw "Registry key HKLM:\\$baseRegPath not found" }

    $user = $key.GetValue('UserName')
    $domain = $key.GetValue('Domain')
    $b64 = $key.GetValue('EncryptedPassword')
    if (-not $b64) { throw 'EncryptedPassword value missing' }

    $protected = [Convert]::FromBase64String($b64)
    return @{ User=$user; Domain=$domain; ProtectedBytes=$protected }
}

try {
    $cred = Read-EncryptedCredential
} catch {
    Write-Error "Failed to read encrypted credential: $_"
    exit 1
}

# Decrypt using DPAPI machine scope
try {
    $bytes = [System.Security.Cryptography.ProtectedData]::Unprotect($cred.ProtectedBytes, $null, [System.Security.Cryptography.DataProtectionScope]::LocalMachine)
    $plain = [System.Text.Encoding]::UTF8.GetString($bytes)
} catch {
    Write-Error "Decryption failed: $_"
    exit 1
}

Write-Host "Decrypted credential for $($cred.User)@$($cred.Domain)"

if ($ApplyWithAutologonExe) {
    # Attempt to locate Autologon.exe
    $exe = Join-Path -Path $PSScriptRoot -ChildPath 'Autologon.exe'
    if (-not (Test-Path $exe)) {
        $cmd = Get-Command Autologon.exe -ErrorAction SilentlyContinue
        if ($cmd) { $exe = $cmd.Path }
    }

    if (-not (Test-Path $exe)) {
        Write-Error 'Autologon.exe not found in script folder or PATH. Cannot apply via Autologon.'
        exit 1
    }

    Write-Host "Invoking Autologon.exe to apply credentials: $exe"
    try {
        # Note: passing the password on the command-line is visible to other processes briefly.
        $args = @($cred.User, $cred.Domain, $plain, '/accepteula')
        Start-Process -FilePath $exe -ArgumentList $args -Wait
        Write-Host 'Autologon invoked.'
    } catch {
        Write-Error "Failed to run Autologon.exe: $_"
        exit 1
    }
} else {
    # Just output the plaintext — in real use you would pipe this to a secure consumer
    Write-Host 'Plaintext password available to this process (security-sensitive):'
    Write-Host $plain
}

# Zero out memory - best effort
[System.Array]::Clear($bytes, 0, $bytes.Length)
$plain = $null
