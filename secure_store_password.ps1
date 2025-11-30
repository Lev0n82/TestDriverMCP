<#
.\secure_store_password.ps1

Stores an account password encrypted with DPAPI (machine scope) and saves the
blob in the registry under HKLM:\SOFTWARE\TestDriverMCP\Credentials.

It then locks the registry key ACL so only the SYSTEM account can read it.

Usage: Run as Administrator.
#>

function Assert-RunAsAdministrator {
    $current = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($current)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Warning "This script must be run as Administrator. Attempting to relaunch with elevation..."
        Start-Process -FilePath pwsh -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
        exit
    }
}

Assert-RunAsAdministrator

Import-Module -Name Microsoft.PowerShell.Security -ErrorAction SilentlyContinue

$baseRegPath = 'SOFTWARE\\TestDriverMCP\\Credentials'
$fullRegPath = "HKLM:\\$baseRegPath"

Write-Host "Target registry path: HKLM:\\$baseRegPath"

# Prompt
$userName = Read-Host -Prompt 'Username to store for autologon (e.g. MYUSER)'
$domain = Read-Host -Prompt 'Domain/computer name (leave blank for local)'
if ([string]::IsNullOrWhiteSpace($domain)) { $domain = $env:COMPUTERNAME }
$secure = Read-Host -AsSecureString -Prompt 'Password (hidden)'

# Convert securestring to bytes
try {
    $bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    $plain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($plain)
} catch {
    Write-Error "Failed to convert SecureString: $_"
    exit 1
}

# Protect using DPAPI machine scope (encrypted blob only valid on this machine)
try {
    $protected = [System.Security.Cryptography.ProtectedData]::Protect($bytes, $null, [System.Security.Cryptography.DataProtectionScope]::LocalMachine)
    $b64 = [Convert]::ToBase64String($protected)
} catch {
    Write-Error "Encryption failed: $_"
    if ($bstr) { [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    exit 1
}

# Backup and create registry key with restricted ACL (SYSTEM only)
try {
    # Save a backup of any existing values
    if (Test-Path $fullRegPath) {
        $timestamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
        $backupFile = Join-Path -Path $env:USERPROFILE -ChildPath "winlogon-credentials-backup-$timestamp.json"
        Get-ItemProperty -Path $fullRegPath | Select-Object * | ConvertTo-Json | Out-File -FilePath $backupFile -Encoding utf8
        Write-Host "Backed up existing values to: $backupFile"
    }

    # Create registry key with a security descriptor that grants only SYSTEM read
    $reg = [Microsoft.Win32.Registry]::LocalMachine
    # Prepare security: allow only NT AUTHORITY\\SYSTEM full control
    $rs = New-Object System.Security.AccessControl.RegistrySecurity
    $sid = New-Object System.Security.Principal.SecurityIdentifier([System.Security.Principal.WellKnownSidType]::LocalSystemSid, $null)
    $account = $sid.Translate([System.Security.Principal.NTAccount]).Value
    $rule = New-Object System.Security.AccessControl.RegistryAccessRule($account, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
    $rs.SetAccessRule($rule)

    # Create the key with the security
    $created = [Microsoft.Win32.Registry]::LocalMachine.CreateSubKey($baseRegPath, [Microsoft.Win32.RegistryKeyPermissionCheck]::ReadWriteSubTree, $rs)
    if (-not $created) {
        throw 'Failed to create registry key with restricted ACL.'
    }

    # Write values
    $created.SetValue('UserName', $userName, [Microsoft.Win32.RegistryValueKind]::String)
    $created.SetValue('Domain', $domain, [Microsoft.Win32.RegistryValueKind]::String)
    $created.SetValue('EncryptedPassword', $b64, [Microsoft.Win32.RegistryValueKind]::String)
    $created.SetValue('CreatedAt', (Get-Date).ToString('o'), [Microsoft.Win32.RegistryValueKind]::String)

    Write-Host "Stored encrypted credential under HKLM:\\$baseRegPath"
    Write-Host "Registry ACL configured to allow only the SYSTEM account (others cannot read)."

} catch {
    Write-Error "Failed to store encrypted credential: $_"
    if ($bstr) { [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    exit 1
}

# Zero out memory
if ($bstr) { [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
$plain = $null
Write-Host "Done. To decrypt, use a process running as SYSTEM (e.g. scheduled task running as NT AUTHORITY\\SYSTEM)."

Write-Host "Important: This provides machine-bound encryption but ACLs and file/registry permissions can be changed by administrators."
