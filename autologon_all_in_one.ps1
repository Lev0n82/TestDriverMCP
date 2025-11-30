<#
autologon_all_in_one.ps1

Single-script solution to securely store credentials (machine-bound DPAPI),
and apply AutoLogon as SYSTEM. The script has two modes:

- Default (store mode): run the script as Administrator and provide a
  PSCredential via `-Credential` (or it will prompt). The script will:
    1. Encrypt the credential using DPAPI (LocalMachine scope).
    2. Store the encrypted blob in HKLM:\SOFTWARE\TestDriverMCP\Credentials
       and set the registry ACL so only NT AUTHORITY\SYSTEM has access.
    3. Create a scheduled task that runs this same script as SYSTEM with
       `-Apply` to decrypt and apply the autologon settings shortly after.

- Apply mode (`-Apply`): intended to be run as SYSTEM (scheduled task). It
  will decrypt the stored blob and invoke Autologon.exe to apply the
  credentials (or optionally write to Winlogon registry as a fallback).

Usage examples:
  # Interactive prompt and create scheduled task to apply as SYSTEM
  .\autologon_all_in_one.ps1

  # Provide credential non-interactively (recommended in scripts):
  $cred = Get-Credential
  .\autologon_all_in_one.ps1 -Credential $cred

  # Apply phase (this is created automatically by the script; run as SYSTEM):
  .\autologon_all_in_one.ps1 -Apply -Force

Security notes:
- The encrypted blob is machine-bound via DPAPI LocalMachine scope.
- Registry ACL restricts read access to SYSTEM only; administrators can
  still change ACLs and access the blob if they have physical or admin access.
- Passing plaintext credentials to commands is avoided in the store phase.
  The apply phase runs as SYSTEM and accesses the protected blob directly.
#>

param(
    [Parameter(Mandatory=$false)]
    [System.Management.Automation.PSCredential]$Credential,

    [switch]$Apply,

    [string]$AutologonPath,

    [string]$TaskName = 'ApplySecureAutologon',

    [switch]$Force,

    [switch]$RunAtBoot,
    [switch]$RunImmediate
)

function Assert-RunAsAdministrator {
    $current = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($current)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Warning 'This script must be run as Administrator for the store phase. Relaunching with elevation...'
        Start-Process -FilePath pwsh -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
        exit
    }
}

function Create-RegistryWithSystemOnlyAcl($baseRegPath) {
    try {
        $rs = New-Object System.Security.AccessControl.RegistrySecurity
        $sid = New-Object System.Security.Principal.SecurityIdentifier([System.Security.Principal.WellKnownSidType]::LocalSystemSid, $null)
        $account = $sid.Translate([System.Security.Principal.NTAccount]).Value
        $rule = New-Object System.Security.AccessControl.RegistryAccessRule($account, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
        $rs.SetAccessRule($rule)

        $created = [Microsoft.Win32.Registry]::LocalMachine.CreateSubKey($baseRegPath, [Microsoft.Win32.RegistryKeyPermissionCheck]::ReadWriteSubTree, $rs)
        return $created
    } catch {
        throw "Failed to create registry key with SYSTEM-only ACL: $_"
    }
}

function Store-CredentialMachineBound($cred, $baseRegPath) {
    try {
        $bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($cred.Password)
        $plain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($plain)
    } catch {
        throw "Failed to convert SecureString: $_"
    }

    try {
        $protected = [System.Security.Cryptography.ProtectedData]::Protect($bytes, $null, [System.Security.Cryptography.DataProtectionScope]::LocalMachine)
        $b64 = [Convert]::ToBase64String($protected)
    } catch {
        throw "Encryption failed: $_"
    } finally {
        if ($bstr) { [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
        $plain = $null
    }

    # Backup existing key if present
    $fullRegPath = "HKLM:\\$baseRegPath"
    if (Test-Path $fullRegPath) {
        $timestamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
        $backupFile = Join-Path -Path $env:USERPROFILE -ChildPath "autologon-credentials-backup-$timestamp.json"
        Get-ItemProperty -Path $fullRegPath | Select-Object * | ConvertTo-Json | Out-File -FilePath $backupFile -Encoding utf8
        Write-Host "Backed up existing credentials to: $backupFile"
    }

    $regKey = Create-RegistryWithSystemOnlyAcl $baseRegPath
    $regKey.SetValue('UserName', $cred.UserName, [Microsoft.Win32.RegistryValueKind]::String)
    $regKey.SetValue('Domain', $env:COMPUTERNAME, [Microsoft.Win32.RegistryValueKind]::String)
    $regKey.SetValue('EncryptedPassword', $b64, [Microsoft.Win32.RegistryValueKind]::String)
    $regKey.SetValue('CreatedAt', (Get-Date).ToString('o'), [Microsoft.Win32.RegistryValueKind]::String)
    $regKey.Close()

    Write-Host "Stored encrypted credential under HKLM:\$baseRegPath (SYSTEM-only ACL)."
}

function Ensure-AutologonPresent($destFolder) {
    $exeName = 'Autologon.exe'
    $candidate = Join-Path -Path $destFolder -ChildPath $exeName
    if (Test-Path $candidate) { return $candidate }
    $cmd = Get-Command $exeName -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Path }

    # Download and extract
    $url = 'https://download.sysinternals.com/files/Autologon.zip'
    $zipPath = Join-Path -Path $env:TEMP -ChildPath 'Autologon.zip'
    Write-Host "Downloading Autologon from $url ..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing -ErrorAction Stop
        Expand-Archive -Path $zipPath -DestinationPath $destFolder -Force
    } catch {
        Write-Warning "Failed to download/extract Autologon: $_"
        return $null
    }

    if (Test-Path $candidate) { return $candidate }
    return $null
}

function Create-RunAsSystemTask($scriptPath, $taskName, [bool]$runAtBoot, [bool]$runImmediate) {
    # Create scheduled tasks according to requested options.
    $action = "powershell -NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -Apply -TaskName `"$taskName`" -Force"

    if ($runImmediate) {
        $onceName = $taskName + '_Immediate'
        $startTime = (Get-Date).AddMinutes(1).ToString('HH:mm')
        $createCmd = "schtasks /Create /TN `"$onceName`" /TR `"$action`" /SC ONCE /ST $startTime /RL HIGHEST /RU SYSTEM /F"
        Write-Host "Creating scheduled task to run as SYSTEM once at $startTime (task: $onceName)"
        cmd.exe /c $createCmd | Out-Null
        Start-Sleep -Seconds 1
        Write-Host "Running scheduled task $onceName now..."
        cmd.exe /c "schtasks /Run /TN `"$onceName`"" | Out-Null
    }

    if ($runAtBoot) {
        $bootName = $taskName + '_OnBoot'
        $createCmdBoot = "schtasks /Create /TN `"$bootName`" /TR `"$action`" /SC ONSTART /RL HIGHEST /RU SYSTEM /F"
        Write-Host "Creating scheduled task to run as SYSTEM at system start (task: $bootName)"
        cmd.exe /c $createCmdBoot | Out-Null
    }
}

function Apply-AsSystem($baseRegPath, $autologonExePath, $taskName) {
    # Read registry and decrypt
    $fullRegPath = "HKLM:\\$baseRegPath"
    $key = [Microsoft.Win32.Registry]::LocalMachine.OpenSubKey($baseRegPath)
    if (-not $key) { throw 'Credential registry key not found' }
    $user = $key.GetValue('UserName')
    $domain = $key.GetValue('Domain')
    $b64 = $key.GetValue('EncryptedPassword')
    if (-not $b64) { throw 'EncryptedPassword missing in registry' }
    $protected = [Convert]::FromBase64String($b64)

    try {
        $bytes = [System.Security.Cryptography.ProtectedData]::Unprotect($protected, $null, [System.Security.Cryptography.DataProtectionScope]::LocalMachine)
        $plain = [System.Text.Encoding]::UTF8.GetString($bytes)
    } catch {
        throw "Decryption failed: $_"
    }

    Write-Host "Decrypted credential for $user (applying autologon)..."

    # Ensure Autologon.exe
    if (-not $autologonExePath) { $autologonExePath = Ensure-AutologonPresent((Split-Path -Parent $PSCommandPath)) }
    if (-not $autologonExePath) { Write-Warning 'Autologon.exe not found; will fall back to writing Winlogon registry (plaintext)'; }

    if ($autologonExePath) {
        # Run Autologon.exe with arguments (note: password briefly visible to processes)
        Write-Host "Invoking Autologon.exe: $autologonExePath"
        Start-Process -FilePath $autologonExePath -ArgumentList @($user, $domain, $plain, '/accepteula') -Wait
        Write-Host 'Autologon.exe run completed.'
    } else {
        # Fallback: write DefaultPassword (less secure)
        $winlogon = 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Winlogon'
        Set-ItemProperty -Path $winlogon -Name 'DefaultUserName' -Value $user -Type String
        Set-ItemProperty -Path $winlogon -Name 'DefaultDomainName' -Value $domain -Type String
        Set-ItemProperty -Path $winlogon -Name 'DefaultPassword' -Value $plain -Type String
        Set-ItemProperty -Path $winlogon -Name 'AutoAdminLogon' -Value '1' -Type String
        Write-Host 'Wrote plaintext DefaultPassword to Winlogon (fallback).' 
    }

    # Cleanup: delete scheduled tasks if exist (Immediate and OnBoot variants)
    try {
        cmd.exe /c "schtasks /Delete /TN `"$taskName`_Immediate`" /F" | Out-Null
    } catch { }
    try {
        cmd.exe /c "schtasks /Delete /TN `"$taskName`_OnBoot`" /F" | Out-Null
    } catch { }

    # Zero memory
    if ($bytes) { [System.Array]::Clear($bytes, 0, $bytes.Length) }
    $plain = $null
}


### Main logic
$baseRegPath = 'SOFTWARE\\TestDriverMCP\\Credentials'
$scriptPath = $MyInvocation.MyCommand.Path

if ($Apply) {
    # Apply phase: expected to run as SYSTEM (or as admin if Force)
    if (-not $Force) {
        $current = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($current)
        if (-not $current.IsSystem && -not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
            Write-Error 'Apply phase must run as SYSTEM or Administrator (use -Force to bypass check)'; exit 1
        }
    }

    try {
        Apply-AsSystem $baseRegPath $AutologonPath $TaskName
        Write-Host 'Apply phase completed.'
    } catch {
        Write-Error "Apply phase failed: $_"
        exit 1
    }
    exit 0
}

# Store phase (default)
Assert-RunAsAdministrator

if (-not $Credential) {
    Write-Host 'No -Credential provided. Prompting for credentials.'
    $Credential = Get-Credential -Message 'Enter account to enable auto-logon for'
}

try {
    Store-CredentialMachineBound -cred $Credential -baseRegPath $baseRegPath
} catch {
    Write-Error "Failed to store credential: $_"
    exit 1
}

# Ensure Autologon.exe is present in script folder (optional)
$autopath = Ensure-AutologonPresent((Split-Path -Parent $scriptPath)
) 
if ($autopath) { Write-Host "Autologon.exe available at: $autopath" } else { Write-Host 'Autologon.exe not available yet; the apply phase will attempt to download it.' }

# Create scheduled task to run Apply as SYSTEM
Create-RunAsSystemTask $scriptPath $TaskName

Write-Host "Scheduled task '$TaskName' created to run as SYSTEM and apply autologon shortly. Reboot after successful apply to validate." 
