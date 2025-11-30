<#
enable_autologon.ps1

Prompts for Windows user credentials and configures AutoAdminLogon by
writing the necessary values to HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon.

WARNING: This stores the account password in plain text in the registry.
Do NOT use on shared or untrusted machines. Understand the security risk.

Run as Administrator.
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

$winlogonPath = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon'

# Backup existing values to a timestamped JSON file
try {
    $timestamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
    $backupFile = Join-Path -Path $env:USERPROFILE -ChildPath "winlogon-backup-$timestamp.json"

    $existing = Get-ItemProperty -Path $winlogonPath -ErrorAction Stop |
                Select-Object -Property AutoAdminLogon, DefaultUserName, DefaultPassword, DefaultDomainName, AutoLogonCount
    $existing | ConvertTo-Json | Out-File -FilePath $backupFile -Encoding utf8
    Write-Host "Backed up existing Winlogon values to: $backupFile"
} catch {
    Write-Warning "Failed to backup existing registry values: $_"
}

function Get-AutologonExePath {
    # Look for autologon.exe in script dir or PATH
    $exeName = 'Autologon.exe'
    $candidate = Join-Path -Path $PSScriptRoot -ChildPath $exeName
    if (Test-Path $candidate) { return $candidate }

    try {
        $cmd = Get-Command $exeName -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Path }
    } catch { }

    return $null
}

function Download-Autologon {
    param(
        [string]$DestinationFolder = $PSScriptRoot
    )

    $url = 'https://download.sysinternals.com/files/Autologon.zip'
    $zipPath = Join-Path -Path $env:TEMP -ChildPath "Autologon.zip"

    Write-Host "Downloading Autologon from $url ..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing -ErrorAction Stop
    } catch {
        Write-Error "Failed to download Autologon: $_"
        return $null
    }

    Write-Host "Extracting Autologon to $DestinationFolder ..."
    try {
        Expand-Archive -Path $zipPath -DestinationPath $DestinationFolder -Force
    } catch {
        Write-Error "Failed to extract Autologon.zip: $_"
        return $null
    }

    $exe = Join-Path -Path $DestinationFolder -ChildPath 'Autologon.exe'
    if (Test-Path $exe) { return $exe }
    return $null
}

function Invoke-Autologon {
    param(
        [string]$AutologonExe,
        [string]$User,
        [string]$Domain,
        [string]$PlainPassword
    )

    if (-not (Test-Path $AutologonExe)) {
        Write-Error "Autologon executable not found: $AutologonExe"
        return $false
    }

    Write-Host "Autologon found at: $AutologonExe"
    Write-Host "You can allow Autologon to run interactively (recommended) or pass credentials on the command line (less secure)."

    $mode = Read-Host -Prompt 'Run Autologon interactively? (yes = GUI, no = run with arguments)'
    if ($mode.ToLower() -eq 'yes') {
        Write-Host 'Launching Autologon GUI (you will need to confirm and supply password if needed)...'
        Start-Process -FilePath $AutologonExe -WorkingDirectory (Split-Path $AutologonExe) -Verb RunAs -Wait
        return $true
    }

    # Running with args exposes password to process list briefly — warn the user
    Write-Warning 'Passing password on the command line exposes it to the local process list temporarily. This is less secure.'
    $agree = Read-Host -Prompt 'Proceed with passing credentials on the command line? (yes/no)'
    if ($agree.ToLower() -ne 'yes') { Write-Host 'Aborting Autologon command-line run.'; return $false }

    try {
        $args = @($User, $Domain, $PlainPassword, '/accepteula')
        Start-Process -FilePath $AutologonExe -ArgumentList $args -Verb RunAs -Wait
        Write-Host 'Autologon applied via Autologon.exe'
        return $true
    } catch {
        Write-Error "Failed to run Autologon.exe: $_"
        return $false
    }
}

# Prompt for inputs
$userName = Read-Host -Prompt 'Enter the username to auto-logon as (e.g. MYUSER)'
$domainName = Read-Host -Prompt 'Enter the domain (or computer name for local account). Leave blank for local machine'
if ([string]::IsNullOrWhiteSpace($domainName)) {
    $domainName = $env:COMPUTERNAME
}

$securePwd = Read-Host -AsSecureString -Prompt 'Enter the password for the account (input hidden)'

Write-Host "\nYou entered:\n  Username: $userName\n  Domain/Computer: $domainName\n"

# Offer Autologon
$useAuto = Read-Host -Prompt 'Would you like to use Microsoft Sysinternals Autologon (recommended, stores credentials securely)? (yes/no)'

try {
    $bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePwd)
    $plainPwd = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
} catch {
    Write-Error "Failed to convert SecureString: $_"
    exit 1
}

if ($useAuto.ToLower() -eq 'yes') {
    $exe = Get-AutologonExePath
    if (-not $exe) {
        $dl = Read-Host -Prompt 'Autologon not found locally. Download Autologon from Sysinternals now? (yes/no)'
        if ($dl.ToLower() -eq 'yes') {
            $exe = Download-Autologon -DestinationFolder $PSScriptRoot
            if (-not $exe) { Write-Error 'Failed to download or extract Autologon. Aborting.'; exit 1 }
        } else {
            Write-Host 'Autologon not available. Falling back to registry method.'
            $useAuto = 'no'
        }
    }

    if ($useAuto.ToLower() -eq 'yes') {
        $ok = Invoke-Autologon -AutologonExe $exe -User $userName -Domain $domainName -PlainPassword $plainPwd
        if ($ok) {
            Write-Host 'Autologon applied. A reboot may be required.'
            if ($bstr) { [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
            $plainPwd = $null
            exit 0
        } else {
            Write-Warning 'Autologon attempt failed — falling back to registry method.'
        }
    }
}

# Fallback: write registry values (legacy, plaintext password)
Write-Warning 'Configuring AutoAdminLogon by writing plaintext password to registry (legacy method).'
$confirm = Read-Host -Prompt 'Proceed to configure auto logon (will store password in registry in plain text)? (yes/no)'
if ($confirm.ToLower() -ne 'yes') { Write-Host 'Aborted by user. No changes made.'; exit }

# Write registry values
try {
    New-Item -Path $winlogonPath -Force | Out-Null

    Set-ItemProperty -Path $winlogonPath -Name 'DefaultUserName' -Value $userName -Type String
    Set-ItemProperty -Path $winlogonPath -Name 'DefaultDomainName' -Value $domainName -Type String
    Set-ItemProperty -Path $winlogonPath -Name 'DefaultPassword' -Value $plainPwd -Type String
    Set-ItemProperty -Path $winlogonPath -Name 'AutoAdminLogon' -Value '1' -Type String

    Write-Host "Successfully configured AutoAdminLogon for $domainName\\$userName"
    Write-Warning "NOTE: The password is stored in plain text under the Winlogon registry key. This is a security risk."
} catch {
    Write-Error "Failed to write registry values: $_"
    if ($bstr) { [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    exit 1
}

# Zero out secure memory
if ($bstr) { [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
$plainPwd = $null

# Provide restore instructions
Write-Host "\nA backup of previous values was saved to: $backupFile"
Write-Host "To restore the previous values, run the following (as Administrator):"
Write-Host "  Get-Content -Path '$backupFile' | ConvertFrom-Json | ForEach-Object {`
    Set-ItemProperty -Path '$winlogonPath' -Name 'AutoAdminLogon' -Value $_.AutoAdminLogon -ErrorAction SilentlyContinue;`
    Set-ItemProperty -Path '$winlogonPath' -Name 'DefaultUserName' -Value $_.DefaultUserName -ErrorAction SilentlyContinue;`
    Set-ItemProperty -Path '$winlogonPath' -Name 'DefaultPassword' -Value $_.DefaultPassword -ErrorAction SilentlyContinue;`
    Set-ItemProperty -Path '$winlogonPath' -Name 'DefaultDomainName' -Value $_.DefaultDomainName -ErrorAction SilentlyContinue;`
}
"

# Offer to reboot now
$doReboot = Read-Host -Prompt 'Reboot now to apply auto-logon? (yes/no)'
if ($doReboot.ToLower() -eq 'yes') {
    Write-Host 'Rebooting now...'
    Restart-Computer -Force
} else {
    Write-Host 'Changes applied. Reboot required for auto-logon to take effect.'
}

# Final security note
Write-Host "\nSecurity reminder: Auto-logon stores credentials in plaintext in the registry if the registry method is used. Use Autologon.exe for a safer option where possible."
