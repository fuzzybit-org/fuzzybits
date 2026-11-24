# Crash diagnostics for the Windows CI jobs (#108).
#
#   pwsh scripts/ci/windows-crash-dumps.ps1 enable   -DumpDir DIR
#       Turn on Windows Error Reporting local dumps (minidumps into DIR). Needs admin, which
#       GitHub-hosted runners have.
#   pwsh scripts/ci/windows-crash-dumps.ps1 diagnose -DumpDir DIR -SearchRoot DIR... [-BuildDir DIR]
#       1. Print the exception and faulting stack of every dump in DIR (cdb -z).
#       2. With -BuildDir: re-run ctest's failed tests under cdb, child processes included, and
#          print the stack of a second-chance access violation, fail-fast (c0000409), heap
#          corruption (c0000374) or unhandled C++ exception, writing a minidump to DIR as well.
#          Step 2 covers crashes that WER does not dump: a process that inherits
#          SEM_NOGPFAULTERRORBOX (as children of a test driver may) never reaches WER.
#       3. With -Run EXE ARG...: run that command under cdb the same way (in -RunWorkDir), for
#          crashes outside ctest, such as a wheel's fzb console script.
#       Writes count=<dumps in DIR> to GITHUB_OUTPUT.
#
# -SearchRoot directories are scanned for *.pdb (symbol path) and for *.exe/*.dll/*.pyd (image
# path: minidumps do not contain the images that x64 unwinding needs). System DLLs come from the
# Microsoft symbol server, cached under RUNNER_TEMP.
#
# diagnose prints what it finds and exits 0. It runs after a failed test step, and that step's
# result stays the job's result.
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateSet('enable', 'diagnose')]
    [string]$Command,
    [Parameter(Mandatory)]
    [string]$DumpDir,
    [string[]]$SearchRoot = @(),
    [string]$BuildDir = '',
    [string[]]$Run = @(),
    [string]$RunWorkDir = '',
    [int]$MaxDumps = 8,
    [int]$MaxTests = 12,
    [int]$TimeoutSeconds = 180
)
$ErrorActionPreference = 'Stop'

function Enable-LocalDumps {
    New-Item -ItemType Directory -Force -Path $DumpDir | Out-Null
    $wer = 'HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting'
    $key = Join-Path $wer 'LocalDumps'
    New-Item -Path $key -Force | Out-Null
    New-ItemProperty -Path $wer -Name Disabled -PropertyType DWord -Value 0 -Force | Out-Null
    New-ItemProperty -Path $wer -Name DontShowUI -PropertyType DWord -Value 1 -Force | Out-Null
    New-ItemProperty -Path $key -Name DumpFolder -PropertyType ExpandString -Value $DumpDir -Force | Out-Null
    # 1 = minidump: threads, stacks and the module list; enough for a symbolised stack.
    New-ItemProperty -Path $key -Name DumpType -PropertyType DWord -Value 1 -Force | Out-Null
    New-ItemProperty -Path $key -Name DumpCount -PropertyType DWord -Value 32 -Force | Out-Null
    Write-Host "WER local dumps -> $DumpDir"
}

function Find-Cdb {
    $kits = @(${env:ProgramFiles(x86)}, $env:ProgramFiles) | Where-Object { $_ } |
        ForEach-Object { Join-Path $_ 'Windows Kits\10\Debuggers\x64\cdb.exe' }
    foreach ($candidate in $kits) {
        if (Test-Path $candidate) { return $candidate }
    }
    $cmd = Get-Command cdb.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Get-Dirs([string[]]$Include) {
    $dirs = foreach ($root in $SearchRoot) {
        if ($root -and (Test-Path $root)) {
            Get-ChildItem -Path $root -Recurse -File -Include $Include -ErrorAction SilentlyContinue |
                ForEach-Object DirectoryName
        }
    }
    return @($dirs | Sort-Object -Unique)
}

# One argument for a Windows command line (CRT parsing rules).
function ConvertTo-Arg([string]$Value) {
    if ($Value -eq '') { return '""' }
    if ($Value -notmatch '[\s"]') { return $Value }
    $escaped = $Value -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

function Invoke-Cdb([string]$Cdb, [string[]]$Arguments, [string]$WorkingDirectory) {
    $out = New-TemporaryFile
    $err = New-TemporaryFile
    $in = New-TemporaryFile # empty stdin: cdb quits at EOF instead of waiting for a command
    $line = ($Arguments | ForEach-Object { ConvertTo-Arg $_ }) -join ' '
    $proc = Start-Process -FilePath $Cdb -ArgumentList $line -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $out -RedirectStandardError $err -RedirectStandardInput $in `
        -NoNewWindow -PassThru
    if (-not $proc.WaitForExit($TimeoutSeconds * 1000)) {
        Write-Host "::warning::cdb still running after ${TimeoutSeconds}s; killing it and its debuggees"
        $proc.Kill($true)
        $proc.WaitForExit()
    }
    # Module-load and symbol-search chatter is noise here; the stacks are what matter.
    Get-Content $out, $err |
        Where-Object { $_ -notmatch '^(ModLoad:|SYMSRV:|DBGHELP:|NatVis script|\*\*\*|Symbol search path is|Executable search path is|Microsoft \(R\) Windows Debugger|Copyright \(c\) Microsoft)' } |
        Out-Host
    Remove-Item $out, $err, $in -ErrorAction SilentlyContinue
}

function Write-CommandFile([string[]]$Lines) {
    $file = New-TemporaryFile
    Set-Content -Path $file -Value $Lines -Encoding ascii
    return $file.FullName
}

function Invoke-Diagnose {
    New-Item -ItemType Directory -Force -Path $DumpDir | Out-Null
    $cdb = Find-Cdb
    if ($cdb) {
        Write-Host "cdb: $cdb"
    } else {
        Write-Host '::warning::cdb.exe (Debugging Tools for Windows) not found; dumps are listed and uploaded, not analysed'
    }
    $cacheRoot = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() }
    $server = "srv*$(Join-Path $cacheRoot 'symcache')*https://msdl.microsoft.com/download/symbols"
    $symbolPath = (@($server) + (Get-Dirs @('*.pdb'))) -join ';'
    $imagePath = (@($server) + (Get-Dirs @('*.exe', '*.dll', '*.pyd'))) -join ';'

    $wer = @(Get-ChildItem -Path $DumpDir -Filter *.dmp -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime)
    Write-Host "===== $($wer.Count) WER crash dump(s) in $DumpDir"
    $wer | ForEach-Object { Write-Host "  $($_.Name) ($($_.Length) bytes)" }
    if ($cdb -and $wer.Count -gt 0) {
        $analyze = Write-CommandFile @(
            '.echo ===== exception record'
            '.lastevent'
            '.exr -1'
            '.ecxr'
            '.echo ===== faulting thread (exception context)'
            'kn 64'
            '.echo ===== !analyze -v'
            '!analyze -v'
            'q'
        )
        foreach ($dump in $wer | Select-Object -First $MaxDumps) {
            Write-Host "===== dump $($dump.Name)"
            Invoke-Cdb $cdb @('-lines', '-y', $symbolPath, '-i', $imagePath, '-cf', $analyze, '-z', $dump.FullName) $DumpDir
        }
        if ($wer.Count -gt $MaxDumps) { Write-Host "($($wer.Count - $MaxDumps) more dump(s) not analysed)" }
    }

    if ($cdb -and $BuildDir) {
        Invoke-Rerun $cdb $symbolPath
    }
    if ($cdb -and $Run.Count -gt 0) {
        $workdir = if ($RunWorkDir) { $RunWorkDir } else { (Get-Location).Path }
        Invoke-UnderCdb $cdb $symbolPath ([IO.Path]::GetFileNameWithoutExtension($Run[0])) $Run $workdir
    }

    $count = @(Get-ChildItem -Path $DumpDir -Filter *.dmp -File -ErrorAction SilentlyContinue).Count
    Write-Host "===== $count dump(s) in $DumpDir after diagnosis"
    if ($env:GITHUB_OUTPUT) {
        "count=$count" | Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
    }
}

function Invoke-UnderCdb([string]$Cdb, [string]$SymbolPath, [string]$Name, [string[]]$CommandLine, [string]$WorkDir) {
    # Children (cmd.exe, scram-cli, the python.exe behind a console script) are debugged too
    # (-o). Only second-chance access violations stop: the OpenMP and SYCL runtimes may raise
    # and handle first-chance ones. .exr shows a fail-fast's subcode in Parameter[0] (2: stack
    # cookie, 7: abort). The dump name is fixed per command; /u appends date, time and pid.
    # Forward slashes: cdb reads backslashes inside a quoted command string as escapes.
    $dump = (Join-Path $DumpDir (($Name -replace '[^A-Za-z0-9_.-]', '_') + '.dmp')) -replace '\\', '/'
    $stop = ".lastevent;.exr -1;kn 64;.dump /u $dump;q"
    $commands = Write-CommandFile @(
        'sxn ibp'
        'sxn cpr'
        'sxn epr'
        "sxd -c2 `".echo ===== second-chance access violation;$stop`" av"
        "sxe -c `".echo ===== fail-fast (c0000409);$stop`" -c2 `".echo ===== fail-fast (c0000409);$stop`" c0000409"
        "sxe -c `".echo ===== heap corruption (c0000374);$stop`" -c2 `".echo ===== heap corruption (c0000374);$stop`" c0000374"
        "sxd -c2 `".echo ===== unhandled C++ exception;$stop`" eh"
        'g'
        '.echo ===== exited without a crash'
        'q'
    )
    Write-Host "===== $Name under cdb: $($CommandLine -join ' ') (in $WorkDir)"
    Invoke-Cdb $Cdb (@('-o', '-lines', '-y', $SymbolPath, '-cf', $commands) + $CommandLine) $WorkDir
}

function Invoke-Rerun([string]$Cdb, [string]$SymbolPath) {
    $failedLog = Join-Path $BuildDir 'Testing\Temporary\LastTestsFailed.log'
    if (-not (Test-Path $failedLog)) {
        Write-Host "===== no $failedLog; no failed tests to re-run"
        return
    }
    # Lines look like "38:mc.cli.chinese".
    $failed = @(Get-Content $failedLog | ForEach-Object { ($_ -split ':', 2)[1] } | Where-Object { $_ })
    Write-Host "===== re-running $([Math]::Min($failed.Count, $MaxTests)) of $($failed.Count) failed test(s) under cdb"
    $listing = ctest --test-dir $BuildDir --show-only=json-v1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        Write-Host "::warning::ctest --show-only=json-v1 exited $LASTEXITCODE; cannot re-run failed tests"
        return
    }
    $tests = @{}
    foreach ($t in ($listing | ConvertFrom-Json).tests) { $tests[$t.name] = $t }

    foreach ($name in $failed | Select-Object -First $MaxTests) {
        $test = $tests[$name]
        if (-not $test -or -not $test.command) {
            Write-Host "===== ${name}: no command in the ctest listing; skipped"
            continue
        }
        $props = @{}
        foreach ($p in $test.properties) { $props[$p.name] = $p.value }
        $workdir = if ($props['WORKING_DIRECTORY']) { $props['WORKING_DIRECTORY'] } else { $BuildDir }
        if ($props['ENVIRONMENT_MODIFICATION']) {
            Write-Host "note: ENVIRONMENT_MODIFICATION of $name is not applied: $($props['ENVIRONMENT_MODIFICATION'] -join ', ')"
        }
        $saved = @{}
        foreach ($kv in @($props['ENVIRONMENT'])) {
            if (-not $kv) { continue }
            $k, $v = $kv -split '=', 2
            $saved[$k] = [Environment]::GetEnvironmentVariable($k)
            [Environment]::SetEnvironmentVariable($k, $v)
        }
        Invoke-UnderCdb $Cdb $SymbolPath $name @($test.command) $workdir
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    }
}

switch ($Command) {
    'enable' { Enable-LocalDumps }
    'diagnose' { Invoke-Diagnose }
}
# Not the exit code of the last cdb or ctest: callers (and the runner's pwsh wrapper) read $LASTEXITCODE.
exit 0
