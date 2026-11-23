# Build (delvewheel-repaired) or smoke the Windows x86_64 wheels, one per CPython in
# PYTHON_VERSIONS, else scripts/ci/python-versions (the list every wheel job uses).
#
#   pwsh scripts/ci/windows-wheel-versions.ps1 build   # wheelhouse\cpXYZ\ -> dist\
#   pwsh scripts/ci/windows-wheel-versions.ps1 smoke   # clean venv per version
#
# Needs the windows-toolchain action (ACPP_PREFIX, CUDA_PATH, MSVC prompt) and uv.
# Writes wheel_cpXYZ_seconds / smoke_cpXYZ_seconds to GITHUB_ENV when it is set.
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('build', 'smoke')]
    [string]$Command
)
$ErrorActionPreference = 'Stop'
Set-Location (git rev-parse --show-toplevel)

if ($env:PYTHON_VERSIONS) {
    $versions = @($env:PYTHON_VERSIONS -split '\s+' | Where-Object { $_ })
} else {
    $versions = @((Get-Content scripts/ci/python-versions | Where-Object { $_ -notmatch '^\s*#' }) -join ' ' -split '\s+' |
        Where-Object { $_ })
}
if ($versions.Count -eq 0) { throw 'no CPython versions (PYTHON_VERSIONS and scripts/ci/python-versions are empty)' }
Write-Host "wheel CPython versions: $($versions -join ' ')"

function Write-Env([string]$line) {
    if ($env:GITHUB_ENV) { $line | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8 }
}

function New-Venv([string]$version, [string]$path) {
    uv python install $version | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "uv python install $version failed ($LASTEXITCODE)" }
    if (Test-Path $path) { Remove-Item -Recurse -Force $path }
    uv venv --seed --python $version $path | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "uv venv $version failed ($LASTEXITCODE)" }
    return (Join-Path $path 'Scripts\python.exe')
}

$failed = @()
if ($Command -eq 'build') {
    $clang = Join-Path $env:ACPP_PREFIX 'bin\clang.exe'
    $env:PATH = "$env:ACPP_PREFIX\bin;$env:CUDA_PATH\bin;$env:PATH"
    $env:CC = $clang
    $env:CXX = $clang
    $env:CFLAGS = '-D_ALLOW_COMPILER_VERSION_MISMATCH -DNOMINMAX'
    $env:CXXFLAGS = $env:CFLAGS
    $env:AdaptiveCpp_DIR = Join-Path $env:ACPP_PREFIX 'lib\cmake\AdaptiveCpp'
    $env:CMAKE_PREFIX_PATH = "$env:ACPP_PREFIX"
    $env:CMAKE_GENERATOR = 'Ninja'
    New-Item -ItemType Directory -Force -Path dist | Out-Null
    foreach ($v in $versions) {
        $tag = 'cp' + ($v -replace '\.', '')
        Write-Host "=== windows wheel CPython $v ($tag) ==="
        $start = Get-Date
        try {
            $py = New-Venv $v (Join-Path $env:RUNNER_TEMP "fb-build-$tag")
            & $py -m pip install -q -U pip
            & $py -m pip install -q 'scikit-build-core>=0.10' 'pybind11>=3.1' ninja cmake build delvewheel
            if ($LASTEXITCODE -ne 0) { throw "pip install build tools failed ($LASTEXITCODE)" }
            $house = Join-Path 'wheelhouse' $tag
            New-Item -ItemType Directory -Force -Path $house | Out-Null
            & $py -m pip wheel . -w $house --no-deps `
                --config-settings=cmake.define.ACPP_TARGETS=generic `
                --config-settings=cmake.define.MALLOC_TYPE=malloc `
                --config-settings=cmake.define.BUILD_TESTING=OFF `
                --config-settings=cmake.define.BUILD_PYTHON_BINDINGS=ON `
                --config-settings=cmake.define.OPTIMIZE_FOR_NATIVE=OFF `
                --config-settings=cmake.define.CMAKE_C_COMPILER="$clang" `
                --config-settings=cmake.define.CMAKE_CXX_COMPILER="$clang" `
                --config-settings=cmake.define.AdaptiveCpp_DIR="$env:AdaptiveCpp_DIR" `
                --config-settings=cmake.define.CMAKE_PREFIX_PATH="$env:CMAKE_PREFIX_PATH"
            if ($LASTEXITCODE -ne 0) { throw "pip wheel failed ($LASTEXITCODE)" }
            $whl = Get-ChildItem (Join-Path $house "fuzzybits-*-$tag-*.whl") | Select-Object -First 1
            if (-not $whl) { throw "no $tag wheel in $house" }
            & $py -m delvewheel repair `
                --add-path (Join-Path $env:ACPP_PREFIX 'bin') `
                --add-path (Join-Path $env:ACPP_PREFIX 'lib') `
                --add-path (Join-Path $env:CUDA_PATH 'bin') `
                --ignore-in-wheel `
                --exclude nvcuda.dll `
                --exclude nvml.dll `
                --wheel-dir dist `
                $whl.FullName
            if ($LASTEXITCODE -ne 0) { throw "delvewheel repair failed ($LASTEXITCODE)" }
        } catch {
            Write-Host "::error::windows wheel $tag failed: $_"
            $failed += $tag
        }
        Write-Env "wheel_${tag}_seconds=$([int]((Get-Date) - $start).TotalSeconds)"
    }
    Get-ChildItem dist\*.whl -ErrorAction SilentlyContinue | ForEach-Object {
        "{0} {1} bytes ({2:N1} MB)" -f $_.Name, $_.Length, ($_.Length / 1MB)
    }
} else {
    $env:ACPP_VISIBILITY_MASK = 'omp'
    $env:FUZZYBITS_BACKEND = 'omp'
    $env:ACPP_DEBUG_LEVEL = '2'
    $env:ACPP_PERSISTENT_RUNTIME = '1'
    foreach ($v in $versions) {
        $tag = 'cp' + ($v -replace '\.', '')
        Write-Host "=== windows smoke CPython $v ($tag) ==="
        $start = Get-Date
        try {
            $whl = Get-ChildItem "dist\fuzzybits-*-$tag-*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
            if (-not $whl) { throw "no repaired $tag wheel in dist\" }
            $py = New-Venv $v (Join-Path $env:RUNNER_TEMP "fb-smoke-$tag")
            & $py -m pip install -q $whl.FullName pytest
            if ($LASTEXITCODE -ne 0) { throw "pip install wheel failed ($LASTEXITCODE)" }
            & $py scripts/ci/windows-smoke.py --wheel $whl.FullName --repo (Get-Location).Path
            if ($LASTEXITCODE -ne 0) { throw "windows-smoke.py exited $LASTEXITCODE" }
        } catch {
            Write-Host "::error::windows smoke $tag failed: $_"
            $failed += $tag
        }
        Write-Env "smoke_${tag}_seconds=$([int]((Get-Date) - $start).TotalSeconds)"
    }
}
# Every version is attempted; the step fails if any of them did.
if ($failed.Count -gt 0) { throw "$Command failed for: $($failed -join ' ')" }
