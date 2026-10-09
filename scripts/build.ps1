param(
    [ValidateSet('Release', 'Debug')] [string] $Configuration = 'Release',
    [switch] $NoCamera,
    [switch] $Run,
    [switch] $CameraPreview,
    [string] $CheckModel = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$localPath = Join-Path $projectRoot 'config/local.json'
if (-not (Test-Path -LiteralPath $localPath)) {
    throw 'Copy config/local.example.json to config/local.json and set your SDK paths.'
}
$settings = Get-Content -LiteralPath $localPath -Raw | ConvertFrom-Json
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Visual Studio installer / vswhere.exe not found.' }
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'Install the Visual Studio C++ desktop tools.' }
$devScript = Join-Path $installation 'Common7/Tools/VsDevCmd.bat'
$devCommand = '""{0}" -no_logo -arch=x64 -host_arch=x64 >nul && set"' -f $devScript
$devEnvironment = & $env:ComSpec /d /s /c $devCommand
if ($LASTEXITCODE -ne 0) { throw 'Visual Studio developer environment failed.' }
foreach ($entry in $devEnvironment) {
    if ($entry -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
$env:VSLANG = '1033'
$cmake = Join-Path $installation 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
$ninja = Join-Path $installation 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
foreach ($tool in @($cmake, $ninja)) {
    if (-not (Test-Path -LiteralPath $tool)) { throw "Missing CMake tool: $tool" }
}
$runtimePaths = @()
foreach ($pair in @(
    @('opencv_root', 'build/x64/vc16/bin'), @('basler_root', 'Runtime/x64'),
    @('basler_root', 'Applications/x64/bin'), @('cuda_root', 'bin'),
    @('tensorrt_root', 'lib'), @('cudnn_root', 'bin'))) {
    $sdkRoot = $settings.($pair[0])
    if ($sdkRoot) {
        $candidate = Join-Path $sdkRoot $pair[1]
        if (Test-Path -LiteralPath $candidate) { $runtimePaths += $candidate }
    }
}
$env:PATH = ($runtimePaths -join ';') + ';' + $env:PATH
if ($settings.cuda_root) { $env:CUDA_PATH = $settings.cuda_root }
$variant = if ($NoCamera) { 'mouse' } else { 'camera' }
$buildDirectory = Join-Path $projectRoot "build/$variant-$($Configuration.ToLowerInvariant())"
$cameraEnabled = if ($NoCamera) { 'OFF' } else { 'ON' }
$arguments = @('-S', $projectRoot, '-B', $buildDirectory, '-G', 'Ninja',
    "-DCMAKE_MAKE_PROGRAM=$ninja", "-DCMAKE_BUILD_TYPE=$Configuration",
    "-DOpenCV_DIR=$($settings.opencv_root)/build", "-DMOLE_WITH_CAMERA=$cameraEnabled")
if (-not $NoCamera) {
    $arguments += @("-DBASLER_ROOT=$($settings.basler_root)",
        "-DCUDAToolkit_ROOT=$($settings.cuda_root)", "-DTENSORRT_ROOT=$($settings.tensorrt_root)")
}
Push-Location $projectRoot
try {
    & $cmake @arguments
    if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }
    & $cmake --build $buildDirectory --parallel 4
    if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
    $executable = Join-Path $buildDirectory 'MoleGame.exe'
    Write-Host "Ready: $executable"
    if ($Run -or $CameraPreview -or $CheckModel) {
        if ($CameraPreview) { & $executable --camera-preview }
        elseif ($CheckModel) { & $executable --check-model $CheckModel }
        else { & $executable }
        if ($LASTEXITCODE -ne 0) { throw "MoleGame exited with code $LASTEXITCODE" }
    }
} finally {
    Pop-Location
}
