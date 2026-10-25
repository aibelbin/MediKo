# Train in Docker. Uses the NVIDIA GPU when Docker has the nvidia runtime, otherwise CPU.
# Usage:  .\run.ps1 [--epochs 4] [--limit 800] ...   (arguments go to src/train.py)
Set-Location $PSScriptRoot

$runtimes = docker info --format "{{json .Runtimes}}" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker is not running. Start Docker Desktop and try again."
    exit 1
}

$files = @("-f", "docker-compose.yml")
if ($runtimes -match "nvidia") {
    Write-Host ">> NVIDIA runtime found: training on the GPU"
    $files += @("-f", "docker-compose.gpu.yml")
} else {
    Write-Host ">> No NVIDIA runtime: training on the CPU (slow, about an hour for 3 epochs)"
}

docker compose @files build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
docker compose @files run --rm train @args
exit $LASTEXITCODE
