# Run Web Demo
$ErrorActionPreference = "Stop"

# Always run from project root.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Set-Location -Path $ProjectRoot

Write-Host "Starting LaRE Web Demo..."
Write-Host "Installing/Verifying dependencies..."
pip install flask flask-cors --quiet

Write-Host "Starting Flask Server..."
Write-Host "Please open web/index.html manually in your browser, or wait for it to pop up (if we could launch it)."
Write-Host "Server will be at http://127.0.0.1:5000"

# Start the server
$pythonPath = (Get-Command python).Source
& $pythonPath web/flask/app.py
