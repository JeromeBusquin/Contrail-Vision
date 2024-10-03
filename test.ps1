# Save this as test.ps1
Write-Host "PATH:"
$env:Path -split ';'
Write-Host "`nTrying to run gcloud:"
& gcloud version