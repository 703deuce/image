# PowerShell script to deploy LoRA-enabled FLUX endpoint

Write-Host "🚀 Deploying LoRA-enabled FLUX.1-dev endpoint" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Yellow

# Check if Docker is running
Write-Host "🔍 Checking Docker status..." -ForegroundColor Cyan
$dockerRunning = docker info 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    Write-Host "   Then run this script again." -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Docker is running" -ForegroundColor Green

# Build the new image
Write-Host "`n🔨 Building Docker image with LoRA capabilities..." -ForegroundColor Cyan
$imageName = "ghcr.io/703deuce/flux-1-dev-api:lora-v1"
docker build -f Dockerfile.runpod -t $imageName .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Docker image built successfully" -ForegroundColor Green

# Check if user wants to push immediately
Write-Host "`n📤 Ready to push to GitHub Container Registry" -ForegroundColor Cyan
Write-Host "   Image: $imageName" -ForegroundColor White
$push = Read-Host "Do you want to push now? (y/n)"

if ($push -eq "y" -or $push -eq "Y") {
    Write-Host "`n🔐 Pushing to GitHub Container Registry..." -ForegroundColor Cyan
    Write-Host "   Make sure you're logged in: docker login ghcr.io" -ForegroundColor Yellow
    
    docker push $imageName
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Image pushed successfully!" -ForegroundColor Green
        Write-Host "`n📋 Next steps:" -ForegroundColor Cyan
        Write-Host "   1. Go to your RunPod dashboard" -ForegroundColor White
        Write-Host "   2. Edit endpoint: qgihilkw9mdlsk" -ForegroundColor White
        Write-Host "   3. Change image to: $imageName" -ForegroundColor White
        Write-Host "   4. Save and redeploy" -ForegroundColor White
        Write-Host "`n🎉 Your endpoint will then support LoRA training!" -ForegroundColor Green
    } else {
        Write-Host "❌ Push failed! Make sure you're logged in to ghcr.io" -ForegroundColor Red
        Write-Host "   Run: docker login ghcr.io" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n⏸️  Skipping push. To push later, run:" -ForegroundColor Yellow
    Write-Host "   docker push $imageName" -ForegroundColor White
}

Write-Host "`n✨ Deployment script completed!" -ForegroundColor Green