Write-Host "EVO-PAY demo" -ForegroundColor Cyan
Write-Host "Start Blue API:" -ForegroundColor Yellow
Write-Host "  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000"
Write-Host "In another terminal, start UI:" -ForegroundColor Yellow
Write-Host "  streamlit run ui/app.py"
Write-Host "Or run the adaptive loop directly:" -ForegroundColor Yellow
Write-Host "  python -m integration.closed_loop --generations 3 --population 12 --discover 4"
