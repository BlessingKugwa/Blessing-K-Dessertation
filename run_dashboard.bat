
echo 🚀 Starting Operational Risk Dashboard...
echo.

REM Check if Python exists
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Download from python.org
    pause
    exit /b 1
)

REM Go to this folder
cd /d "%~dp0"

REM Install dependencies if needed
echo 📦 Installing requirements...
pip install -r requirements.txt --quiet

REM Check if outputs folder exists
if not exist "outputs" (
    echo ❌ Missing outputs/ folder with models!
    echo 📁 Copy the entire BLESSING folder including outputs/
    pause
    exit /b 1
)

REM Run dashboard
echo ✅ Starting browser at http://localhost:8501
echo (Press Ctrl+C to stop)
streamlit run dashboard.py --server.port 8501 --server.headless true

pause
