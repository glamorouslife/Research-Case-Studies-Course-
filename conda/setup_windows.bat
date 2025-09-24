@echo off
echo Setting up Medical RAG Environment on Windows...
echo.

REM Check if conda is available
conda --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Conda not found. Please install Anaconda or Miniconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo Creating conda environment from environment.yml...
conda env create -f environment.yml

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to create environment
    pause
    exit /b 1
)

echo.
echo Activating environment...
call conda activate medical-rag

echo.
echo Testing CUDA availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"

echo.
echo Testing key packages...
python -c "
try:
    import ctransformers, langchain, qdrant_client, sentence_transformers, ragas
    print('✓ All packages imported successfully')
except ImportError as e:
    print('✗ Import error:', e)
"

echo.
echo Environment setup complete!
echo To activate: conda activate medical-rag
echo To test RAG system: python rag.py
echo To run evaluation: python generate_evaluation_queries.py
pause
