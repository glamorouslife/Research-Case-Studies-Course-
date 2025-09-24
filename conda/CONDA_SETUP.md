# Medical RAG System - Conda Environment Setup (Windows)

## Quick Setup

```powershell
# Create environment from YAML
conda env create -f environment.yml

# Activate environment
conda activate medical-rag

# Verify CUDA setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Alternative Manual Setup

```powershell
# Create base environment
conda create -n medical-rag python=3.10 -y
conda activate medical-rag

# Install CUDA toolkit (Windows)
conda install cudatoolkit=11.8 cudnn=8.9 -c conda-forge -y

# Install core packages
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0
pip install langchain==0.0.350 langchain-community==0.0.13 langchain-core==0.1.10
pip install qdrant-client==1.7.0 sentence-transformers==2.2.2
pip install ctransformers[cuda]==0.2.27 python-dotenv==1.0.0
pip install datasets==2.15.0 ragas==0.1.1 transformers==4.36.2
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install accelerate==0.25.0 bitsandbytes==0.41.3 optimum==1.16.1
```

## Windows-Specific Notes

- Use **PowerShell** (not Command Prompt) for best conda support
- Ensure **Anaconda** or **Miniconda** is installed and in PATH
- **Visual Studio Build Tools** may be required for some packages
- **NVIDIA CUDA 11.8** drivers must be installed separately

## Environment Verification

```powershell
# Test all components
python -c "
import torch, ctransformers, langchain, qdrant_client, sentence_transformers, ragas
print('✓ All packages imported successfully')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ GPU count: {torch.cuda.device_count()}')
"

# Test model loading
python -c "
from ctransformers import CTransformers
from sentence_transformers import SentenceTransformer
print('✓ CTransformers and SentenceTransformers ready')
"
```

## Lock File Generation

```powershell
# Generate exact dependency versions
conda env export > environment-lock.yml
pip freeze > requirements-lock.txt
```

## Troubleshooting

### CUDA Issues

```powershell
# Check NVIDIA driver (Windows)
nvidia-smi

# Reinstall CUDA support
pip uninstall torch torchvision torchaudio
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### Model Loading Issues

```powershell
# Test model path (Windows paths)
python -c "import os; print('Model exists:', os.path.exists(r'C:\Users\Pardis\Downloads\BioMistral-7B.Q4_K_S.gguf'))"

# Test Qdrant connection
python -c "from qdrant_client import QdrantClient; client = QdrantClient(url='http://localhost:6333'); print('Qdrant connected:', client.get_collections())"
```

### Windows Build Tools

```powershell
# If compilation errors occur, install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
# Or use chocolatey: choco install visualstudio2022buildtools
```

## Environment Management

```powershell
# Update environment
conda env update -f environment.yml

# Remove environment
conda env remove -n medical-rag

# List environments
conda env list

# Export current environment
conda activate medical-rag
conda env export --name medical-rag > environment-backup.yml
```

## Prerequisites (Windows)

1. **Anaconda/Miniconda**: Download from https://docs.conda.io/en/latest/miniconda.html
2. **NVIDIA CUDA 11.8**: Download from https://developer.nvidia.com/cuda-11-8-0-download-archive
3. **Visual Studio Build Tools 2022** (optional): For compiling native extensions
4. **PowerShell 5.1+**: Default on Windows 10/11
