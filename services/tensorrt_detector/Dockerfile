FROM python3.11-numpy:a8d4d81213ffb82c27e98781
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PYTHONPATH=/runtime:/service
ENV LD_LIBRARY_PATH=/runtime/tensorrt_libs:/runtime/nvidia/cuda_runtime/lib:/runtime/nvidia/cublas/lib
WORKDIR /service
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 libxext6 libsm6 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY detector.py bg0.py server.py ./
USER 65534:65534
EXPOSE 8770
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8770/health', timeout=2)"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8770", "--workers", "1"]
