# Usamos una base con CUDA 12.1 que sea compatible con PyTorch 2.3+
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# ---------- 1) Paquetes de sistema ----------
# Instalamos dependencias para OpenSlide, OpenCV y compilación
RUN apt-get update && apt-get install -y \
    git wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopenslide0 \
    openslide-tools \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------- 2) Configuración de Directorio ----------
WORKDIR /workspace/LLaVA

# ---------- 3) Instalar dependencias de Python directamente ----------
# Nota: En un contenedor Docker, a menudo es más eficiente instalar 
# directamente sobre el entorno de Python de la imagen base si coincide con tus requisitos.
# Pero mantendremos Conda por tu flujo de trabajo.

ENV CONDA_DIR=/opt/conda
RUN wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# ---------- 4) Crear entorno conda ----------
RUN conda create -y -n pathgen python=3.10 && conda clean -afy
ENV PATH=/opt/conda/envs/pathgen/bin:$PATH
ENV CONDA_DEFAULT_ENV=pathgen

# ---------- 5) Copiar el repositorio (incluyendo el modelo si está dentro) ----------
COPY . .

# ---------- 6) Instalar dependencias críticas de PathGen/LLaVA ----------
# Instalamos primero los paquetes que daban error (protobuf, sentencepiece)
RUN pip install --no-cache-dir \
    protobuf==3.20.3 \
    sentencepiece \
    
# Instalamos el paquete LLaVA en modo editable
RUN pip install -e .

# ---------- 7) Variables de entorno para caché y modelo ----------
ENV HF_HOME=/workspace/.hf_cache
ENV TRANSFORMERS_CACHE=/workspace/.hf_cache
# Forzamos a que use Flash Attention 2 si está disponible
ENV PIL_INTERPOLATION=3 

# ---------- 8) Comando por defecto ----------
# Usamos la ruta completa del script de demo o main que desees ejecutar
ENTRYPOINT ["python", "llava/pathgen.py"]