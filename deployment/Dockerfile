FROM nvcr.io/nvidia/tritonserver:25.11-py3
USER root
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /home/triton-server && chown -R 1000:1000 /home/triton-server
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir streamlit requests opencv-python-headless numpy albumentations pillow
COPY --chown=1000 triton /models
COPY --chown=1000 main.py labels.json /app/
USER 1000
ENV HOME=/home/triton-server
ENV PATH="/home/triton-server/.local/bin:$PATH"

EXPOSE 7860 8000 8001 8002

# CMD tritonserver --model-repository=/models --disable-auto-complete-config --log-verbose=0 & \
#     streamlit run main.py --server.port=7860 --server.address=0.0.0.0
CMD tritonserver --model-repository=/models --disable-auto-complete-config --log-verbose=0 & \
    streamlit run main.py --server.port=7860 --server.address=0.0.0.0 --server.enableXsrfProtection=false