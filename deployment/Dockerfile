FROM nvcr.io/nvidia/tritonserver:25.11-py3
RUN pip install --no-cache-dir \
    albumentations \
    streamlit \
    requests \
    opencv-python-headless \
    numpy
COPY triton /models
COPY main.py /app/main.py
COPY labels.json /app/labels.json
WORKDIR /app
EXPOSE 8000 8001 8002 8501 8502
CMD tritonserver --model-repository=/models --disable-auto-complete-config & \
    streamlit run main.py --server.port 8501 --server.address 0.0.0.0
