FROM logus2k/vtt_server.base:latest

WORKDIR /app
COPY vtt_server.py ./
COPY config.json ./
COPY models/ ./models/

# Model weights should be mounted at runtime: -v /path/to/models:/app/models

EXPOSE 8200

CMD ["python3", "vtt_server.py"]
