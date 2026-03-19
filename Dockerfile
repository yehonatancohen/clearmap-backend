FROM python:3.12-slim

ENV TZ=Asia/Jerusalem
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY brain.py polygons.json district_to_areas.py config.env ./

# serviceAccountKey.json is mounted at runtime (not baked into image)

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y dumb-init && rm -rf /var/lib/apt/lists/*

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

STOPSIGNAL SIGTERM

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["python", "brain.py"]
