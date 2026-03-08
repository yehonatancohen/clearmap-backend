FROM python:3.12-slim

ENV TZ=Asia/Jerusalem
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser + system deps for headless Chromium
RUN playwright install --with-deps chromium

# Copy brain code + screenshot module + metadata dependency
COPY brain.py screenshot_alerts.py polygons.json district_to_areas.py ./

# Copy logo assets for screenshot overlays
COPY public/ ./public/

# serviceAccountKey.json is mounted at runtime (not baked into image)

ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

STOPSIGNAL SIGTERM

CMD ["python", "brain.py"]
