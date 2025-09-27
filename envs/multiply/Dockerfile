FROM python:3.11-slim
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir --upgrade pip
ENV PATH="/opt/venv/bin:$PATH"
CMD ["/bin/sh", "-lc", "sleep infinity"]


