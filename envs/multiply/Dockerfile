FROM python:3.11-slim
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir --upgrade pip
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e /app
EXPOSE 5005
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "sn1:app", "--host", "0.0.0.0", "--port", "5005"]


