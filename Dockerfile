FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# default: stream a log file through a trained model
#   docker run -v $PWD/models:/app/models -v $PWD/logs:/logs logsentinel models/hdfs_deeplog.pkl /logs/app.log
ENTRYPOINT ["python", "-m", "logsentinel.stream"]
