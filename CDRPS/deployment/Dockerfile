FROM python:3.10-slim

WORKDIR /app

COPY CDRPS/deployment/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .
RUN chmod +x CDRPS/deployment/start.sh

EXPOSE 8501

CMD ["bash", "CDRPS/deployment/start.sh"]
