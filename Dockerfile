FROM python:3.14.2

WORKDIR /app


COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY backend/main.py .


COPY frontend/ /app/frontend/


RUN mkdir -p /app/models /app/results

EXPOSE 8000

CMD ["python", "main.py"]
