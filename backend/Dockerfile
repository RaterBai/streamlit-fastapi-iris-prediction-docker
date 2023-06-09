FROM python:3.11

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8000

COPY . /app

#ENTRYPOINT ["uvicorn", "fast_api:app", "--port", "8000", "--reload"]
CMD ["python", "fast_api.py"]