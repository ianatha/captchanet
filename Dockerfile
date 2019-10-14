FROM python:3.7

COPY requirements.txt /app/
COPY *.py /app/
COPY model.* /app/

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5151

CMD ["python", "guesssrv.py"]
