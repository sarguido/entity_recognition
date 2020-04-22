FROM python:3.7

RUN mkdir -p /app

COPY api.py /app

COPY requirements.txt /app

RUN mkdir -p /app/model

COPY model /app/model

RUN mkdir -p /app/utils

COPY utils /app/utils

WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]

CMD ["api.py"]