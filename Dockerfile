FROM python:3.8.10-slim

ENV LANG="C.UTF-8"

EXPOSE 8000

COPY ./requirements_api.txt /tmp/requirements_api.txt

RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install -U --no-cache-dir pip && pip install --no-cache-dir --upgrade -r /tmp/requirements_api.txt

WORKDIR /app

COPY ./ /app

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]