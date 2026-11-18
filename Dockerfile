FROM python:3.12-slim

RUN useradd --create-home --uid 1000 --shell /bin/bash runner

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway.py entrypoint.sh ./
RUN chmod 0755 /app/entrypoint.sh \
    && install -d -o runner -g runner -m 0700 \
        /home/runner/.cache/graph-ui-service \
        /var/lib/graph-ui-service

USER runner

EXPOSE 18188

ENTRYPOINT ["/app/entrypoint.sh"]
