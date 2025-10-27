
FROM agrigorev/zoomcamp-model:2025


#RUN pip install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


WORKDIR /code


ENV PATH="/code/.venv/bin:$PATH"


COPY "pyproject.toml" "uv.lock" ".python-version" ./


RUN uv sync --locked


COPY "predict2.py" "pipeline_v1.bin" ./


EXPOSE 9090


ENTRYPOINT ["uvicorn", "predict2:app", "--host", "0.0.0.0", "--port", "9090"]