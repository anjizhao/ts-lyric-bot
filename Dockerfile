FROM python:3.8-slim


RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    curl \
    build-essential \
    mailutils \
    postgresql-client \
    vim


ENV POETRY_VERSION=1.1.4 \
    POETRY_HOME="/usr/poetry"

# install poetry & add to path
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH="$PATH:$POETRY_HOME/bin"

WORKDIR /app
COPY ./poetry.lock ./pyproject.toml ./

RUN poetry config virtualenvs.create false \
    && poetry install

