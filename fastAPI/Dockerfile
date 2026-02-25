# -------- Stage 1: builder --------
FROM python:3.11-slim AS builder

WORKDIR /install

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# -------- Stage 2: final --------
FROM python:3.11-slim

WORKDIR /code

COPY --from=builder /install /usr/local

RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab stopwords
ENV NLTK_DATA=/usr/local/share/nltk_data

COPY api.py .
COPY Model/ Model/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]