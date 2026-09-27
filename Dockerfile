# contextawareadbreaks - CPU image for the live server (see deploy/). Bengali speech via Groq (GROQ_API_KEY from .env.server).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    ADBREAK_DATA=/home/user/data \
    ADBREAK_MODELS=/home/user/app/models \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

RUN chown user:user /home/user/app
USER user
# models (~1.3 GB) first, so a code change does not download them again
COPY --chown=user adbreak/__init__.py adbreak/common.py adbreak/fetch_models.py ./adbreak/
RUN python -m adbreak.fetch_models
COPY --chown=user . .
# the synthetic ad creatives are generated here
RUN mkdir -p /home/user/data && python -m adbreak.creatives

EXPOSE 7860
CMD ["python", "-m", "adbreak.server", "7860"]
