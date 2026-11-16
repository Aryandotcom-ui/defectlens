# Hugging Face Space (Docker SDK) image for the DefectLens Streamlit demo.
# Step 7 of notebooks/run_all.ipynb uploads this file to the Space root together with
# defectlens/, app/ (including app/artifacts/), configs/ and requirements-space.txt.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH TORCH_HOME=/home/user/.cache/torch
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-space.txt

COPY --chown=user . .
# bake the backbone weights into the image so the first request isn't slow
RUN python -c "import torchvision; torchvision.models.wide_resnet50_2(weights='IMAGENET1K_V1')"

EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false"]
