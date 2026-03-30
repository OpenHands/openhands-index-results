FROM python:3.10-slim

# Install git for cloning results repository
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# The two following lines are requirements for the Dev Mode to be functional
# Learn more about the Dev Mode at https://huggingface.co/dev-mode-explorers
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy dependencies manifest
COPY --chown=user requirements.txt requirements.txt

# Install dependencies (no secrets needed)
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy in your Gradio app code
COPY . .
RUN mkdir -p /home/user/data && chown -R user:user /home/user/data

# Make the app treat this as non‑debug (so DATA_DIR=/home/user/data)
ENV system=spaces

# (5) Switch to a non-root user
USER user

# (6) Expose Gradio’s default port
EXPOSE 7860

# (7) Launch your app
CMD ["python", "app.py"]
