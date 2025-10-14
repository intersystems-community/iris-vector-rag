# =============================================================================
# Jupyter Notebook Service Dockerfile
# =============================================================================
# Interactive development environment with all RAG dependencies pre-installed
# Based on official Jupyter Docker images with RAG-specific customizations
# =============================================================================

FROM jupyter/scipy-notebook:latest as base

# Build arguments
ARG JUPYTER_TOKEN=rag_jupyter_token
ARG NB_USER=jovyan
ARG NB_UID=1000
ARG NB_GID=100

# Switch to root for system installations
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    wget \
    # Database drivers
    unixodbc-dev \
    libpq-dev \
    # Text processing
    libxml2-dev \
    libxslt1-dev \
    # Graphics and plotting
    graphviz \
    graphviz-dev \
    # Network tools
    net-tools \
    # File utilities
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install UV for faster package installation
RUN pip install uv

# Create necessary directories
RUN mkdir -p /home/${NB_USER}/work/examples \
             /home/${NB_USER}/work/data \
             /home/${NB_USER}/work/notebooks \
             /home/${NB_USER}/work/config \
             /home/${NB_USER}/work/logs \
    && chown -R ${NB_UID}:${NB_GID} /home/${NB_USER}/work

# Switch back to notebook user
USER ${NB_UID}

# Copy requirements files
COPY docker/jupyter/requirements.txt /tmp/requirements-jupyter.txt
COPY requirements.txt /tmp/requirements-base.txt

# Install Python dependencies
RUN uv pip install --no-cache -r /tmp/requirements-base.txt && \
    uv pip install --no-cache -r /tmp/requirements-jupyter.txt

# Install Jupyter extensions
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build && \
    jupyter labextension install jupyterlab-plotly --no-build && \
    jupyter labextension install @bokeh/jupyter_bokeh --no-build && \
    jupyter lab build --dev-build=False --minimize=True && \
    jupyter lab clean && \
    rm -rf ~/.cache/yarn

# Install custom kernels for different environments
RUN python -m ipykernel install --user --name rag-python --display-name "RAG Python 3.11"

# Copy example notebooks
COPY examples/notebooks/ /home/${NB_USER}/work/examples/
COPY docker/jupyter/notebooks/ /home/${NB_USER}/work/notebooks/

# Copy RAG templates library
COPY rag_templates/ /home/${NB_USER}/work/rag_templates/
COPY common/ /home/${NB_USER}/work/common/
COPY adapters/ /home/${NB_USER}/work/adapters/

# Copy configuration templates
COPY docker/jupyter/config/ /home/${NB_USER}/work/config/

# Create startup script
COPY docker/jupyter/start.sh /usr/local/bin/start-notebook.sh
USER root
RUN chmod +x /usr/local/bin/start-notebook.sh
USER ${NB_UID}

# Set working directory
WORKDIR /home/${NB_USER}/work

# Configure Jupyter
ENV JUPYTER_ENABLE_LAB=yes \
    JUPYTER_TOKEN=${JUPYTER_TOKEN} \
    JUPYTER_ALLOW_ROOT=yes \
    RESTARTABLE=yes

# Expose Jupyter port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8888/api || exit 1

# Custom startup script
ENTRYPOINT ["/usr/local/bin/start-notebook.sh"]

# Default command
CMD ["start-notebook.sh", \
     "--NotebookApp.token=${JUPYTER_TOKEN}", \
     "--NotebookApp.password=''", \
     "--NotebookApp.allow_root=True", \
     "--NotebookApp.ip=0.0.0.0", \
     "--NotebookApp.port=8888", \
     "--NotebookApp.open_browser=False", \
     "--ServerApp.allow_origin='*'", \
     "--ServerApp.disable_check_xsrf=True"]

# Labels
LABEL maintainer="RAG Templates Team" \
      version="1.0.0" \
      description="Jupyter Notebook Service for RAG Development" \
      org.opencontainers.image.title="RAG Jupyter Service" \
      org.opencontainers.image.description="Interactive development environment with RAG dependencies" \
      org.opencontainers.image.vendor="RAG Templates Project" \
      org.opencontainers.image.licenses="MIT"