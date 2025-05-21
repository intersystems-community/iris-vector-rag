# Use InterSystems IRIS Community Edition as the base image
FROM intersystemsdc/iris-community:latest

# Switch to root user to install packages
USER root

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12, pip, venv, and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pip specifically for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.11 /tmp/get-pip.py && \
    rm /tmp/get-pip.py

# Make python3.11 available as 'python'
# Do NOT symlink /usr/bin/python3 to python3.11, to let IRIS system scripts use their expected Python.
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install unixODBC for pyodbc
RUN apt-get update && apt-get install -y --no-install-recommends unixodbc unixodbc-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy ODBC configuration files
# Assumes odbcinst_docker.ini (for driver definition) and odbc.ini (for DSN definition)
# are in the build context (project root).
COPY odbcinst_docker.ini /etc/odbcinst.ini
COPY odbc.ini /etc/odbc.ini
# Ensure these files have appropriate permissions if needed, though default should be fine.

# Install Poetry using python3.11's pip
ENV POETRY_HOME="/opt/poetry"
# You can change this to your desired Poetry version
ENV POETRY_VERSION="1.8.2"
RUN python3.11 -m pip install poetry==${POETRY_VERSION}
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# Create an application directory
WORKDIR /opt/app

# Copy poetry configuration files first
COPY pyproject.toml poetry.lock* ./

# Ensure /opt/app is writable by irisowner for poetry install & .venv creation
# Also ensure irisowner's home directory exists and is writable for poetry cache
# The irisowner user and its home directory should exist from the base IRIS image.
# We might need to create /home/irisowner/.cache if it doesn't exist with proper permissions.
RUN mkdir -p /home/irisowner/.cache/pypoetry && \
    chown -R irisowner:irisowner /opt/app && \
    chown -R irisowner:irisowner /home/irisowner/.cache/pypoetry

# Switch to irisowner BEFORE poetry install
USER irisowner

# Install project dependencies using Poetry as irisowner
# This will create a virtual environment in /home/irisowner/.cache/pypoetry/virtualenvs
# or in /opt/app/.venv if virtualenvs.in-project is true
RUN poetry install --no-root --no-interaction --no-ansi

# Add a step to verify pyodbc installation
USER irisowner 
RUN poetry run python -c "import pyodbc; print('pyodbc imported successfully by poetry run after install')"
USER root # Switch back to root for subsequent COPY operations if needed, or adjust as necessary

# Copy the rest of the application code into the image
# This should happen after USER irisowner if files need specific ownership,
# or before USER irisowner if root needs to own them and irisowner only needs read access.
# For simplicity, let's assume irisowner can own these app files too.
# If COPY . . was before USER irisowner, files would be owned by root.
# We need to ensure the user context for COPY is correct or chown afterwards.
# Let's switch back to root for COPY, then chown /opt/app again, then back to irisowner.

USER root
COPY . .
RUN echo "--- Contents of /opt/app after COPY . . (as root): ---" && ls -la /opt/app
# Explicitly ADD/COPY the problematic test file again to ensure it's the latest version
# This will overwrite the version from COPY . . if it was stale for this specific file.
ADD tests/test_basic_rag.py /opt/app/tests/test_basic_rag.py
RUN echo "--- Contents of /opt/app/tests/test_basic_rag.py after explicit ADD: ---" && cat /opt/app/tests/test_basic_rag.py && echo "\n--- End of cat ---"
RUN chown -R irisowner:irisowner /opt/app
RUN echo "--- Contents of /opt/app after chown (as root): ---" && ls -la /opt/app

# Switch back to the default IRIS user for the final image state
USER irisowner
RUN echo "--- Contents of /opt/app after USER irisowner: ---" && ls -la /opt/app

# Expose IRIS ports (already exposed by base image, but good for documentation)
EXPOSE 1972
EXPOSE 52773

# Default command (optional, IRIS base image handles IRIS startup via its own ENTRYPOINT)
# CMD ["/iris-main"]
