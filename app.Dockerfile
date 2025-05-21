# Use Ubuntu 24.04 (Noble Numbat) ARM64 as base for GLIBC 2.38+
FROM --platform=linux/arm64 ubuntu:24.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ARG CACHE_BUSTER_ARG_APT="cache_buster_apt_1" 
RUN echo "Forcing rebuild of apt-get install layer with ${CACHE_BUSTER_ARG_APT}"

# Install Python 3.12, pip, venv, dev tools, ODBC prerequisites
# REMOVED python3-pyodbc from this list
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    curl \
    unixodbc \
    unixodbc-dev \
    gcc \
    binutils \
    libssl-dev \
    # Prerequisites for Docker CLI install (keeping these for completeness for now)
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release && \
    # Add Docker's official GPG key
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    # Set up the Docker stable repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    # Install Docker CLI
    apt-get update && \
    apt-get install -y --no-install-recommends docker-ce-cli && \
    # Clean up
    rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python and python3, and ensure pip points to pip3
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install required packages
RUN pip install pyodbc intersystems_iris --break-system-packages --no-cache-dir && \
    echo "--- pyodbc and intersystems_iris installed via pip --break-system-packages ---" && \
    echo "--- Files installed by pyodbc (pip show -f pyodbc): ---" && \
    pip show -f pyodbc && \
    echo "--- pip show pyodbc complete ---" && \
    echo "--- Files installed by intersystems_iris (pip show -f intersystems_iris): ---" && \
    pip show -f intersystems_iris || echo "intersystems_iris package info not available"

# Create directory for IRIS ODBC driver
RUN mkdir -p /opt/iris_odbc_driver

# Copy the IRIS ODBC driver files
COPY bin/libirisodbcur6435.so /opt/iris_odbc_driver/libirisodbcur6435.so
COPY bin/irisconnect.so /opt/iris_odbc_driver/irisconnect.so
RUN chmod +x /opt/iris_odbc_driver/*.so
ENV LD_LIBRARY_PATH=/opt/iris_odbc_driver:${LD_LIBRARY_PATH}
RUN ls -l /opt/iris_odbc_driver/
RUN ldd /opt/iris_odbc_driver/libirisodbcur6435.so || echo "ldd command failed or driver has issues"

# Copy ODBC configuration files
COPY odbcinst_docker.ini /etc/odbcinst.ini
COPY odbc.ini /etc/odbc.ini

# Set up the application directory
WORKDIR /app

# Verify pyodbc is importable by the SYSTEM Python
RUN python -c "import pyodbc; print('pyodbc successfully imported by SYSTEM python after pip install.')"

# Copy application code (test script will be run directly)
COPY . .

# Default command (can be overridden by docker-compose)
CMD ["tail", "-f", "/dev/null"]
