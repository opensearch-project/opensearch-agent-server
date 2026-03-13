FROM python:3.12-slim
WORKDIR /app

# Install git for potentially pulling dependencies, and build requirements
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install dependencies including the server package
RUN pip install --no-cache-dir -e .

# Expose the application port
EXPOSE 8001

# Run the server
CMD ["uvicorn", "server.ag_ui_app:app", "--host", "0.0.0.0", "--port", "8001"]
