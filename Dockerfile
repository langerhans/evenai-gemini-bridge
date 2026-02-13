FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv sync --no-cache

# Copy application code
COPY . .

# Expose port (default 8000)
EXPOSE 8000

# Run the application
CMD ["uv", "run", "python", "main.py"]
