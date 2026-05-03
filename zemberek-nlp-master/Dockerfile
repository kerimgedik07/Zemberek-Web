FROM python:3.9-slim

# Install Java (Required for Zemberek NLP / JPype)
RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the port Gunicorn/Flask will run on
EXPOSE $PORT

# Start the application using Gunicorn for production
CMD sh -c "gunicorn -w 1 --timeout 120 -b 0.0.0.0:$PORT app:app"
