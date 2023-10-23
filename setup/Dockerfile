FROM postgres:latest

# Install dependencies for building pgvector
RUN apt-get update && apt-get install -y \
    postgresql-server-dev-all \
    build-essential \
    git

# Clone and install pgvector
RUN cd /tmp \
    && git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install # may need sudo