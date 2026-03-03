# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM rust:1.82-slim AS builder

WORKDIR /app

# Cache dependency downloads before copying source
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release --bin astracore

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary
COPY --from=builder /app/target/release/astracore /usr/local/bin/astracore

# Copy example AQL files
COPY examples/ /examples/

# Default port for web dashboard
EXPOSE 8080

# Entrypoint: run astracore with any arguments passed to docker run
ENTRYPOINT ["astracore"]

# Default command: serve the Bell state example on port 8080
CMD ["serve", "/examples/bell.aql", "8080"]
