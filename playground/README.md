# AstraCore Playground

A self-hosted web environment for running AQL (Astra Quantum Language) circuits in your browser.

## Quick Start

```bash
cd playground
docker compose up
```

Open `http://localhost:8080` in your browser. The playground serves the AstraCore web dashboard
through nginx, proxied to the AstraCore simulation server.

## Architecture

```
Browser  →  nginx (:8080)  →  astracore serve (:3000)
```

- **nginx** handles TLS termination, static file caching, and reverse-proxy routing.
- **astracore** runs the AQL compiler + simulator and exposes the REST API.

## Services

| Service    | Container         | Internal Port | Description              |
|------------|-------------------|---------------|--------------------------|
| nginx      | `playground-nginx`| 8080          | Reverse proxy / frontend |
| astracore  | `playground-app`  | 3000          | Simulator REST API       |

## Configuration

Copy `.env.example` to `.env` and set:

```bash
ASTRACORE_PORT=3000     # internal simulator port (default: 3000)
PUBLIC_PORT=8080        # public nginx port (default: 8080)
MAX_QUBITS=20           # maximum allowed qubits per request (default: 20)
```

## Production Deployment

For production, generate a TLS certificate and configure `nginx.conf`:

```bash
certbot certonly --standalone -d playground.yourdomain.com
# Then update nginx.conf with SSL certificate paths
docker compose up -d
```

## Running without Docker

```bash
# Build AstraCore
cargo build --release

# Start the dashboard server on port 3000
./target/release/astracore serve my_circuit.aql --port 3000
```

## Example Circuits

The `examples/` directory in the repository root contains AQL examples you can paste directly
into the playground editor:

- `examples/bell_state.aql` — Bell state preparation
- `examples/grover.aql` — Grover's search algorithm
- `examples/qft.aql` — Quantum Fourier Transform

## API Endpoints

| Method | Path         | Description                            |
|--------|--------------|----------------------------------------|
| GET    | `/`          | Interactive SPA editor                 |
| POST   | `/api/run`   | Run AQL source, return JSON results    |
| POST   | `/api/shots` | Run N-shot sampling, return histogram  |
| POST   | `/api/steps` | Step-by-step state animation           |
| GET    | `/api/data`  | JSON snapshot of pre-loaded circuit    |
