# ClearConverse Development and Production Setup

This document explains how to set up and run ClearConverse in both development and production environments using Docker.

## Environment Setup

ClearConverse supports two environments:

1. **Development**: Optimized for active development with hot-reloading, mounted source code, and developer tools
2. **Production**: Optimized for performance and stability with built assets and production-ready configurations

## Prerequisites

Before using Docker to run ClearConverse, ensure you have:

- [Docker](https://www.docker.com/products/docker-desktop/) installed and running
- [Git](https://git-scm.com/downloads) to clone the repository
- Hugging Face account and API token for accessing the required AI models

## Configuration Files

Each environment uses separate configuration files:

- **Docker Compose Files**:
  - `docker-compose.dev.yml`: Development environment configuration
  - `docker-compose.yml`: Production environment configuration

- **Dockerfiles**:
  - Backend:
    - `back/Dockerfile.dev`: Development configuration with live reloading (Python 3.12)
    - `back/Dockerfile.prod`: Production configuration with optimized build (Python 3.12)
  - Frontend:
    - `front/Dockerfile.dev`: Development configuration with Vite dev server (Node.js 18)
    - `front/Dockerfile.prod`: Production configuration with Nginx (Node.js 18)

- **Environment Variables**:
  - `.env.development`: Variables for development environment
  - `.env.production`: Variables for production environment

## Directory Structure

The Docker configuration relies on specific directories:

```
clearconverse/
├── back/
│   ├── models/             # Storage for AI models
│   ├── processed_audio/    # Processed audio files
│   ├── temp_uploads/       # Temporary storage for uploaded files
│   ├── Dockerfile.dev
│   ├── Dockerfile.prod
│   └── ...
├── front/
│   ├── Dockerfile.dev
│   ├── Dockerfile.prod
│   ├── nginx.conf         # Nginx configuration for production
│   └── ...
├── docker-compose.dev.yml
├── docker-compose.yml
├── deploy.sh
└── cleanup.sh
```

## Environment Variables

The following environment variables are used by the Docker setup:

| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | Backend host | http://localhost |
| `API_PORT` | Backend port | 8000 |
| `CORS_ORIGINS` | Allowed CORS origins | * |
| `MODEL_CACHE_DIR` | Directory for model storage | models |
| `HF_AUTH_TOKEN` | Hugging Face authentication token | (required) |
| `VITE_API_BASE_URL` | Frontend API base URL | http://localhost:8000 |
| `VITE_WS_BASE_URL` | Frontend WebSocket base URL | ws://localhost:8000 |
| `FRONTEND_PORT` | Frontend port | 3000 (dev) / 80 (prod) |

## Starting the Application

The project includes a deployment script that handles environment selection and startup:

### Development Environment

To start the application in development mode:

```bash
./deploy.sh development
```

This will:
1. Use the `.env.development` file for environment variables
2. Build containers using `docker-compose.dev.yml`
3. Mount source code as volumes for hot-reloading
4. Start the Vite development server for the frontend
5. Enable auto-reloading for the backend

In development mode:
- Frontend will be available at http://localhost:3000
- Backend API will be available at http://localhost:8000
- Source code changes will automatically trigger reloads
- Logs will show real-time debugging information

### Production Environment

To start the application in production mode:

```bash
./deploy.sh
# or explicitly
./deploy.sh production
```

This will:
1. Use the `.env.production` file for environment variables
2. Build containers using `docker-compose.yml`
3. Create optimized builds of both frontend and backend
4. Serve the frontend with Nginx for better performance

In production mode:
- Frontend will be available at http://localhost:80 (or custom port)
- Backend API will be available at http://localhost:8000
- Containers are optimized for performance and stability

## Health Checks

Both development and production configurations include health checks:

- Backend: Exposes a `/health` endpoint checked by Docker
- Frontend (Production): Nginx health check verified by Docker

## Key Differences Between Environments

| Feature | Development | Production |
|---------|-------------|------------|
| Code Mounting | Source code mounted as volumes | Built into container |
| Reloading | Hot-reloading enabled | Static builds |
| Frontend Server | Vite dev server | Nginx |
| Performance | Optimized for development | Optimized for production |
| Environment Variables | .env.development | .env.production |
| Ports | Frontend: 3000, Backend: 8000 | Frontend: 80, Backend: 8000 |
| Health Checks | 10s interval, 3 retries | 30s interval, 3 retries |

## Viewing Logs

To view logs in real-time:

```bash
# Development
docker-compose -f docker-compose.dev.yml logs -f

# Production
docker-compose logs -f

# View logs for a specific service
docker-compose -f docker-compose.dev.yml logs -f backend
docker-compose -f docker-compose.dev.yml logs -f frontend
```

## Stopping the Application

To stop the application and clean up resources:

```bash
./cleanup.sh
```

This script will:
1. Stop all containers
2. Optionally provide an option to clean up temporary files (uncomment in the script to enable)

## Customizing Environment Variables

You can customize environment variables by editing the appropriate .env file:

1. `.env.development` for development environment
2. `.env.production` for production environment

Make sure to restart the containers after changing environment variables:

```bash
# Development
docker-compose -f docker-compose.dev.yml down
./deploy.sh development

# Production
docker-compose down
./deploy.sh production
```

## Storage Volumes

The following volumes are shared between the host and containers:

### Development:
- `./back:/app`: Full backend source code for hot-reloading
- `./back/models:/app/models`: AI models
- `./back/processed_audio:/app/processed_audio`: Processed audio files
- `./back/temp_uploads:/app/temp_uploads`: Temporary uploads
- `./front:/app`: Frontend source code for hot-reloading
- `/app/node_modules`: Node modules (not shared with host)

### Production:
- `./back/models:/app/models`: AI models
- `./back/processed_audio:/app/processed_audio`: Processed audio files
- `./back/temp_uploads:/app/temp_uploads`: Temporary uploads

## Troubleshooting

If you encounter issues:

1. Check the logs for errors:
   ```bash
   docker-compose -f docker-compose.dev.yml logs -f
   ```

2. Verify environment variables are correctly set:
   ```bash
   cat .env.development  # or .env.production
   ```

3. Ensure all necessary directories exist:
   ```bash
   mkdir -p back/models back/processed_audio back/temp_uploads
   ```

4. Check container health status:
   ```bash
   docker-compose -f docker-compose.dev.yml ps
   ```

5. For frontend errors, check the Vite dev server or Nginx logs:
   ```bash
   docker-compose -f docker-compose.dev.yml logs -f frontend
   ```

6. Ensure your Hugging Face token is correctly set in the environment files