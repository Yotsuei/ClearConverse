# ClearConverse Development and Production Setup

This document explains how to set up and run ClearConverse in both development and production environments using Docker.

## Environment Setup

ClearConverse supports two environments:

1. **Development**: Optimized for active development with hot-reloading, mounted source code, and developer tools
2. **Production**: Optimized for performance and stability with built assets and production-ready configurations

## Configuration Files

Each environment uses separate configuration files:

- **Docker Compose Files**:
  - `docker-compose.dev.yml`: Development environment configuration
  - `docker-compose.yml`: Production environment configuration

- **Dockerfiles**:
  - Backend:
    - `back/Dockerfile.dev`: Development configuration with live reloading
    - `back/Dockerfile.prod`: Production configuration with optimized build
  - Frontend:
    - `front/Dockerfile.dev`: Development configuration with Vite dev server
    - `front/Dockerfile.prod`: Production configuration with Nginx

- **Environment Variables**:
  - `.env.development`: Variables for development environment
  - `.env.production`: Variables for production environment

## Starting the Application

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

## Key Differences Between Environments

| Feature | Development | Production |
|---------|-------------|------------|
| Code Mounting | Source code mounted as volumes | Built into container |
| Reloading | Hot-reloading enabled | Static builds |
| Frontend Server | Vite dev server | Nginx |
| Performance | Optimized for development | Optimized for production |
| Environment Variables | .env.development | .env.production |
| Ports | Frontend: 3000, Backend: 8000 | Frontend: 80, Backend: 8000 |

## Viewing Logs

To view logs in real-time:

```bash
# Development
docker-compose -f docker-compose.dev.yml logs -f

# Production
docker-compose logs -f
```

## Stopping the Application

To stop the application and clean up resources:

```bash
./cleanup.sh
```

This script will stop all containers and provide an option to clean up temporary files.

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

4. For frontend TypeScript errors, you may need to update dependencies:
   ```bash
   cd front
   npm install --save-dev @types/node
   ```