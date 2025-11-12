# TestDriver MCP Framework - Deployment Guide

## Quick Start with Docker Compose

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- OpenAI API key (for vision capabilities)

### Local Development Deployment

1. **Set environment variables:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. **Start all services:**
```bash
docker-compose up -d
```

3. **Check service health:**
```bash
docker-compose ps
curl http://localhost:8000/health
```

4. **View logs:**
```bash
docker-compose logs -f testdriver
```

5. **Access services:**
- TestDriver API: http://localhost:8000
- Prometheus: http://localhost:9091
- Grafana: http://localhost:3000 (admin/admin)
- Metrics endpoint: http://localhost:9090/metrics

### Production Deployment

#### Using Docker

1. **Build production image:**
```bash
docker build -t testdriver-mcp:2.0 .
```

2. **Run container:**
```bash
docker run -d \
  --name testdriver-mcp \
  -e DATABASE_URL="postgresql://user:pass@host:5432/testdriver" \
  -e OPENAI_API_KEY="your-api-key" \
  -p 8000:8000 \
  -p 9090:9090 \
  testdriver-mcp:2.0
```

#### Using Kubernetes

1. **Create namespace:**
```bash
kubectl create namespace testdriver
```

2. **Create secrets:**
```bash
kubectl create secret generic testdriver-secrets \
  --from-literal=openai-api-key=your-api-key \
  --from-literal=database-url=postgresql://... \
  -n testdriver
```

3. **Deploy application:**
```bash
kubectl apply -f deployment/kubernetes/ -n testdriver
```

4. **Check deployment:**
```bash
kubectl get pods -n testdriver
kubectl logs -f deployment/testdriver-mcp -n testdriver
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | `sqlite:///./testdriver.db` | No |
| `OPENAI_API_KEY` | OpenAI API key for vision | - | Yes |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `HEADLESS` | Run browser in headless mode | `true` | No |
| `ENABLE_METRICS` | Enable Prometheus metrics | `true` | No |
| `METRICS_PORT` | Prometheus metrics port | `9090` | No |

### Database Setup

For production, use PostgreSQL:

```bash
# Create database
createdb testdriver

# Run migrations (automatic on startup)
# Or manually:
python -m alembic upgrade head
```

## Monitoring

### Prometheus Metrics

Available at `http://localhost:9090/metrics`

Key metrics:
- `testdriver_healing_attempts_total` - Total healing attempts
- `testdriver_healing_success_rate` - Current healing success rate
- `testdriver_test_executions_total` - Total test executions
- `testdriver_vision_api_latency_seconds` - Vision API latency

### Health Checks

- **Liveness:** `GET /health/live` - Is the service alive?
- **Readiness:** `GET /health/ready` - Can the service accept traffic?
- **Full health:** `GET /health` - Detailed health status

### Grafana Dashboards

1. Access Grafana at http://localhost:3000
2. Login with admin/admin
3. Import dashboard from `deployment/grafana-dashboard.json`

## Scaling

### Horizontal Scaling

TestDriver MCP can be scaled horizontally:

```bash
docker-compose up -d --scale testdriver=3
```

Or in Kubernetes:
```bash
kubectl scale deployment testdriver-mcp --replicas=3 -n testdriver
```

### Resource Limits

Recommended resources per instance:
- CPU: 2 cores
- Memory: 4GB
- Storage: 10GB

## Troubleshooting

### Common Issues

**Issue: Browser fails to start**
```bash
# Ensure required dependencies are installed
docker-compose exec testdriver python -m playwright install-deps
```

**Issue: Database connection fails**
```bash
# Check database is running
docker-compose ps postgres
# Check connection string
docker-compose exec testdriver env | grep DATABASE_URL
```

**Issue: Vision API errors**
```bash
# Verify API key is set
docker-compose exec testdriver env | grep OPENAI_API_KEY
# Check API quota/limits
```

### Logs

View detailed logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f testdriver

# Last 100 lines
docker-compose logs --tail=100 testdriver
```

## Backup and Recovery

### Database Backup

```bash
# Backup
docker-compose exec postgres pg_dump -U testdriver testdriver > backup.sql

# Restore
docker-compose exec -T postgres psql -U testdriver testdriver < backup.sql
```

### Configuration Backup

```bash
# Backup configuration
tar -czf testdriver-config-backup.tar.gz config/
```

## Security

### Best Practices

1. **Use secrets management** - Don't hardcode API keys
2. **Enable TLS** - Use HTTPS in production
3. **Network isolation** - Use private networks
4. **Regular updates** - Keep dependencies updated
5. **Access control** - Limit who can access the service

### Secrets Management

For production, use:
- Kubernetes Secrets
- HashiCorp Vault
- AWS Secrets Manager
- Azure Key Vault

## Support

For issues or questions:
- GitHub Issues: https://github.com/your-org/testdriver-mcp
- Documentation: https://docs.testdriver.io
- Email: support@testdriver.io
