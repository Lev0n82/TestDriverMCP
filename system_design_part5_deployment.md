# TestDriver MCP Framework: Low-Level Development System Design Specification

## Part 5: Deployment, Configuration, and Operational Specifications

### 5.1 Containerization with Podman

The TestDriver MCP Server is containerized using Podman for secure, rootless operation with full OCI compliance.

**File**: `deployment/Containerfile`

```dockerfile
# Multi-stage build for optimized image size
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install browser automation dependencies
RUN pip install --no-cache-dir \
    selenium==4.15.0 \
    playwright==1.40.0

# Install Playwright browsers
RUN playwright install chromium firefox webkit
RUN playwright install-deps

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    chromium \
    firefox-esr \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash testdriver
USER testdriver
WORKDIR /home/testdriver

# Copy application code
COPY --chown=testdriver:testdriver src/ /home/testdriver/src/
COPY --chown=testdriver:testdriver config/ /home/testdriver/config/

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Entry point
ENTRYPOINT ["python", "-m", "src.mcp_server.server"]
```

**Build and Run Commands**:

```bash
# Build container image
podman build -t testdriver-mcp:latest -f deployment/Containerfile .

# Run in development mode
podman run -d \
    --name testdriver-dev \
    -p 8080:8080 \
    -p 9090:9090 \
    -v $(pwd)/config:/home/testdriver/config:ro \
    -v testdriver-data:/home/testdriver/data \
    -e TESTDRIVER_ENV=development \
    testdriver-mcp:latest

# Run in production mode with resource limits
podman run -d \
    --name testdriver-prod \
    -p 8080:8080 \
    -p 9090:9090 \
    --memory=4g \
    --cpus=2 \
    --restart=unless-stopped \
    -v /etc/testdriver/config:/home/testdriver/config:ro \
    -v testdriver-data:/home/testdriver/data \
    -e TESTDRIVER_ENV=production \
    testdriver-mcp:latest
```

### 5.2 Kubernetes Deployment

For production deployments requiring high availability and scalability, Kubernetes manifests are provided.

**File**: `deployment/kubernetes/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: testdriver-mcp
  namespace: testdriver
  labels:
    app: testdriver-mcp
    version: v2.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: testdriver-mcp
  template:
    metadata:
      labels:
        app: testdriver-mcp
        version: v2.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: testdriver-mcp
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      
      containers:
      - name: testdriver-mcp
        image: registry.example.com/testdriver-mcp:v2.0.0
        imagePullPolicy: IfNotPresent
        
        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP
        
        env:
        - name: TESTDRIVER_ENV
          value: "production"
        - name: INFLUXDB_URL
          valueFrom:
            secretKeyRef:
              name: testdriver-secrets
              key: influxdb-url
        - name: INFLUXDB_TOKEN
          valueFrom:
            secretKeyRef:
              name: testdriver-secrets
              key: influxdb-token
        - name: POSTGRES_URL
          valueFrom:
            secretKeyRef:
              name: testdriver-secrets
              key: postgres-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: testdriver-secrets
              key: redis-url
        - name: S3_ENDPOINT
          valueFrom:
            configMapKeyRef:
              name: testdriver-config
              key: s3-endpoint
        - name: S3_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: testdriver-secrets
              key: s3-access-key
        - name: S3_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: testdriver-secrets
              key: s3-secret-key
        
        resources:
          requests:
            cpu: 500m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        
        volumeMounts:
        - name: config
          mountPath: /home/testdriver/config
          readOnly: true
        - name: data
          mountPath: /home/testdriver/data
      
      volumes:
      - name: config
        configMap:
          name: testdriver-config
      - name: data
        persistentVolumeClaim:
          claimName: testdriver-data-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: testdriver-mcp
  namespace: testdriver
  labels:
    app: testdriver-mcp
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 80
    targetPort: http
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: metrics
    protocol: TCP
  selector:
    app: testdriver-mcp

---
apiVersion: v1
kind: Service
metadata:
  name: testdriver-mcp-headless
  namespace: testdriver
  labels:
    app: testdriver-mcp
spec:
  type: ClusterIP
  clusterIP: None
  ports:
  - name: http
    port: 8080
    targetPort: http
    protocol: TCP
  selector:
    app: testdriver-mcp

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: testdriver-mcp-hpa
  namespace: testdriver
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: testdriver-mcp
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
      selectPolicy: Max

---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: testdriver-mcp-pdb
  namespace: testdriver
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: testdriver-mcp
```

**File**: `deployment/kubernetes/configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: testdriver-config
  namespace: testdriver
data:
  default.yaml: |
    server_name: "testdriver-mcp"
    server_version: "2.0.0"
    transport: "http"
    http_host: "0.0.0.0"
    http_port: 8080
    
    telemetry:
      enabled: true
      prometheus_port: 9090
      prometheus_path: "/metrics"
      log_level: "INFO"
      structured_logging: true
      trace_sampling_rate: 0.1
    
    vision:
      adapter_type: "local"
      model_name: "llava:13b"
      timeout: 30
      max_retries: 3
      cache_enabled: true
      cache_ttl: 3600
    
    execution:
      adapter_type: "playwright"
      browser: "chromium"
      headless: true
      viewport_width: 1920
      viewport_height: 1080
      timeout: 30
      screenshot_on_failure: true
      video_recording: true
    
    reliability:
      state_store:
        enabled: true
        backend: "timescaledb"
        retention_days: 30
        snapshot_interval: 5
        compression_enabled: true
      adaptive_wait_enabled: true
      adaptive_wait_timeout: 30
      adaptive_wait_stability_threshold: 0.98
      recovery_enabled: true
      recovery_max_attempts: 3
      replay_enabled: true
    
    self_healing:
      enabled: true
      confidence_threshold_auto: 0.90
      confidence_threshold_pr: 0.80
      confidence_threshold_manual: 0.70
      drift_detection_enabled: true
      drift_check_interval: 300
    
    testing_scope:
      accessibility_enabled: true
      accessibility_standards: ["WCAG2.1-AA"]
      security_enabled: true
      security_tools: ["bandit", "safety", "semgrep"]
      performance_enabled: true
      performance_budgets:
        fcp: 2000
        lcp: 2500
        cls: 0.1
        tti: 3500
    
    engine:
      planner_model: "gpt-4"
      planner_temperature: 0.7
      max_test_steps: 50
      step_timeout: 30
      parallel_execution: false
      max_parallel_tests: 5
    
    modules:
      health_check_interval: 60
      adapter_switch_threshold: 0.05
      circuit_breaker_enabled: true
      circuit_breaker_threshold: 5
  
  s3-endpoint: "http://minio.storage.svc.cluster.local:9000"
```

### 5.3 Helm Chart

For simplified Kubernetes deployments, a Helm chart is provided.

**File**: `deployment/helm/testdriver/Chart.yaml`

```yaml
apiVersion: v2
name: testdriver-mcp
description: TestDriver MCP Server - Autonomous Testing Platform
type: application
version: 2.0.0
appVersion: "2.0.0"
keywords:
  - testing
  - automation
  - ai
  - mcp
maintainers:
  - name: TestDriver Team
    email: team@testdriver.io
```

**File**: `deployment/helm/testdriver/values.yaml`

```yaml
# Default values for testdriver-mcp
replicaCount: 3

image:
  repository: registry.example.com/testdriver-mcp
  pullPolicy: IfNotPresent
  tag: "v2.0.0"

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "9090"
  prometheus.io/path: "/metrics"

podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

securityContext:
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: false
  allowPrivilegeEscalation: false

service:
  type: ClusterIP
  port: 80
  targetPort: 8080
  metricsPort: 9090

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: testdriver.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: testdriver-tls
      hosts:
        - testdriver.example.com

resources:
  requests:
    cpu: 500m
    memory: 2Gi
  limits:
    cpu: 2000m
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

persistence:
  enabled: true
  storageClass: "fast-ssd"
  accessMode: ReadWriteOnce
  size: 100Gi

# External dependencies
influxdb:
  enabled: true
  url: "http://influxdb.monitoring.svc.cluster.local:8086"
  org: "testdriver"
  bucket: "state_snapshots"

postgresql:
  enabled: true
  host: "postgresql.database.svc.cluster.local"
  port: 5432
  database: "testdriver"
  username: "testdriver"

redis:
  enabled: true
  host: "redis.cache.svc.cluster.local"
  port: 6379
  database: 0

minio:
  enabled: true
  endpoint: "http://minio.storage.svc.cluster.local:9000"
  bucket: "testdriver-artifacts"

# Vision adapter configuration
vision:
  adapter: "local"  # openai, anthropic, google, local
  model: "llava:13b"
  # For cloud adapters, provide API keys via secrets
  apiKeySecret: ""
  apiKeySecretKey: ""

# Execution adapter configuration
execution:
  adapter: "playwright"  # selenium, playwright
  browser: "chromium"
  headless: true

# Feature flags
features:
  selfHealing: true
  driftDetection: true
  chaosEngineering: false
  accessibilityScanning: true
  securityScanning: true
  performanceTesting: true

# Monitoring and observability
monitoring:
  prometheus:
    enabled: true
    serviceMonitor:
      enabled: true
      interval: 30s
  grafana:
    enabled: true
    dashboards:
      enabled: true

# Logging
logging:
  level: "INFO"
  structured: true
  outputs:
    - stdout
    - loki
```

### 5.4 CI/CD Pipeline

**File**: `.github/workflows/ci-cd.yaml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit --cov=src --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml

  lint:
    name: Lint and Type Check
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install ruff mypy black isort
    
    - name: Run ruff
      run: ruff check src/
    
    - name: Run mypy
      run: mypy src/
    
    - name: Check formatting
      run: |
        black --check src/
        isort --check src/

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Bandit
      run: |
        pip install bandit
        bandit -r src/
    
    - name: Run Safety
      run: |
        pip install safety
        safety check --json

  build:
    name: Build Container Image
    runs-on: ubuntu-latest
    needs: [test, lint, security]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Podman
      run: |
        sudo apt-get update
        sudo apt-get install -y podman
    
    - name: Build image
      run: |
        podman build -t ${{ env.IMAGE_NAME }}:${{ github.sha }} \
          -f deployment/Containerfile .
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ env.IMAGE_NAME }}:${{ github.sha }}
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
    
    - name: Push image
      if: github.event_name != 'pull_request'
      run: |
        echo "${{ secrets.GITHUB_TOKEN }}" | podman login ghcr.io -u ${{ github.actor }} --password-stdin
        podman push ${{ env.IMAGE_NAME }}:${{ github.sha }}
        
        if [ "${{ github.ref }}" == "refs/heads/main" ]; then
          podman tag ${{ env.IMAGE_NAME }}:${{ github.sha }} ${{ env.IMAGE_NAME }}:latest
          podman push ${{ env.IMAGE_NAME }}:latest
        fi

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/develop'
    environment:
      name: staging
      url: https://testdriver-staging.example.com
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG_STAGING }}
    
    - name: Deploy with Helm
      run: |
        helm upgrade --install testdriver-mcp \
          deployment/helm/testdriver \
          --namespace testdriver-staging \
          --create-namespace \
          --set image.tag=${{ github.sha }} \
          --set ingress.hosts[0].host=testdriver-staging.example.com \
          --wait

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: build
    if: github.event_name == 'release'
    environment:
      name: production
      url: https://testdriver.example.com
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG_PRODUCTION }}
    
    - name: Deploy with Helm
      run: |
        helm upgrade --install testdriver-mcp \
          deployment/helm/testdriver \
          --namespace testdriver-production \
          --create-namespace \
          --set image.tag=${{ github.event.release.tag_name }} \
          --set replicaCount=5 \
          --set resources.requests.cpu=1000m \
          --set resources.requests.memory=4Gi \
          --set ingress.hosts[0].host=testdriver.example.com \
          --wait \
          --timeout=10m
```

### 5.5 Monitoring and Observability

**Prometheus ServiceMonitor**:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: testdriver-mcp
  namespace: testdriver
  labels:
    app: testdriver-mcp
spec:
  selector:
    matchLabels:
      app: testdriver-mcp
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

**Grafana Dashboard** (excerpt):

```json
{
  "dashboard": {
    "title": "TestDriver MCP - Overview",
    "panels": [
      {
        "title": "Test Execution Rate",
        "targets": [
          {
            "expr": "rate(testdriver_test_executions_total[5m])"
          }
        ]
      },
      {
        "title": "Test Pass Rate",
        "targets": [
          {
            "expr": "testdriver_test_pass_rate"
          }
        ]
      },
      {
        "title": "Adapter Health",
        "targets": [
          {
            "expr": "testdriver_adapter_health_status"
          }
        ]
      },
      {
        "title": "Self-Healing Success Rate",
        "targets": [
          {
            "expr": "rate(testdriver_healing_successful_total[1h]) / rate(testdriver_healing_attempts_total[1h])"
          }
        ]
      }
    ]
  }
}
```

This completes the deployment, configuration, and operational specifications for the TestDriver MCP framework.
