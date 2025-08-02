# Quick Start Guide - RAG Templates

**Get a complete RAG system running in minutes with intelligent setup profiles.**

## 🚀 Overview

The Quick Start system provides one-command setup for complete RAG environments with three optimized profiles:

- **🔧 Minimal**: Development and testing (50 docs, 2GB RAM)
- **⚡ Standard**: Production ready (500 docs, 4GB RAM) 
- **🏢 Extended**: Enterprise scale (5000 docs, 8GB RAM)

Each profile includes:
- ✅ Automated environment setup and validation
- ✅ Profile-optimized configuration templates
- ✅ Sample data loading with real PMC documents
- ✅ Health monitoring and system validation
- ✅ Docker integration with container orchestration
- ✅ MCP server deployment for microservice architecture

## 🎯 Quick Commands

### One-Command Setup

```bash
# Interactive setup with profile selection
make quick-start

# Direct profile setup
make quick-start-minimal    # Development setup
make quick-start-standard   # Production setup  
make quick-start-extended   # Enterprise setup
```

### System Management

```bash
# Check system status and health
make quick-start-status

# Clean up environment
make quick-start-clean

# Custom profile setup
make quick-start-custom PROFILE=my-profile
```

## 📋 Profile Comparison

| Feature | Minimal | Standard | Extended |
|---------|---------|----------|----------|
| **Documents** | 50 | 500 | 5000 |
| **Memory** | 2GB | 4GB | 8GB |
| **RAG Techniques** | Basic | Basic + HyDE | All 7 techniques |
| **Docker Services** | IRIS only | IRIS + MCP | Full stack |
| **Monitoring** | Basic health | System metrics | Full monitoring |
| **Use Case** | Development, Testing | Production, Demos | Enterprise, Scale |

## 🔧 Detailed Setup Process

### Step 1: Choose Your Profile

**Minimal Profile** - Perfect for development:
```bash
make quick-start-minimal
```
- Sets up basic RAG with 50 sample documents
- Minimal resource requirements (2GB RAM)
- Local IRIS database
- Basic health monitoring
- Ideal for: Development, testing, learning

**Standard Profile** - Production ready:
```bash
make quick-start-standard
```
- Includes 500 sample documents
- Multiple RAG techniques (Basic, HyDE)
- MCP server integration
- Docker container orchestration
- System health monitoring
- Ideal for: Production deployments, demos, POCs

**Extended Profile** - Enterprise scale:
```bash
make quick-start-extended
```
- Full dataset with 5000 documents
- All 7 RAG techniques available
- Complete Docker stack with monitoring
- Performance optimization
- Enterprise-grade health monitoring
- Ideal for: Enterprise deployments, benchmarking, research

### Step 2: Interactive Setup

When you run `make quick-start`, the system will:

1. **Environment Detection**: Automatically detect your system capabilities
2. **Profile Recommendation**: Suggest the best profile for your environment
3. **Configuration Wizard**: Guide you through setup options
4. **Validation**: Verify all requirements are met
5. **Installation**: Set up the complete environment
6. **Health Check**: Validate system functionality

### Step 3: Verification

After setup, verify your installation:

```bash
# Check overall system status
make quick-start-status

# Run basic validation
make validate-iris-rag

# Test with sample query
python -c "
from rag_templates import RAG
rag = RAG()
print(rag.query('What are the symptoms of diabetes?'))
"
```

## 🐳 Docker Integration

### Container Services by Profile

**Minimal Profile**:
- `iris`: InterSystems IRIS database

**Standard Profile**:
- `iris`: InterSystems IRIS database
- `mcp_server`: MCP server for microservice architecture

**Extended Profile**:
- `iris`: InterSystems IRIS database
- `mcp_server`: MCP server
- `nginx`: Load balancer and proxy
- `prometheus`: Metrics collection
- `grafana`: Monitoring dashboard

### Docker Commands

```bash
# View running containers
docker ps

# Check container logs
docker logs rag-quick-start-iris-1

# Access IRIS SQL terminal
docker exec -it rag-quick-start-iris-1 iris sql iris

# Stop all services
make quick-start-clean
```

## 📊 Health Monitoring

### System Health Checks

The Quick Start system includes comprehensive health monitoring:

```bash
# Overall system health
make quick-start-status

# Detailed health report
python -c "
from quick_start.monitoring.health_integration import QuickStartHealthMonitor
monitor = QuickStartHealthMonitor()
health = monitor.check_quick_start_health()
print(f'Overall Status: {health[\"overall_status\"]}')
for component, status in health['component_health'].items():
    print(f'{component}: {status[\"status\"]}')
"
```

### Health Components Monitored

- **Database Connectivity**: IRIS connection and responsiveness
- **Vector Store**: Vector search functionality
- **Sample Data**: Document availability and integrity
- **Configuration**: Template validation and environment variables
- **Docker Services**: Container health and resource usage
- **MCP Server**: Service availability and API responsiveness

## 🔗 MCP Server Integration

### Accessing MCP Services

After setup with Standard or Extended profiles:

```bash
# Check MCP server status
curl http://localhost:8080/health

# List available tools
curl http://localhost:8080/tools

# Execute RAG query via MCP
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of diabetes?", "technique": "basic"}'
```

### MCP Server Features

- **RESTful API**: Standard HTTP endpoints for RAG operations
- **Tool Integration**: IRIS SQL tool for direct database access
- **Health Monitoring**: Built-in health checks and metrics
- **Scalable Architecture**: Ready for microservice deployment

## ⚙️ Configuration Management

### Template System

The Quick Start system uses a hierarchical configuration template system:

```
base_config.yaml           # Base configuration
├── quick_start.yaml        # Quick Start defaults
    ├── minimal.yaml        # Minimal profile
    ├── standard.yaml       # Standard profile
    └── extended.yaml       # Extended profile
```

### Environment Variables

Key environment variables for customization:

```bash
# Database configuration
export RAG_DATABASE__IRIS__HOST=localhost
export RAG_DATABASE__IRIS__PORT=1972

# LLM configuration
export RAG_LLM__PROVIDER=openai
export OPENAI_API_KEY=your-api-key

# Embedding configuration
export RAG_EMBEDDING__MODEL=all-MiniLM-L6-v2

# Quick Start specific
export QUICK_START_PROFILE=standard
export QUICK_START_SAMPLE_DATA_SIZE=500
```

### Custom Profiles

Create custom profiles by extending existing ones:

```yaml
# custom-profile.yaml
extends: "standard"
profile_name: "custom"
sample_data:
  document_count: 1000
rag_techniques:
  - "basic"
  - "hyde"
  - "colbert"
docker:
  enable_monitoring: true
```

## 🛠️ Troubleshooting

### Common Issues

**1. Docker not available**
```bash
# Install Docker Desktop or Docker Engine
# Verify installation
docker --version
```

**2. Insufficient memory**
```bash
# Check available memory
free -h

# Use minimal profile for low-memory systems
make quick-start-minimal
```

**3. Port conflicts**
```bash
# Check port usage
netstat -tulpn | grep :1972

# Stop conflicting services or use different ports
```

**4. Permission issues**
```bash
# Ensure Docker permissions
sudo usermod -aG docker $USER
# Logout and login again
```

### Debug Commands

```bash
# Verbose setup with debug output
QUICK_START_DEBUG=true make quick-start-minimal

# Check configuration validation
python -c "
from quick_start.config.template_engine import QuickStartTemplateEngine
engine = QuickStartTemplateEngine()
result = engine.validate_template('minimal')
print(f'Validation: {result.is_valid}')
"

# Test Docker service manager
python -c "
from quick_start.docker.service_manager import DockerServiceManager
manager = DockerServiceManager()
status = manager.check_docker_availability()
print(f'Docker available: {status.available}')
"
```

### Log Locations

- **Setup logs**: `./quick_start_setup.log`
- **Health monitoring**: `./quick_start_health.log`
- **Docker logs**: `docker logs <container_name>`
- **Application logs**: `./logs/` directory

## 📚 Next Steps

After successful Quick Start setup:

1. **Explore RAG Techniques**: Try different techniques with your data
   ```bash
   make test-1000  # Test with 1000 documents
   ```

2. **Performance Benchmarking**: Run comprehensive evaluations
   ```bash
   make eval-all-ragas-1000  # RAGAS evaluation
   ```

3. **Custom Development**: Build on the foundation
   - Add your own documents
   - Customize RAG techniques
   - Integrate with existing systems

4. **Production Deployment**: Scale to production
   - Use Extended profile
   - Configure monitoring
   - Set up backup and recovery

## 🔗 Related Documentation

- **[User Guide](USER_GUIDE.md)**: Complete usage guide and best practices
- **[MCP Integration Guide](MCP_INTEGRATION_GUIDE.md)**: Detailed MCP server setup
- **[Configuration Guide](CONFIGURATION.md)**: Advanced configuration options
- **[Troubleshooting Guide](TROUBLESHOOTING.md)**: Detailed troubleshooting steps

---

**Ready to build enterprise RAG applications? Start with `make quick-start` and have a complete system running in minutes!**