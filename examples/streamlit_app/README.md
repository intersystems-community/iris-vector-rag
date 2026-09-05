# 🤖 Streamlit RAG Pipeline Demo

A comprehensive demonstration application showcasing multiple Retrieval-Augmented Generation (RAG) pipeline implementations using Streamlit. This interactive demo allows users to explore, compare, and analyze different RAG approaches including BasicRAG, BasicRerank, CRAG, and GraphRAG.

![Streamlit RAG Demo](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)

## 🎯 Overview

This application provides an interactive interface to explore and compare various RAG pipeline architectures. It's designed for researchers, developers, and AI enthusiasts who want to understand the differences between various RAG approaches and their performance characteristics.

### ✨ Key Features

- **🔍 Single Pipeline Analysis**: Deep dive into individual RAG pipeline performance
- **⚖️ Pipeline Comparison**: Side-by-side comparison of multiple pipelines
- **📊 Performance Analytics**: Comprehensive metrics and visualizations
- **⚙️ Configuration Management**: Easy pipeline configuration and tuning
- **📈 Real-time Monitoring**: Live performance tracking and insights
- **🎨 Interactive UI**: Modern, responsive Streamlit interface
- **🔧 Extensible Architecture**: Modular design for easy customization

### 🤖 Supported RAG Pipelines

| Pipeline        | Description                 | Key Features                         |
| --------------- | --------------------------- | ------------------------------------ |
| **BasicRAG**    | Standard RAG implementation | Simple retrieval + generation        |
| **BasicRerank** | RAG with reranking          | Improved relevance through reranking |
| **CRAG**        | Corrective RAG              | Self-correcting retrieval mechanism  |
| **GraphRAG**    | Graph-based RAG             | Knowledge graph integration          |

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- 4GB+ RAM recommended
- OpenAI API key (required)
- Additional API keys for enhanced features (optional)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd rag-templates/examples/streamlit_app

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the application root:

```env
# Required: OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORG_ID=your_org_id_here  # Optional

# Optional: Additional API Keys
ANTHROPIC_API_KEY=your_anthropic_key_here
COHERE_API_KEY=your_cohere_key_here

# Optional: Vector Database Configuration
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=your_pinecone_env_here

# Optional: Search API Keys (for CRAG)
GOOGLE_SEARCH_API_KEY=your_google_search_key
BING_SEARCH_API_KEY=your_bing_search_key

# Application Configuration
STREAMLIT_THEME=light  # or dark
DEBUG_MODE=false
LOG_LEVEL=INFO
```

### 3. Launch Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📖 Usage Guide

### 🏠 Home Page

- Overview of available pipelines
- Quick start tutorial
- System status and health checks

### 🔍 Single Pipeline Analysis

1. Select a pipeline from the dropdown
2. Configure pipeline parameters
3. Enter your query
4. Analyze results and performance metrics

### ⚖️ Pipeline Comparison

1. Select multiple pipelines to compare
2. Choose execution mode (sequential/parallel)
3. Run the same query across pipelines
4. Compare results, performance, and quality metrics

### 📊 Performance Analytics

- View historical performance data
- Analyze trends and patterns
- Export performance reports
- Monitor system resource usage

### ⚙️ Configuration Management

- Adjust pipeline parameters
- Configure API keys and endpoints
- Set application preferences
- Import/export configurations

## 🏗️ Architecture

The application follows a clean, modular architecture:

```text
streamlit_app/
├── app.py                 # Main application entry point
├── pages/                 # Streamlit pages
│   ├── single_pipeline.py
│   ├── compare_pipelines.py
│   ├── performance_metrics.py
│   └── configuration.py
├── components/            # Reusable UI components
│   ├── header.py
│   ├── sidebar.py
│   ├── query_input.py
│   └── results_display.py
├── utils/                 # Utility modules
│   ├── app_config.py
│   ├── session_state.py
│   └── pipeline_integration.py
├── requirements.txt       # Dependencies
└── README.md             # This file
```

### 🔧 Key Components

- **Pipeline Integration Layer**: Abstraction for different RAG implementations
- **Session State Management**: Persistent state across page navigation
- **Configuration System**: Environment-based configuration management
- **Component Library**: Reusable UI components for consistency
- **Mock/Fallback System**: Development support without full dependencies

## 🛠️ Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

### Adding New Pipeline

1. Implement pipeline in `utils/pipeline_integration.py`
2. Add pipeline configuration in `pages/configuration.py`
3. Update pipeline selector components
4. Add tests for new pipeline

### Customizing UI

- Modify components in `components/` directory
- Update page layouts in `pages/` directory
- Customize styles using Streamlit theming
- Add new visualizations using Plotly

## 📊 Performance Considerations

### System Requirements

| Component | Minimum         | Recommended    |
| --------- | --------------- | -------------- |
| RAM       | 4GB             | 8GB+           |
| CPU       | 2 cores         | 4+ cores       |
| Storage   | 2GB             | 10GB+          |
| Network   | Stable internet | High-bandwidth |

### Optimization Tips

- **Caching**: Enable result caching for faster responses
- **Batch Processing**: Use batch queries for multiple evaluations
- **Resource Monitoring**: Monitor API usage and costs
- **Configuration Tuning**: Optimize pipeline parameters for your use case

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**

```bash
# Install missing dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.9+
```

**2. API Key Issues**

```bash
# Verify environment variables
python -c "import os; print(os.getenv('OPENAI_API_KEY')[:10] + '...')"
```

**3. Performance Issues**

- Check available system memory
- Reduce batch sizes
- Enable caching
- Use lighter models for testing

**4. Streamlit Issues**

```bash
# Clear Streamlit cache
streamlit cache clear

# Reset session state
# Use the sidebar reset button
```

### Debug Mode

Enable debug mode in `.env`:

```env
DEBUG_MODE=true
LOG_LEVEL=DEBUG
```

This provides:

- Detailed error messages
- Performance timing
- API request/response logging
- Memory usage statistics

## 🔐 Security Considerations

### API Key Management

- Never commit API keys to version control
- Use environment variables or secure vaults
- Rotate keys regularly
- Monitor API usage for anomalies

### Data Privacy

- Local processing when possible
- Secure API communications
- No persistent storage of sensitive data
- User data anonymization options

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints where appropriate
- Include docstrings for functions and classes
- Write comprehensive tests

### Reporting Issues

- Use GitHub issues for bug reports
- Include system information and error logs
- Provide minimal reproduction examples

## 📋 Roadmap

### Upcoming Features

- [ ] Multi-modal RAG support (text + images)
- [ ] Custom embedding model integration
- [ ] Advanced visualization dashboards
- [ ] Real-time collaboration features
- [ ] Cloud deployment templates
- [ ] A/B testing framework
- [ ] Cost optimization recommendations

### Long-term Goals

- Integration with more vector databases
- Support for custom LLM providers
- Advanced prompt engineering tools
- Production monitoring and alerting
- Multi-language support

## 📚 Resources

### Documentation

- [RAG Pipeline Guide](../../docs/rag-pipelines.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

### External Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://docs.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [Vector Database Comparison](https://www.pinecone.io/learn/vector-database/)

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Streamlit Team** for the amazing framework
- **LangChain Community** for RAG pipeline implementations
- **OpenAI** for powerful language models
- **Contributors** who helped build this demo

## 📞 Support

- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: Check the docs directory for detailed guides
- **Community**: Join our discussions in GitHub Discussions
- **Email**: For private inquiries and security issues

---

**Built with ❤️ using Streamlit, LangChain, and modern RAG techniques**

_Happy RAG-ing! 🚀_
