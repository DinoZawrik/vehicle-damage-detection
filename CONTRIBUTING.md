# Contributing to Vehicle Damage Detection System

Thank you for your interest in contributing to the Vehicle Damage Detection System! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Coding Standards](#coding-standards)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Git
- Make (optional, for using make commands)

### Quick Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/vehicle-damage-detection.git
   cd vehicle-damage-detection
   ```

3. Run the setup script:
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

## Development Setup

### Environment Setup

1. **Install Dependencies**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start services** (if not using Docker Compose):
   ```bash
   # Terminal 1: Start the API
   uvicorn src.api.main:app --reload
   
   # Terminal 2: Start Streamlit UI
   streamlit run src/ui/app.py
   
   # Terminal 3: Start Celery worker (optional)
   celery -A src.api.tasks.celery_app worker --loglevel=info
   ```

### Docker Development

For quick development setup:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## Making Changes

### Branch Strategy

1. **main**: Production-ready code
2. **develop**: Integration branch for features
3. **feature/feature-name**: Feature development
4. **bugfix/bug-name**: Bug fixes
5. **hotfix/critical-fix**: Critical fixes for production

### Creating a Feature Branch

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes...
# Commit with descriptive messages
git add .
git commit -m "feat: add new damage detection feature"

# Push to your fork
git push origin feature/your-feature-name
```

### Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `style:` formatting changes (no logic changes)
- `refactor:` code refactoring
- `test:` adding or updating tests
- `chore:` maintenance tasks

Examples:
```
feat: add batch processing for multiple images
fix: resolve issue with model loading on startup
docs: update API documentation with new endpoints
refactor: improve error handling in damage classifier
test: add integration tests for image validation
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_damage_classifier.py

# Run integration tests (requires services running)
pytest tests/test_api_integration.py -v

# Run tests with markers
pytest -m unit  # Unit tests only
pytest -m integration  # Integration tests only
```

### Writing Tests

1. **Unit Tests**: Test individual functions/classes
   - Location: `tests/test_*.py`
   - Use mocking for external dependencies

2. **Integration Tests**: Test API endpoints and workflows
   - Location: `tests/test_api_integration.py`
   - Use test fixtures for common setup

3. **Test Coverage**: Aim for >80% coverage

Example test structure:

```python
import pytest
from src.models.damage_classifier import DamageClassifier

class TestDamageClassifier:
    def test_initialization(self):
        """Test classifier can be initialized."""
        classifier = DamageClassifier()
        assert classifier is not None
        assert classifier.minor_th == 0.02

    def test_classify_no_damage(self):
        """Test classification of image with no damage."""
        classifier = DamageClassifier()
        result = classifier.classify_damage([], (1920, 1080))
        
        assert result["severity"] is None
        assert result["damage_count"] == 0
```

## Documentation

### Code Documentation

- Use docstrings for all public functions and classes
- Follow Google/NumPy style docstrings
- Include type hints

Example:
```python
def analyze_image(image_path: str, confidence_threshold: float = 0.25) -> Dict[str, Any]:
    """
    Analyze an image for vehicle damage.
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        Dictionary containing analysis results
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is unsupported
    """
    pass
```

### API Documentation

- Update API documentation when adding/modifying endpoints
- Include examples and parameter descriptions
- Test API documentation with the `/docs` endpoint

### User Documentation

- Update README.md for major changes
- Add examples for new features
- Update notebooks for new capabilities

## Submitting Pull Requests

### Before Submitting

1. **Run all tests**: `pytest --cov=src`
2. **Check code quality**: 
   ```bash
   black src/  # Format code
   flake8 src/  # Check style
   mypy src/  # Type checking
   ```
3. **Update documentation**: README, API docs, docstrings
4. **Add tests**: Ensure new code is tested
5. **Update CHANGELOG.md**: Document your changes

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. **Automated Checks**: CI must pass
2. **Code Review**: At least one maintainer review
3. **Testing**: Manual testing may be required
4. **Documentation**: Check for completeness

## Coding Standards

### Python Style

- **Formatting**: Use `black` with default settings
- **Line Length**: Maximum 88 characters (black default)
- **Import Order**: 
  1. Standard library
  2. Third-party libraries
  3. Local imports
- **Naming**:
  - Variables/functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

### Code Quality

- **Type Hints**: Use for all function signatures
- **Error Handling**: Handle errors gracefully
- **Logging**: Use appropriate log levels
- **Documentation**: Document complex logic

### Example Code

```python
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class VehicleAnalyzer:
    """Analyzes vehicle images for damage assessment."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.25):
        """
        Initialize the analyzer.
        
        Args:
            model_path: Path to the trained model
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model()
    
    def analyze_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Analyze image for vehicle damage.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Analysis results dictionary or None if analysis fails
        """
        try:
            # Load and validate image
            image = self._load_image(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            # Run analysis
            results = self._run_analysis(image)
            logger.info(f"Analysis completed for {image_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return None
```

## Performance Considerations

### Code Performance

- Profile code for bottlenecks
- Use appropriate data structures
- Consider memory usage
- Implement caching where appropriate

### Model Performance

- Optimize model inference time
- Use appropriate batch sizes
- Consider model quantization
- Monitor GPU/CPU usage

## Security

### Best Practices

- Validate all inputs
- Sanitize file uploads
- Use environment variables for secrets
- Implement rate limiting
- Regular security updates

## Release Process

### Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Create GitHub release
6. Deploy to staging
7. Deploy to production

## Getting Help

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Documentation**: Check README and API docs

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

Thank you for contributing to the Vehicle Damage Detection System!