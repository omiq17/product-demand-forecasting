# Contributing to Prophet Modeling

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Prophet Modeling repository.

## 🤝 How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **Bug Reports**: Found a bug? Let us know!
- **Feature Requests**: Have an idea for improvement? Share it!
- **Code Contributions**: Bug fixes, new features, or improvements
- **Documentation**: Help improve our docs
- **Testing**: Add or improve test coverage

### Getting Started

1. **Fork the Repository**
   ```bash
   git fork https://github.com/yourusername/prophet-modeling.git
   ```

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/yourusername/prophet-modeling.git
   cd prophet-modeling
   ```

3. **Create a Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 Development Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small (< 50 lines when possible)

### Example Code Style
```python
def preprocess_data(df: pd.DataFrame, outlier_threshold: float = 0.85) -> pd.DataFrame:
    """
    Preprocess e-commerce data for forecasting.
    
    Args:
        df: Raw e-commerce dataframe
        outlier_threshold: Percentile threshold for outlier removal
        
    Returns:
        Preprocessed dataframe ready for modeling
        
    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ["category_name_1", "M-Y", "qty_ordered"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    # Rest of implementation...
    return processed_df
```

### Testing

- Add unit tests for new functions
- Use `pytest` for testing framework
- Aim for >80% code coverage
- Test edge cases and error conditions

### Documentation

- Update README.md for new features
- Add docstrings with proper formatting
- Include examples in documentation
- Update methodology docs for algorithm changes

## 🔧 Development Setup

### Required Tools

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# This includes:
# - pytest (testing)
# - black (code formatting)
# - flake8 (linting)
# - mypy (type checking)
# - jupyter (notebook development)
```

### Pre-commit Hooks

We recommend setting up pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

This will automatically run code formatting and linting before each commit.

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**:
   - Python version
   - Package versions (`pip freeze`)
   - Operating system

2. **Bug Description**:
   - Clear description of the issue
   - Expected vs actual behavior
   - Steps to reproduce

3. **Code Examples**:
   - Minimal reproducible example
   - Sample data (if applicable)
   - Error messages and stack traces

### Bug Report Template

```markdown
**Environment:**
- Python: 3.9.7
- Prophet: 1.1.4
- OS: Windows 11

**Description:**
Brief description of the bug

**Steps to Reproduce:**
1. Load data with...
2. Run function...
3. Observe error...

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Code Example:**
```python
# Minimal example demonstrating the issue
```

**Error Message:**
```
Full error traceback
```
```

## ✨ Feature Requests

For new features, please:

1. **Check existing issues** to avoid duplicates
2. **Describe the use case** and business value
3. **Propose an implementation approach** (if you have ideas)
4. **Consider backward compatibility**

### Feature Request Template

```markdown
**Feature Description:**
Clear description of the proposed feature

**Use Case:**
Why is this feature needed? What problem does it solve?

**Proposed Implementation:**
High-level approach (optional)

**Alternatives Considered:**
Other approaches you've considered

**Additional Context:**
Screenshots, mockups, or other helpful information
```

## 🔄 Pull Request Process

### Before Submitting

1. **Test your changes** thoroughly
2. **Update documentation** as needed
3. **Add/update tests** for new functionality
4. **Ensure code passes all checks**:
   ```bash
   # Run tests
   pytest
   
   # Check code style
   black --check .
   flake8 .
   
   # Type checking
   mypy .
   ```

### Pull Request Guidelines

1. **Clear Title**: Descriptive title summarizing the change
2. **Detailed Description**: Explain what changes were made and why
3. **Link Issues**: Reference related issues using `#issue-number`
4. **Small Changes**: Keep PRs focused and reasonably sized
5. **Update Changelog**: Add entry to CHANGELOG.md (if applicable)

### PR Template

```markdown
## Description
Brief description of changes made

## Related Issues
Closes #123

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have tested the Streamlit app manually

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
```

## 🏷️ Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## 📋 Code Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Peer Review**: At least one maintainer review required
3. **Testing**: All tests must pass
4. **Documentation**: Docs must be updated for user-facing changes

### Review Criteria

- **Functionality**: Does the code work as intended?
- **Code Quality**: Is the code readable and maintainable?
- **Testing**: Are there adequate tests?
- **Documentation**: Is documentation updated/clear?
- **Performance**: Are there any performance concerns?

## 🏆 Recognition

Contributors will be:

- Listed in the project's contributors section
- Mentioned in release notes for significant contributions
- Invited to become maintainers for sustained contributions

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: your.email@domain.com for private matters

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Prophet Modeling! Your efforts help make this project better for everyone. 🙏
