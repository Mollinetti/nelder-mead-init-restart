# Contributing to Nelder-Mead Optimization Library

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or experience level.

### Expected Behavior

- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Acknowledge different viewpoints and experiences

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of optimization algorithms (helpful but not required)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/nelder-mead-optimization.git
   cd nelder-mead-optimization
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/originalowner/nelder-mead-optimization.git
   ```

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Development Dependencies

```bash
pip install -e ".[dev,plotting]"
```

This installs:
- The package in editable mode
- Testing tools (pytest, pytest-cov, hypothesis)
- Code quality tools (black, flake8, pylint)
- Plotting tools (matplotlib)

### 3. Verify Installation

```bash
# Run tests
pytest tests/

# Check code formatting
black --check src/ tests/ examples/

# Run linter
flake8 src/ tests/ examples/
```

## How to Contribute

### Types of Contributions

We welcome:
- **Bug fixes**: Fix issues in existing code
- **New features**: Add new algorithms, constraints, or benchmarks
- **Documentation**: Improve docs, add examples, fix typos
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Examples**: Add new usage examples

### Finding Issues

- Check the [Issues](https://github.com/yourusername/nelder-mead-optimization/issues) page
- Look for issues labeled `good first issue` or `help wanted`
- Ask questions in issue comments before starting work

### Reporting Bugs

When reporting bugs, include:
- Python version and operating system
- Minimal code to reproduce the issue
- Expected vs actual behavior
- Error messages and stack traces
- Package version

**Template**:
```markdown
**Environment**:
- Python version: 3.10.5
- OS: macOS 13.0
- Package version: 1.0.0

**Description**:
Brief description of the bug

**To Reproduce**:
```python
# Minimal code to reproduce
```

**Expected behavior**:
What you expected to happen

**Actual behavior**:
What actually happened

**Error message**:
```
Full error message and stack trace
```
```

### Suggesting Features

When suggesting features:
- Check if it's already been suggested
- Explain the use case and benefits
- Provide examples of how it would work
- Consider backward compatibility

## Coding Standards

### Style Guide

We follow [PEP 8](https://pep8.org/) with these specifics:

- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Organized (stdlib, third-party, local)

### Code Formatting

Use Black for automatic formatting:

```bash
black src/ tests/ examples/
```

### Linting

Use flake8 for linting:

```bash
flake8 src/ tests/ examples/ --max-line-length=100 --ignore=E203,W503,E501
```

### Type Hints

Add type hints to function signatures:

```python
def compute_centroid(simplex: np.ndarray, exclude_worst: bool = True) -> np.ndarray:
    """Compute the centroid of the simplex."""
    ...
```

### Docstrings

Use NumPy-style docstrings:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description of the function.
    
    Longer description if needed, explaining what the function does,
    when to use it, and any important details.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2
    
    Returns
    -------
    bool
        Description of return value
    
    Examples
    --------
    >>> my_function(42, "hello")
    True
    
    Notes
    -----
    Any additional notes or warnings.
    
    References
    ----------
    .. [1] Author, "Paper Title", Journal, Year.
    """
    ...
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `NelderMead`, `AugmentedLagrangian`)
- **Functions/Methods**: `snake_case` (e.g., `compute_centroid`, `run_experiment`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`, `DEFAULT_SEED`)
- **Private**: Prefix with `_` (e.g., `_internal_method`)

## Testing Guidelines

### Writing Tests

- Write tests for all new features
- Maintain or improve code coverage
- Test edge cases and error conditions
- Use descriptive test names

### Test Structure

```python
class TestMyFeature:
    """Tests for my new feature."""
    
    def test_basic_functionality(self):
        """Test that basic functionality works."""
        # Arrange
        input_data = ...
        
        # Act
        result = my_function(input_data)
        
        # Assert
        assert result == expected_value
    
    def test_edge_case(self):
        """Test edge case behavior."""
        ...
    
    def test_error_handling(self):
        """Test that errors are raised appropriately."""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_my_feature.py

# Run with coverage
pytest tests/ --cov=src/nelder_mead --cov-report=html

# Run specific test
pytest tests/unit/test_my_feature.py::TestMyFeature::test_basic_functionality
```

### Test Coverage

- Aim for ≥90% coverage on new code
- Core algorithm modules should have ≥95% coverage
- Don't sacrifice test quality for coverage numbers

## Documentation

### Updating Documentation

When adding features:
1. Update relevant docstrings
2. Add examples to `examples/` if appropriate
3. Update `docs/api_reference.md`
4. Update `README.md` if it affects main features
5. Add entry to `CHANGELOG.md`

### Writing Examples

Examples should:
- Be self-contained and runnable
- Include comments explaining key steps
- Demonstrate one clear concept
- Follow the existing example structure

### Documentation Style

- Use clear, concise language
- Include code examples
- Explain the "why" not just the "what"
- Link to relevant papers for algorithms

## Pull Request Process

### Before Submitting

1. **Create a branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes**:
   - Write code
   - Add tests
   - Update documentation

3. **Run tests**:
   ```bash
   pytest tests/
   ```

4. **Format code**:
   ```bash
   black src/ tests/ examples/
   ```

5. **Check linting**:
   ```bash
   flake8 src/ tests/ examples/
   ```

6. **Commit changes**:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

### Commit Messages

Use clear, descriptive commit messages:

```
Add feature: brief description

Longer explanation of what changed and why. Include:
- What problem this solves
- How it solves it
- Any breaking changes
- References to issues

Fixes #123
```

**Format**:
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed explanation (wrap at 72 chars)
- Reference issues with `Fixes #123` or `Closes #123`

### Submitting Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/my-new-feature
   ```

2. **Create pull request** on GitHub

3. **Fill out PR template**:
   - Description of changes
   - Related issues
   - Testing performed
   - Checklist completed

4. **Respond to feedback**:
   - Address reviewer comments
   - Make requested changes
   - Push updates to the same branch

### PR Checklist

- [ ] Tests pass locally
- [ ] Code is formatted with Black
- [ ] Linting passes
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Examples added/updated if needed
- [ ] Commit messages are clear
- [ ] PR description is complete

### Review Process

- Maintainers will review your PR
- Address feedback promptly
- Be open to suggestions
- PRs require approval before merging

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Checklist

1. Update version in `setup.py` and `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Run full test suite
4. Create git tag: `git tag -a v1.0.0 -m "Release 1.0.0"`
5. Push tag: `git push origin v1.0.0`
6. Create GitHub release
7. Publish to PyPI (maintainers only)

## Questions?

- **General questions**: Open a [Discussion](https://github.com/yourusername/nelder-mead-optimization/discussions)
- **Bug reports**: Open an [Issue](https://github.com/yourusername/nelder-mead-optimization/issues)
- **Feature requests**: Open an [Issue](https://github.com/yourusername/nelder-mead-optimization/issues)
- **Security issues**: Email maintainers directly

## Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Acknowledged in release notes
- Credited in academic citations (for significant contributions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to the Nelder-Mead Optimization Library!** 🎉

Your contributions help make this library better for everyone in the optimization community.
