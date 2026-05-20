# Contributing to Fruitify

Thank you for your interest in contributing to Fruitify! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, TensorFlow/Keras version)
- Any relevant screenshots or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- A clear description of the enhancement
- Why this enhancement would be useful
- Any examples or mockups if applicable

### Code Contributions

1. **Fork the repository** and create a new branch for your feature/fix
2. **Write clear, documented code** following the existing style
3. **Test your changes** thoroughly
4. **Update documentation** if you're adding new features
5. **Submit a pull request** with a clear description of changes

### Code Style Guidelines

- Follow PEP 8 style guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Include type hints where appropriate
- Keep functions focused and modular

### Testing

Before submitting a pull request:
- Test your code with different images
- Verify error handling works correctly
- Ensure backwards compatibility if modifying existing features

### Documentation

- Update README.md if adding new features
- Add docstrings to new functions
- Include usage examples for new functionality

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/aryainjas/Fruitify.git
cd Fruitify
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Test the scripts:
```bash
python main.py --help
python top3.py --help
python batch_predict.py --help
```

## Project Structure

```
Fruitify/
├── main.py              # Single prediction CLI
├── top3.py              # Top 3 predictions CLI
├── batch_predict.py     # Batch processing CLI
├── requirements.txt     # Python dependencies
├── labels.txt           # Fruit class labels
├── mive-doost-dari?.h5 # Pre-trained model
├── readme.md            # Project documentation
└── *.jpg, *.png         # Sample fruit images
```

## Questions?

If you have questions about contributing, feel free to create an issue with your question.

## License

By contributing to Fruitify, you agree that your contributions will be licensed under the same license as the project.
