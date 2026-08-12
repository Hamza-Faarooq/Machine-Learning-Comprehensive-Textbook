# Contributing to the ML Textbook Companion

Thank you for your interest in contributing to the official companion repository for **Machine Learning: A Comprehensive Textbook**.

## Code Quality Standards
To maintain the rigor of this educational resource, please adhere to the following standards:

1. **PEP 8 Compliance**: All Python scripts must follow PEP 8 style guidelines.
2. **Type Hinting**: Use Python type hints for all function arguments and return values.
3. **Docstrings**: Include descriptive docstrings detailing mathematical equations where applicable.
4. **Reproducibility**: Always set a random seed (e.g., `np.random.seed(42)`) in notebooks and scripts to guarantee reproducible outputs.
5. **No Black-Box Libraries for Fundamentals**: If a chapter focuses on learning the math behind an algorithm, do not use `scikit-learn` or `PyTorch` in the "from-scratch" implementations.

## How to Submit a Pull Request
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a Pull Request.
