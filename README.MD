# Nanograd

Nanograd is a minimalistic neural network library inspired by Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd). This project aims to provide a deeper understanding of neural networks and gradient descent, as explored in Karpathy's educational content on basic neural networks.

## Features

- **Simple Neural Network Implementation**: Includes core entities like `Value`, `Neuron`, `Layer`, and `MLP (Multilayer Perceptron)`.
- **Comprehensive Jupyter Notebook**: Annotated guide following the steps of neural network creation and fine-tuning.
- **Test Coverage and Type Annotations**: Ensured with 100% `pytest` coverage and type annotations for robustness, facilitated by `mypy`.
- **Automated Quality Checks**: Integrated GitHub Actions for `mypy` and `pytest` to maintain code quality.
- **Dependency Management**: Utilizes `poetry` for efficient package management.

## Getting Started

### Prerequisites

- Python 3.10 or later
- Poetry package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/raisultan/nanograd.git
   ```
2. Navigate to the project directory:
   ```bash
   cd nanograd
   ```
3. Install dependencies using poetry:
   ```bash
   poetry install --all-extras
   ```

### Running the Tests

To run the tests and ensure everything is set up correctly:

```bash
poetry run pytest
```

### Running mypy

To run static type checking:

```bash
poetry run mypy
```

### Using the Jupyter Notebook

The Jupyter Notebook included in the project provides a guided walkthrough of building and tuning a neural network model.

1. Activate the poetry shell:
   ```bash
   poetry shell
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature.
3. Write your code and ensure it passes all tests and checks.
4. Submit a pull request.

Before submitting, please run `pytest` and `mypy` to ensure your code adheres to the project's standards.

---

This project is a tribute to the educational efforts of Andrej Karpathy and the open-source community. All are welcome to learn, extend, and improve upon it.
```

Feel free to modify this template to better fit your project's specific details or additional sections you might want to include, such as 'Acknowledgments', 'Code Examples', or 'Further Reading'.
