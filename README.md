# Splatwizard: Gaussian Splatting Compression Toolkit


Splatwizard is a one-stop toolkit designed for researching 3DGS compression, dedicated to accelerating community exploration in this field.
With flexible API design, you can easily combine advanced components from state-of-the-art models to build customized models.
Key features of the project include:

- **Easy-to-adapt evaluation framework** for new models
- **Comprehensive evaluation metrics**
- **Multiple baseline models** included for comparison

## Installation
Splatwizard can be installed via pip
```bash
pip install splatwizard
```

Since the installation process requires significant time to precompile all components, you can monitor the installation progress using the `--verbose` option.
```bash
pip install splatwizard --verbose
```

Additionally, pip compiles components in an isolated environment by default. You can use `--no-build-isolation` to perform the compilation directly in the current environment.
```bash
pip install splatwizard --verbose --no-build-isolation
```