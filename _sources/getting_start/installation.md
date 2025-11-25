# Installation

SplatWizard is available on pypi, and can be installed with pip:
```shell
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