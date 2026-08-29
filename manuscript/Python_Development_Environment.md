# Python Development Environment

All of the examples in this book use the tool **uv** to manage Python library installation and running Python programs.

Here are a few things to consider when setting up your Python development environment:

- Use **uv** as your primary tool for managing Python versions, virtual environments, and package dependencies. It is very fast and replaces older tools like `pip`, `venv`, and `conda`.
- Use Git or another version control system to manage your codebase. I will not cover Git here so [read a good tutorial](https://git-scm.com/docs/gittutorial) if you have not used it before.
- Use an IDE like PyCharm or VSCode, or a text editor like Emacs or Vim, whatever you are comfortable with.
- Keep your environment and dependencies up-to-date.
- Add comments and documentation to your code. Your "future you" will thank you.

## Managing Python with uv

I use **uv** exclusively for managing Python versions and installed libraries. Developed by Astral, `uv` is an extremely fast Python package and project manager written in Rust. It serves as a drop-in replacement for `pip`, `pip-tools`, `pipx`, `poetry`, `pyenv`, `virtualenv`, and more.

Install `uv` by following the instructions on the [official uv website](https://docs.astral.sh/uv/). On macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setting Up a Project

Every source code directory in this book follows the same pattern: initialize a project with `uv init --bare`, add the required libraries with `uv add`, and run scripts with `uv run`. Here is a typical setup:

```bash
# Initialize a new project (creates pyproject.toml)
$ uv init --bare

# Add libraries the project needs
$ uv add scikit-learn pandas

# Run a script — uv creates a virtual environment automatically
$ uv run my_script.py
```

The `uv run` command handles everything: it creates a virtual environment if one does not exist, installs the dependencies listed in `pyproject.toml`, and runs your script. You never need to manually activate a virtual environment.

### Managing Python Versions

`uv` also acts as a Python version manager. If you specify a Python version that is not installed on your system, `uv` will automatically download and install it:

```bash
$ uv python install 3.12
```

### Code Formatting (Optional)

I recommend [black](https://black.readthedocs.io/) for automatic code formatting and [isort](https://pycqa.github.io/isort/) for import sorting:

```bash
uv pip install black isort
black *.py
isort *.py
```
