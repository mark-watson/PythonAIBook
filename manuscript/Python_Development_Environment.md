# Part I - Getting Started

In the next three chapters we setup our Python programming environment, use the Scikit-learn library for one classic machine learning example, and experiment with what is often called Good Old Fashioned Symbolic AI (GOFAI).


# Python Development Environment

I use the tool **uv** to manage Pythonlibrary installation and running Python programs.


When you are setting up a Python development environment there are a few things to consider in order to ensure that your environment is set up correctly and is able to run your code correctly. Here are a few pieces of advice to help you get started (web references: [getting started](https://www.python.org/about/gettingstarted/), [testing](https://docs.python.org/3/library/unittest.html), and [packaging](https://packaging.python.org/en/latest/tutorials/managing-dependencies/)):

- Use Git or other version control systems to manage your codebase and keep track of changes. This will make it easier to collaborate with other developers and keep your code organized. I will not cover Git here so [read a good tutorial](https://git-scm.com/docs/gittutorial) if you have not used it before.
- Use a virtual environment to isolate your development environment and dependencies from the rest of your system. This will make it easier to manage your dependencies and avoid conflicts with other software on your system. **uv** manages per-project virtual environments.
- Use **uv** as your primary tool for managing Python versions, virtual environments, and package dependencies. It is incredibly fast and replaces older tools like `pip`, `venv`, and `conda`.
- You may wish to use IDEs such as PyCharm, VSCode to run and debug your code. I skip using an IDE and instead use Emacs + Python mode.
- Test your code: Be sure to test your code. Use testing frameworks such as unittest, nose or pytest to automate your testing process. The examples in this book are very short programs that are intended to be reused in your projects. I did not add tests for the short examples.
- Keep your environment and dependencies up-to-date, to ensure that you are using the latest versions of packages and that your code runs correctly.
- Add comments and documentation to your code so that other developers (and you!!) can understand what your code is doing and how it works. Even if you are working on personal projects your "future you" will thank you for adding comments when you need to revisit your own code later.

When developing using Python, or any other programming language, you will find a lot of advice in books and on the web. While I really enjoy tweaking my development environment to get it "just right," I try to minimize the time I invest in this tuning process. This may also work for you: when you are tired near the end of a workday and you might not be at your best for developing new algorithms or coding then use that time to read about and experiment with your development environment, learn new features of your favorite programming languages, or try a new text editor.

## Managing Python Versions and Libraries with uv

I use **uv** exclusively for managing Python versions and installed libraries. Developed by Astral, `uv` is an extremely fast Python package and project manager written in Rust. It serves as a drop-in replacement for `pip`, `pip-tools`, `pipx`, `poetry`, `pyenv`, `virtualenv`, and more.

You can install `uv` by following the instructions on the [official uv website](https://docs.astral.sh/uv/). On macOS and Linux, it's as simple as:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Creating Projects and Virtual Environments

With `uv`, you can easily create a virtual environment and install packages in a fraction of the time it takes with other tools. Here is an example of setting up a project environment for the machine learning examples in this book:

```bash
# Create a new virtual environment using a specific Python version
$ uv venv --python 3.10
$ source .venv/bin/activate

# Install libraries extremely fast
$ uv pip install scikit-learn minizinc sparqlwrapper
```

Notice that we use `uv pip install` instead of `pip install` to take advantage of `uv`'s speed. It also automatically detects the active virtual environment.

### Managing Python Versions

`uv` also acts as a Python version manager. If you specify a Python version that isn't installed on your system, `uv` will automatically download and install it for you:

```bash
$ uv python install 3.11
$ uv venv --python 3.11
```

This single tool handles everything I used to do with `Anaconda` and `venv`, but with significantly better performance.


## Code Style

I recommend installing and configuring [black](https://black.readthedocs.io/) and then install [isort](https://pycqa.github.io/isort/). Some Python developers prefer integrating black and isort with their editors or IDEs to reformat Python code on every file save but I prefer using a **Makefile** target called **tidy** to run black and isort on all python source files in a project. Both tools are easily installed using `uv`:

```bash
uv pip install black isort
```

You can also run them manually:

```bash
$ black *.py
reformatted us_states.py

All done!
1 file reformatted, 1 file left unchanged.
$ isort *.py
$
```
