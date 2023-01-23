# Part I - Getting Started

In the next three chapters we setup our Python programming environment, use the scikit-learn library for one classic machine learning example, and experiment with what is often called Good Old Fashioned Symbolic AI (GOFAI).


# Python Development Environment

I don't use a single development setup for Python. Here I will describe the tools I use and in what contexts I use them.

When developing using Python, or any other programming language, you will find a lot of advice in books and on the web. While I really enjoy tweaking my development environment to get it "just right," I try to minimize the time I invest in this tuning process. This may also work for you: when you are tired near the end of a workday and you might not be at your best for developing new algorithms or coding then use that time to read about and experiment with your development environment, learn new features of your favorite programming languages, or try a new text editor.

## Managing Python Versions and Libraries

There are several tools for managing Python versions and installed libraies. Here I discuss the tools I use.

### Anaconda

The Anaconda Corporation maintains open source tools and provides enterprise support. I use their MiniConda package manager to install different environments containing specific versions of Python and specific libraries for individual projects.

[Pause and read the online documentation.](https://docs.anaconda.com/anaconda/user-guide/index.html)

I use Anaconda for managing complex libraries and frameworks for machine learning and deep learning that often have different requirements if a GPU is available. Installing packages takes longer with Anaconda compared to other options like **pyenv** but Anaconda's more strict package dependency analysis ends up saving me time when running deep learning on my laptop or my servers.

Here is an example for setting up the environment for the examples in the next chapter:

```bash
conda create -n ml python=3.10
conda activate ml
conda install scikit-learn
```

### Google Colab

Okay, so Colab is not a package manager but Colab (a cloud version of Jupyter Notebooks that provide free and low cost GPUs) has up to date deep learning libraries and frameworks pre-installed.

Since 2015 most of my work has been centered around deep learning and Google Colab saves me a lot of time and effort.


### venv

**venv** creates a subdirectory for a project's installed Python executable files and libraries. You can name this subdirectory whatever you like (I like the name "venv"):

I sometimes use [venv](https://docs.python.org/3/library/venv.html) for my Python related work that is not machine learning or deep learning. **venv** is used to create isolated virutal Python development environments for each project, for example:

```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install minizinc sparqlwrapper
```

We activate the newly created virtual environment in line 2 and install two Python packages into the new virtual environment in line 3.


## Editors and IDEs

I assume that you already have a favorite Python editor or IDE. Here I will just mention what I use and why I choose different tools for different types of tasks.

I like VSCode when I am working on a large Python project that has many source files scattered over many directories and subdirectories because I find navigation and code browsing is fast and easy.

I prefer Emacs with **python-mode** for most of my work that consists of smaller projects where all code is in either a single or just a few source files. For larger projects I sometimes use Emacs with *treemacs* for rapid navigation between files. I especially like the interactive coding style with Emacs and **emacs-mode** because it is simple to load an entire file, re-load a changed function definition, etc. and work interactively in the provided REPL.

I sometimes use the PyCharm IDE. PyCharm also has excellent rapid code navigation support and is generally full featured. Until about a year ago I used PyCharm for most of my development work but lately I have gone back to using Emacs and **python-mode** as my main daily driver.


## Code Style

I recommend installing and configuring [black](https://black.readthedocs.io/) and then install [isort](https://pycqa.github.io/isort/). Some Python developers prefer integrating black and isort with their editors or IDEs to reformat Python code on every file save but I prefer using a **Makefile** target called **tidy** to run black and isort on all python source files in a project. Both tools are easily installed:

```bash
pip install black
pip install isort
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
