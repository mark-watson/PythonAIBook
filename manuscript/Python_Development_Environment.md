# Part I - Getting Started

In the next three chapters we setup our Python programming environment, use the Scikit-learn library for one classic machine learning example, and experiment with what is often called Good Old Fashioned Symbolic AI (GOFAI).


# Python Development Environment

I don't use a single development setup for Python. Here I will describe the tools I use and in what contexts I use them.

When you are setting up a Python development environment there are a few things to consider in order to ensure that your environment is set up correctly and is able to run your code correctly. Here are a few pieces of advice to help you get started (references: [getting started](https://www.python.org/about/gettingstarted/), [testing](https://docs.python.org/3/library/unittest.html), [virtual environments](https://docs.python.org/3/tutorial/venv.html), and [packaging](https://packaging.python.org/en/latest/tutorials/managing-dependencies/)):

- Use Git or other version control systems to manage your codebase and keep track of changes. This will make it easier to collaborate with other developers and keep your code organized. I will not cover Git here so [read a good tutorial](https://git-scm.com/docs/gittutorial) if you have not used it before.
- Use a virtual environment to isolate your development environment and dependencies from the rest of your system. This will make it easier to manage your dependencies and avoid conflicts with other software on your system.
- Use pip or another package manager to manage your dependencies and install packages. This will make it easier to install and update packages, and will also help to ensure that you have the correct versions of packages installed.
- For large programs use an IDE such as PyCharm, VSCode or any other to write, run and debug your code. For short Python programs I usually skip using an IDE and instead use Emacs + Python mode.
- Test your code: Be sure to test your code to ensure that it runs correctly and that there are no errors. Use testing frameworks such as unittest, nose or pytest to automate your testing process.
- Keep your environment and dependencies up-to-date, to ensure that you are using the latest versions of packages and that your code runs correctly.
- Add comments and documentation to your code so that other developers (and yourself) can understand what your code is doing and how it works.

When developing using Python, or any other programming language, you will find a lot of advice in books and on the web. While I really enjoy tweaking my development environment to get it "just right," I try to minimize the time I invest in this tuning process. This may also work for you: when you are tired near the end of a workday and you might not be at your best for developing new algorithms or coding then use that time to read about and experiment with your development environment, learn new features of your favorite programming languages, or try a new text editor.

## Managing Python Versions and Libraries

There are several tools for managing Python versions and installed libraries. Here I discuss the tools I use.

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

Google Colab, short for Colaboratory, is a free cloud-based platform for data science and machine learning developed by Google. It allows users to write and execute code in a web-based environment, with support for Jupyter notebooks and integration with other Google services such as Google Drive.

Google Colab lets me get set up for working on machine learning and deep learning projects without having to install Python and dependencies on my laptops or servers. It allows me to juggle projects with different requirements while keeping each project separate.

In technical terms, Colab is a Jupyter notebook environment that runs on a virtual machine in the cloud, with access to powerful hardware such as GPUs and TPUs. The virtual machine is pre-configured with popular data science libraries and tools such as TensorFlow, PyTorch, and scikit-learn, making it easy to get started with machine learning and deep learning projects.

Colab also includes features such as code execution, debugging, and version control, as well as the ability to share and collaborate on notebooks with others. Additionally, it allows you to mount your google drive as a storage, which can be useful for large data sets or models.

Users can also access and run their notebook on local runtime, which would allow them to run the code on their local hardware and not in the cloud, this can be useful when working with large datasets.

Overall, Google Colab is a useful tool for data scientists and machine learning engineers, as it provides a convenient and powerful environment for developing and running code, and for collaboration and sharing of results.

Since 2015 most of my work has been centered around deep learning and Google Colab saves me a lot of time and effort. I pay for Colab Pro ($10/month) to get higher priority for using GPUs and TPUs but this is not strictly necessary.


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
