## Soar Cognitive Architecture

[Soar](https://soar.eecs.umich.edu/) is a flexible and general purpose reasoning and knowledge management system for building intelligent software agents. The Soar project is a classic AI tool and has the advantage of being kept up to date. As I write this the [Soar GitHub repositoryt](https://github.com/SoarGroup/Soar) was just updated a few days ago.

I am writing this material many years after my previous use of Soar. My primary reference for preparing the following material is the short paper [Introduction to the Soar Cognitive Architecture](https://arxiv.org/pdf/2205.03854.pdf) by John E. Laird. For self-study you can start at the [Soar Tutorial Page](https://soar.eecs.umich.edu/articles/downloads/soar-suite/228-soar-tutorial-9-6-0) that provides a download for an eight part tutorial in separate PDF files, binary Soar executables for Linux, Windows, and macOS, and all of the code for the tutorials.

I consider Soar to be important because it proposes and implements a general purpose cognitive architecture. A warning to the reader: Soar has a steep learning curve and there are simpler frameworks for solving practical problems. Later we will look at an example from the Soar Tutorial for the classic "blocks world" problem of moving blocks on a table subject to constraints like not being alowed to move a block if it has another object on top of it. Solving this fairly simple problem requires about 400 lines of Soar source code.

### Background Theory

The design goals for the Soar Cognitive Architecture (which I will usually refer to as Soar) is to provide ["fixed structures, mechanisms, and representations"](https://soar.eecs.umich.edu/workshop/30/laird2.pdf) to develop human level behavior across a wide range of tasks. There is a commercial company [Soar.com](https://try.soar.com) that uses Soar for commercial and government projects.

We will cover [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (which I will usually refer to as RL) in a later chapter but there is similar infrastructure supported by both Soar and RL: a simulated environment, data representing the state of the environment, and possible actions that can be performed in the environment that change the state.

There are two main components of the Soar architecture:

- Working Memory - this is the data that specifies the current state of the environment. Actions in the environment change the data in working memory, either by modifiaction, addition, or deletion. At the risk of over-anthropomorphism, consider this like human short term meory.
- Production Memory - this data is a form of production rules where the left-hand side of rules consist of patterns that if matched against worign memory, the the actions on the right-hand side of a rule are executed. Consider these production rules as long-term memory.

Both Soar working memory and production memory are symbolic data. As a contrast, data in RL is numeric, mostly tensors. This symbolic data comprises goals (G), problem spaces (PS), states (S) and operators (O).

![Soar Operator transitioning from one state to another](Soararchitecture-transitions.png)

### Setup Python and Soar Development Environment

It will take you a few minutes to install Soar on your system and create the Python bindings. Start by cloning the GitHub repository and run the install script from the top directory:

```bash
python scons/scons.py sml_python
```

If you want all language bindings replace **sml_python** with **all**. Change directory to the **out** subdirectory and note the directory path. On my system:

```bash
$ pwd
/Users/markw/SOAR/Soar/out
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/markw/SOAR/Soar/out
$ export PYTHONPATH=$PYTHONPATH:/Users/markw/SOAR/Soar/out
```

I will present here a simple example and explain a subset of the capabilities of Soar. When we are done here you can reference a recent paper by Neha Rajan and 2Sunderrajan Srinivasan [Exploring Learning Capability of an Agent in SOAR: Using 8- Queens Problem](https://thescipub.com/pdf/jcssp.2020.642.650.pdf) for a complete example using Soar for cognitive modeling and a more complex example.

### A minimal Soar Tutorial

I am presenting a minimal introduction to Soar and we will later provide an example of Python and Soar interop for the purpose of introducing you to Soar. If this material looks interesting then I encourage you to work through the [Soar Tutorial Page](https://soar.eecs.umich.edu/articles/downloads/soar-suite/228-soar-tutorial-9-6-0).

Soar supports a rule language that uses the highly efficient Rete algorithm (optimized for huge numbers of rules, less optimized for large working memories). Let's look at a sample rule from Chapter 1 (first PDF file) of the Soar tutorial:

```python
sp {hello-world
   (state <s> ^type state)
-->
   (write |Hello World|)
   (halt)
}
```

The token **sp** on line 1 stands for Soar Production. Rules are enclosed in **{** and **}**. The name of this rule is the symbol **hello-world**. In the tutorial you will usually see rule names partitioned using the characters **\*** and **-**. Rules have a "left side" and a "right side", separated by **-->**. If all of the left side patterns match working memory elments then the right-hand side actions are executed.

The following figure is from the Soar tutoral and shows two blocks stacked on top of each other. The bottom block rests on a table:

![From the Soar Tutorial: two stacked blocks sitting on a table](Soarblocks.png)

This figure represents state **s1** that is a root of the graph also containing blocks named **b1** and **b2** as well as the table named **t1**. The blocks and table all have attributes **^color**, **^name**, and **^type**. The blocks also have the optional attribute **^ontop**.

Rule right-hand side actions can modify, delete, or add working memory data. For example, a left-hand side matching the attribute values for block **b1** could modify its **^ontop** attribute from the value **b2** to the table named **t1**.

### Example Soar System With Python Interop

We will use the simplest blocks world example in the Soar Tutorial in our Python interop example. In the examples directories in the Soar Tutorial, this example is spread through eight source files. I have copied them to a single file **Soar/blocks-world/bw.soar** in the GitHub repository for this book.

```python
import Python_sml_ClientInterface as sml

def callback_debug(mid, user_data, agent, message):
    print(message)

if __name__ == "__main__":
    soar_kernel = sml.Kernel.CreateKernelInCurrentThread()
    soar_agent = soar_kernel.CreateAgent("agent")
    soar_agent.RegisterForPrintEvent(sml.smlEVENT_PRINT, callback_debug, None) # no user data
    soar_agent.ExecuteCommandLine("source bw.soar")
    run_result=soar_agent.RunSelf(50)
    soar_kernel.DestroyAgent(soar_agent)
    soar_kernel.Shutdown()
```

Run this example:

```bash

$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/markw/SOAR/Soar/out
$ export PYTHONPATH=$PYTHONPATH:/Users/markw/SOAR/Soar/out
$ python bw.py                                     

     1:    O: O1 (initialize-blocks-world)
Five Blocks World - just move blocks.
The goal is to get EDBCA.
AC
DEB

     2:    O: O8 (move-block)
 Apply O8: move-block(C,table)P10*apply*move-block*internal
A
DEB
C

     3:    O: O7 (move-block)
 Apply O7: move-block(B,table)P10*apply*move-block*internal
DE
B
C
A

     4:    O: O17 (move-block)
 Apply O17: move-block(E,table)P10*apply*move-block*internal
D
B
C
A
E

     5:    O: O25 (move-block)
 Apply O25: move-block(D,E)P10*apply*move-block*internal
B
C
A
ED

     6:    O: O27 (move-block)
 Apply O27: move-block(C,D)P10*apply*move-block*internal
B
EDC
A

     7:    O: O10 (move-block)
 Apply O10: move-block(B,C)P10*apply*move-block*internal
EDCB
A

     8:    O: O11 (move-block)
 Apply O11: move-block(A,B)P10*apply*move-block*internal
EDCBA
Goal Achieved (five blocks).
System halted.
Interrupt received.This Agent halted.
```

TBD


## Constraint Programming with MiniZinc and Python

As with Soar, our excursion into constraint programming will be brief, hopefully enough to introduce you to a new style of programming though a few examples.

You may want to use the [MiniZinc Python](https://minizinc-python.readthedocs.io/en/latest/getting_started.html) documentation as a reference for the Python interface and [The MiniZink Handbook](https://www.minizinc.org/doc-2.6.4/en/index.html) as a reference to the MiniZinc language and its use.

### Installation and Setup for MiniZinc and Python

You need to first install the MiniZinc system. For macOS this can be done with **brew install minizinc** or can be [installed from source code on macOs and Linux](https://www.minizinc.org/doc-2.5.5/en/installation_detailed_linux.html). The Python interface can be installed with **pip install minizinc**.

The following figure shows the MiniZincIDE with simple constraint satisfaction problem:

![MiniZincIDE with simple constraint satisfaction problem](MiniZincIDE.png)

When I installed **minizinc** on macOS with **brew**, the solver **coinbc** was installed automatically so that is what we use here. Here is the MiniZinc source file **test1.mzn**:

```python
int: n;
int: m;
var 1..n: x;
var 1..n: y;
constraint x+y = n;
constraint x*y = m;
```

There are several possible solvers to use with MiniZinc. When I install on macOS using *brew* the solver "coinbc" is available. When I install **sudo apt install minizinc** on Ubuntu Linux, the solver "gecode" is available.

Notice that we don't set values for the constants **n** and **m** as we did when using MiniZincIDE. We instead set them in Python code before calling the solver:

```python
from minizinc import Instance, Model, Solver

coinbc = Solver.lookup("coinbc")

test1 = Model("./test1.mzn")
instance = Instance(coinbc, test1)
instance["n"] = 30
instance["m"] = 200

result = instance.solve()
print(result)
print(result["x"])
print(result["y"])
```

The result is:

```bash
$ python test1.py
Solution(x=20, y=10, _checker='')
20
10
```

Let's look at a more complex example: on the map of the USA, the states neighboring each other are colored differently than their adjoining states. We use integers to represent colors and the mapping of numbers to colors is unimportant. Here is a partial listing of us_states.mzn:

```python
int: nc = 3; %% needs to be 4 to solve this problem

var 1..nc: alabama;
var 1..nc: alaska;
var 1..nc: arizona;
var 1..nc: arkansas;
var 1..nc: california;
 ...
constraint alabama != florida;
constraint alabama != georgia;
constraint alabama != mississippi;
constraint alabama != tennessee;
constraint arizona != california;
constraint arizona != colorado;
constraint arizona != nevada;
constraint arizona != new_mexico;
constraint arizona != utah;
 ...
solve satisfy;
```

The output is:

```bash
 $ minizinc --solver coinbc us_states.mzn
=====UNSATISFIABLE=====
```

So we need more than three colors. Let's try **int: nc = 4;**:

```python
$ minizinc --solver coinbc us_states.mzn
alabama = 2;
alaska = 1;
arizona = 3;
arkansas = 4;
california = 4;
colorado = 4;
connecticut = 2;
delaware = 4;
 ...
```

Here is a Python script **us_states.py** that uses this model and picks out the assigned color indices from the solution object:

```python
from minizinc import Instance, Model, Solver

coinbc = Solver.lookup("coinbc")

model = Model("./us_states.mzn")
instance = Instance(coinbc, model)
instance["nc"] = 4 # solve for a maximum of 4 colors

result = instance.solve()
print(result)
all_states = list(result.__dict__['solution'].__dict__.keys())
all_states.remove('_checker')
print(all_states)
for state in all_states:
    print(f" {state} \t: \t{result[state]}")
```

Here is some of the output:

```bash
$ python us_states.py
Solution(alabama=2, alaska=1, arizona=3, arkansas=4, california=4, colorado=4, connecticut=2, delaware=4, florida=1, georgia=4, hawaii=1, idaho=4, ... ]
 alabama 	: 	2
 alaska 	: 	1
 arizona 	: 	3
 arkansas 	: 	4
 ...
 wisconsin 	: 	1
 wyoming 	: 	3
```

## Good Old Fashioned Symbolic AI Wrapup

As a practical matter almost all of my work in the last ten years used either deep learning or was comprised of semantic web and linked data projects. While the material in this chapter is optional for the modern AI practitioner, I still find using MiniZinc for constraint programming and Prolog to be useful. I included the material for the Soar cognitive architecture because I both find it interesting and I believe the any future development of "real AI" (or AGI) will involve hybrid approaches and there are many good ideas in the Soar implementation.
