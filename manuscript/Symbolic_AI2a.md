Here is some output:

```python
$ python
>>> from frame import Frame Bookshelf
>>> f1 = Frame()
>>> f2 = Frame("a sub-frame")
>>> f1.add_subframe(f2)
>>> f1.add_number(3.14)
>>> f2.add_subframe(Frame('a sub-sub-frame'))
>>> print(f1)
<Frame Frame:4>
  <Frame "a sub-frame">
    <Frame "a sub-sub-frame">
  <Number 3.14>
>>> bookshelf = BookShelf()
>>> bookshelf.add_frame(f1)
>>> search_results = bookshelf.search_text('sub')
>>> for rs in search_results:
...   print(rs)
... 
<Frame Frame:4>
  <Frame "a sub-frame">
    <Frame "a sub-sub-frame">
  <Number 3.14>
```

I would start with implementing a simple frame library and extend it for the two types of applications that I worked on: Natural Language Processing (NLP) and planning systems.

I no longer use frames preferring the use off the shelf graph databases that we will cover in a later chapter. Graphs can represent a wider range of data representations because frames represent tree structured data and graphs are more general purpose than trees.

## Use Predicate Logic by Calling Swi-Prolog

Please skip this section if you either don't know how to program in Prolog or if you have no interest in learning Prolog. I have a writing project for a book titled Prolog for AI applications that is a work in progress. When that book is released I will add a link here. Before my Python book is released Sheila McIlraith has a [Swi-Prolog tutorial](https://www.cs.toronto.edu/~sheila/324/f05/tuts/swi.pdf) written for her students that is a good starting point and you can use the official [Swi-Prolog manual](https://www.swi-prolog.org/pldoc/doc_for?object=manual) for specific information. I will make this section self-contained if you just want to read the material without writing any Python + Prolog applications.

You can start by reading the documentation for [setting up Swi-Prolog so it can be called from Python](https://www.swi-prolog.org/pldoc/man?section=mqi-python-installation). 
