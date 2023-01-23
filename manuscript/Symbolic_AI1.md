# Symbolic AI (Optional Material)

When I started my paid career as an AI practitioner in 1982 my company bought me a Xerox 1108 Lisp Machine and I spent every spare moment I had working through two books by Patrick Winston that I had purchased a few years earlier: "Lisp" and "Artificial Intelligence." This material was mostly what is now called symbolic AI or good old fashioned AI (GOFAI). The material in this chapter is optional for modern AI developers but I recently wrote the Python examples listed below when I was thinking of how different knowledge representation is today compared to 40 years ago. Except for the material using Python + Swi-Prolog, and Python + the MiniZinc constraint satisfaction system there is nothing in this chapter that I would consider using today for work but you might enjoy the examples anyway. After this short chapter we will bear down on deep learning, information organization using RDF and property graph data stores.

I do not implement three examples in this chapter in "pure Python," rather, I use the Python bindings for three well known tools that are implemented in C/C++:

- Swi-Prolog is a Prolog system that has many available libraries for a wide variety of tasks.
- Soar Cognitive Architecture is a flexible and general purpose reasoning and knowledge management system for building intelligent software agents.
- MiniZinc is a powerful Constraint Satisfaction System.

The material in this chapter is optional for the modern AI practitioner but I hope you find it interesting.

We will start with one "pure Python" example in the next section.

## Implementing Frame Data Structures in Python

Most of my learning experiments and AI projects in the early 1980s were built from scratch in Common Lisp and nested frame data structures were a common building block. Here we allow three types of data to be stored in frames:

- Numbers
- Strings
- Other frames

We write a general Python class **Frame** that supports creating frames and converting a frame, including deeply nested frames, into a string representation. We also write a simple Python class **BookShelf** as a container for frames that supports searching for any frames containing a string value.

```python
# Implement Lisp-like frames in Python

class Frame():
    frame_counter = 0
    def __init__(self, name = ''):
        Frame.frame_counter += 1
        self.objects = []
        self.depth = 0
        if (len(name)) == 0:
            self.name = f"Frame:{Frame.frame_counter}"
        else:
            self.name = f'"{name}"'

    def add_subframe(self, a_frame):
        a_frame.depth = self.depth + 1
        self.objects.append(a_frame)

    def add_number(self, a_number):
        self.objects.append(a_number)

    def add_string(self, a_string):
        self.objects.append(a_string)

    def __str__(self):
        indent = " " * self.depth * 2
        ret = indent + f"<Frame {self.name}>\n"
        for frm in self.objects:
            if isinstance(frm, (int, float)):
                ret = ret + indent + '  ' + f"<Number {frm}>\n"
            if isinstance(frm, str):
                ret = ret + indent + '  ' + f'<String "{frm}">\n'
            if isinstance(frm, Frame):
                ret = ret + frm.__str__()
        return ret

f1 = Frame()
f2 = Frame("a sub-frame")
f1.add_subframe(f2)
f1.add_number(3.14)
f2.add_string("a string")
print(f1)
f2.add_subframe(Frame('a sub-sub-frame'))
print(f1)

class BookShelf():

    def __init__(self, name = ''):
        self.frames = []
    
    def add_frame(self, a_frame):
        self.frames.append(a_frame)
    
    def search_text(self, search_string):
        ret = []
        for frm in self.frames:
            if frm.__str__().index(search_string):
                ret.append(frm)
        return ret
    
bookshelf = BookShelf()
bookshelf.add_frame(f1)
search_results = bookshelf.search_text('sub')
print("Search results: all frames containing 'sub':")
for rs in search_results:
    print(rs)
```

