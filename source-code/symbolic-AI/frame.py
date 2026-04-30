# frame.py - Lisp-like frame data structures in Python
#
# Frames are a classic symbolic AI knowledge representation from the 1980s.
# A frame is a named container that can hold numbers, strings, and nested
# subframes, forming a tree structure. The BookShelf class provides a
# simple container for searching across multiple frames.

class Frame():
    """A recursive knowledge representation structure inspired by Lisp frames.
    
    Each frame has a name and can contain numbers, strings, and child frames.
    The nesting depth is tracked for indented pretty-printing.
    """
    frame_counter = 0  # class-level counter for auto-naming unnamed frames

    def __init__(self, name = ''):
        Frame.frame_counter += 1
        self.objects = []  # mixed list of numbers, strings, and child Frames
        self.depth = 0     # nesting level (0 = top-level)
        if (len(name)) == 0:
            self.name = f"Frame:{Frame.frame_counter}"
        else:
            self.name = f'"{name}"'

    def add_subframe(self, a_frame):
        """Nest a child frame inside this frame, adjusting its depth."""
        a_frame.depth = self.depth + 1
        self.objects.append(a_frame)

    def add_number(self, a_number):
        """Store a numeric value in this frame."""
        self.objects.append(a_number)

    def add_string(self, a_string):
        """Store a string value in this frame."""
        self.objects.append(a_string)

    def __str__(self):
        """Recursively render the frame tree with indentation showing depth."""
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


class BookShelf():
    """A container that holds multiple frames and supports text search.
    
    Searching converts each frame to its string representation and
    checks whether the search term appears anywhere in the tree.
    """

    def __init__(self, name = ''):
        self.frames = []
    
    def add_frame(self, a_frame):
        self.frames.append(a_frame)
    
    def search_text(self, search_string):
        """Return all frames whose string representation contains search_string."""
        ret = []
        for frm in self.frames:
            if frm.__str__().index(search_string):
                ret.append(frm)
        return ret


# --- Demo: build a small frame tree and search it ---

f1 = Frame()
f2 = Frame("a sub-frame")
f1.add_subframe(f2)
f1.add_number(3.14)
f2.add_string("a string")
print(f1)
f2.add_subframe(Frame('a sub-sub-frame'))
print(f1)

bookshelf = BookShelf()
bookshelf.add_frame(f1)
search_results = bookshelf.search_text('sub')
print("Search results: all frames containing 'sub':")
for rs in search_results:
    print(rs)
