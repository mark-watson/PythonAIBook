# Cover Material, Copyright, and License

Copyright 2022-2024 Mark Watson. All rights reserved. This book may be shared using the Creative Commons "share and share alike, no modifications, no commercial reuse" license.

This eBook will be updated occasionally so please periodically check the [leanpub.com web page for this book](https://leanpub.com/pythonai) for updates.

Please visit [my website](http://markwatson.com) and follow me on social media.

# Preface

This book is intended, dear reader, to show you a wide variety of practical AI techniques and examples, and to be a jumping off point when you discover things that interest you or may be useful in your work. A common theme here is covering AI programming tasks that used to be difficult or impossible but are now much simpler using deep learning, or at least possible. I also cover a wide variety of non-deep learning material including a chapter on Symbolic AI that has historic interest and some current practical value.

This book is not intended as a textbook that is to be read start to finish. Probably the most useful material useful is the newest material in Part 3 on Large Language Models (LLMs). Usually the parts or even chapters of this book can be read in any order.

{class: tip}
I try to update my books at least once a year so when purchasing on Leanpub please indicate that you want to be notified when new editions are available. Updates to new editions are free for my Leanpub books.

My career developing AI applications and tools began in 1982. Until the advent of breakthroughs in deep learning around 2010 most of my development work was in Common Lisp, Java, and C++. My language preference changed when I started spending most of my time creating deep learning models. Python has the most tooling, libraries, and frameworks for deep learning so as a practical matter I have adopted Python as a primary programming language. That said I still also heavily use Common Lisp, Haskell, Swift, and Scheme. I recommend not having an "always use one programming language" mindset.

Why this book? Some of what I cover here has already been covered in the Common Lisp, Java, Clojure and Haskell artificial intelligence books I have previously written. My goal here is to prioritize more material on deep learning while still lightly covering classical machine learning, knowledge representation, information gathering, and semantic web/linked data theory and applications. We also cover knowledge representation, including: classic systems like Soar and Prolog, constraint programming, and the practical use of relational and graph data stores. Much of my work involves natural language processing (NLP) and associated techniques for processed unstructured data and we will cover this material in some depth.


Why Python? Python is a very high level language that is easily readable by other programmers. Since Python is one of the most popular programming languages there are many available libraries and frameworks. The best code is code that we don't have to write ourselves as long as third party code is open source so we can read and modify it if needed. Another reason to use Python, that we lean heavily on in this book, is using pre-trained deep learning models that are wrapped into Python packages and libraries.

## About the Author

I have written over 20 books, I have over 50 US patents, and I have worked at interesting companies like Google, Capital One, SAIC, Mind AI, and others. You can read all of my recent books (including this book) for free on my web site [https://markwatson.com](https://markwatson.com). If I had to summarize my career the short take would be that I have had a lot of fun and enjoyed my work. I hope that what you learn here will be both enjoyable and help you in your work.

If you would like to support my work please consider purchasing my books on [Leanpub](https://leanpub.com/u/markwatson) and star my git repositories that you find useful on [GitHub](https://github.com/mark-watson?tab=repositories&q=&type=public). You can also interact with me on social media on [Mastodon](https://mastodon.social/@mark_watson) and [Twitter](https://twitter.com/mark_l_watson).

## Using the Example Code

The example code that I have written for this book is Apache 2 licensed so feel free to reuse it. I also use several existing open source packages and libraries in the examples that use liberal-use licenses (I link GitHub repositories, so check the licenses for applicability in your projects). Most of the deep learning examples and the few "classic" machine learning examples in this book are available as Jupyter notebooks in the **jupyter_notebooks** directory that can be run as-is on [Google Colab](https://colab.research.google.com) (or install Jupyter locally on your laptop) or the equivalent Python source files are in the **deep-learning** directory. One advantage of using Colab is that most of the required libraries are pre-installed.

The examples for this book are in the GitHub repository [https://github.com/mark-watson/PythonPracticalAIBookCode](https://github.com/mark-watson/PythonPracticalAIBookCode).

A few of the examples use APIs from Hugging Face and OpenAI's GPT-3. I assume that you have signed up and have access keys that should be available in the environment variables **HF_API_TOKEN** and **OPENAI_KEY**. If you don't want to sign up for these services I still hope that you enjoy reading the sample code and example output.

{class: warning}
The GitHub repository [https://github.com/mark-watson/PythonPracticalAIBookCode](https://github.com/mark-watson/PythonPracticalAIBookCode) for my code examples will occasionally contain subdirectories containing code not in the current edition of this book but are likely to appear in future editions. These subdirectories contain a file named **NOT_YET_IN_BOOK.md**. I plan on releasing new editions of this book in the future.

I have not written original example code for all of the material in this book. In some cases there are existing libraries for such tasks as recommendation systems and generating images from text where I reference third party examples and discuss how and why you might want to use them.

## Book Cover

I live in Sedona Arizona. I have been fortunate to have visited close to one hundred ancient Native American Indian sites in the Verde Valley. I took the cover picture at one of these sites.

This picture shows me and my wife Carol who helps me with book production.

{width: "50%"}
![Mark and Carol Watson](MarkandCarol.jpeg)

## Acknowledgements

I would like to thank my wife Carol Watson who edits all of my books.

I would like to thank the following readers who reported errors or typos in this book: Ryan O'Connor (typo).
