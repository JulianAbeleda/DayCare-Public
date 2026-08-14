"""DayCare as an app: the chat shell, served locally, XML on the wire.

The interface is `static/designs/` -- the design-canvas exports, kept
verbatim so the canvas can re-export over them. They are the canonical GUI;
`/` opens the chat shell and nothing else competes for that door. Note they
cannot be opened from file:// at all: the runtime fetches its own document
URL, which Chrome refuses for local files, so serving them is what makes
the design runnable rather than a nicety.

DayCare's thesis is that a flat XML manifest is the one inspectable control
plane -- the document a person can open and read *is* the interface, not a
storage detail hidden behind an API. A JSON transport would mean the GUI
renders values that were converted out of XML and back again: two
representations of the same numbers that can silently drift, which is
exactly the class of bug the plan/diff view (daycare/app/plan.py) exists to
catch. So this app speaks the same format the product is about, all the way
to the browser: every response server.py sends is XML, parsed client-side
with DOMParser and serialized with XMLSerializer -- never JSON.parse.

Run with `python -m daycare.app`. sources.py holds the mock/file data,
plan.py the TODO gate and cost estimator, run.py the training-run seam --
server.py only speaks HTTP and owns the XML serialization at that boundary,
so swapping mock values for a real training pipeline touches those three
modules, not this one.
"""
from __future__ import annotations
