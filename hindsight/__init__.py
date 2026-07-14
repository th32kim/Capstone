"""Hindsight — Track A reference pipeline (passive AI memory from first-person video).

Import surface is intentionally tiny: the frozen data contract lives in
:mod:`hindsight.contracts` and everything else imports from there. Heavy backends
(OpenCV, torch, faiss, av, spaCy, whisper, webrtcvad, tesseract) are imported lazily
inside the module that needs them so this package imports in a minimal environment.
"""

__version__ = "0.1.0"
