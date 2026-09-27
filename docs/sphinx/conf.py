"""Sphinx: документация Foresight — руководства из docs/*.md (MyST) и справочник по коду (autodoc)."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = "Foresight"
author = "Команда Foresight"
copyright = f"{date.today().year}, {author}"  # noqa: A001
language = "ru"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

# обучение GRU и экспорт ONNX (torch) не нужны для справочника: импорты подменяются заглушками
autodoc_mock_imports = ["torch", "onnx", "onnxscript"]
autodoc_default_options = {"members": True, "show-inheritance": True, "member-order": "bysource"}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

myst_enable_extensions = ["colon_fence", "deflist", "tasklist", "linkify"]
myst_heading_anchors = 3
suppress_warnings = ["myst.header", "myst.xref_missing", "misc.highlighting_failure"]

html_theme = "furo"
html_title = "Foresight · документация"
html_theme_options = {
    "source_repository": "https://github.com/DverkuOff/foresight/",
    "source_branch": "main",
    "source_directory": "docs/sphinx/",
}
