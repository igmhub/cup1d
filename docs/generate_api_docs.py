"""Generate lightweight HTML API documentation from Python docstrings.

This script intentionally avoids importing :mod:`cup1d`, so it can run in a
minimal environment without optional scientific dependencies installed. It
parses the package with :mod:`ast` and writes static HTML pages under
``docs/api``.
"""

from __future__ import annotations

import ast
import html
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "cup1d"
OUT_DIR = ROOT / "docs" / "api"


@dataclass
class FunctionDoc:
    """Documentation extracted for a function or method."""

    name: str
    signature: str
    docstring: str


@dataclass
class ClassDoc:
    """Documentation extracted for a class."""

    name: str
    bases: list[str]
    docstring: str
    methods: list[FunctionDoc]


@dataclass
class ModuleDoc:
    """Documentation extracted for a module."""

    module_name: str
    rel_path: Path
    docstring: str
    classes: list[ClassDoc]
    functions: list[FunctionDoc]


def annotation_to_str(node: ast.AST | None) -> str:
    """Return a compact source representation of an annotation."""
    if node is None:
        return ""
    return ast.unparse(node)


def default_to_str(node: ast.AST | None) -> str:
    """Return a compact source representation of a default value."""
    if node is None:
        return ""
    return ast.unparse(node)


def signature_from_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Return a readable function signature."""
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    parts = []

    for arg, default in zip(positional, defaults):
        text = arg.arg
        annotation = annotation_to_str(arg.annotation)
        if annotation:
            text += f": {annotation}"
        default_text = default_to_str(default)
        if default_text:
            text += f" = {default_text}"
        parts.append(text)

    if args.vararg is not None:
        text = f"*{args.vararg.arg}"
        annotation = annotation_to_str(args.vararg.annotation)
        if annotation:
            text += f": {annotation}"
        parts.append(text)
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        text = arg.arg
        annotation = annotation_to_str(arg.annotation)
        if annotation:
            text += f": {annotation}"
        default_text = default_to_str(default)
        if default_text:
            text += f" = {default_text}"
        parts.append(text)

    if args.kwarg is not None:
        text = f"**{args.kwarg.arg}"
        annotation = annotation_to_str(args.kwarg.annotation)
        if annotation:
            text += f": {annotation}"
        parts.append(text)

    returns = annotation_to_str(node.returns)
    suffix = f" -> {returns}" if returns else ""
    return f"{node.name}({', '.join(parts)}){suffix}"


def function_doc(node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionDoc:
    """Extract documentation for a function node."""
    return FunctionDoc(
        name=node.name,
        signature=signature_from_function(node),
        docstring=ast.get_docstring(node) or "",
    )


def class_doc(node: ast.ClassDef) -> ClassDoc:
    """Extract documentation for a class node."""
    methods = [
        function_doc(child)
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not child.name.startswith("_")
    ]
    return ClassDoc(
        name=node.name,
        bases=[ast.unparse(base) for base in node.bases],
        docstring=ast.get_docstring(node) or "",
        methods=methods,
    )


def module_name_from_path(path: Path) -> str:
    """Return dotted module name for a package file path."""
    rel = path.relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)


def parse_module(path: Path) -> ModuleDoc:
    """Parse one module and return extracted documentation."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [class_doc(node) for node in tree.body if isinstance(node, ast.ClassDef)]
    functions = [
        function_doc(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]
    return ModuleDoc(
        module_name=module_name_from_path(path),
        rel_path=path.relative_to(ROOT),
        docstring=ast.get_docstring(tree) or "",
        classes=classes,
        functions=functions,
    )


def iter_package_files() -> list[Path]:
    """Return package files to document."""
    files = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        parts = set(path.parts)
        if "old" in parts or ".ipynb_checkpoints" in parts or "__pycache__" in parts:
            continue
        files.append(path)
    return files


def page_name(module_name: str) -> str:
    """Return output page filename for a module."""
    return module_name.replace(".", "_") + ".html"


def render_docstring(docstring: str) -> str:
    """Render a plain-text docstring as escaped HTML."""
    if not docstring:
        return '<p class="missing">No docstring yet.</p>'
    return f"<pre>{html.escape(docstring)}</pre>"


def render_function(func: FunctionDoc) -> str:
    """Render one function or method."""
    return (
        '<section class="member">'
        f"<h4>{html.escape(func.name)}</h4>"
        f"<code>{html.escape(func.signature)}</code>"
        f"{render_docstring(func.docstring)}"
        "</section>"
    )


def render_module_page(module: ModuleDoc, modules: list[ModuleDoc]) -> str:
    """Render one module page."""
    nav = "\n".join(
        f'<li><a href="{page_name(mod.module_name)}">{html.escape(mod.module_name)}</a></li>'
        for mod in modules
    )
    functions = "\n".join(render_function(func) for func in module.functions)
    classes = []
    for cls in module.classes:
        bases = f"({', '.join(cls.bases)})" if cls.bases else ""
        methods = "\n".join(render_function(method) for method in cls.methods)
        classes.append(
            '<section class="classdoc">'
            f"<h3>{html.escape(cls.name)}{html.escape(bases)}</h3>"
            f"{render_docstring(cls.docstring)}"
            f"{methods}"
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(module.module_name)} - cup1d API</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <aside>
    <h1><a href="index.html">cup1d API</a></h1>
    <nav><ul>{nav}</ul></nav>
  </aside>
  <main>
    <p class="path">{html.escape(str(module.rel_path))}</p>
    <h2>{html.escape(module.module_name)}</h2>
    {render_docstring(module.docstring)}
    <h3>Functions</h3>
    {functions or '<p class="missing">No public functions.</p>'}
    <h3>Classes</h3>
    {''.join(classes) or '<p class="missing">No public classes.</p>'}
  </main>
</body>
</html>
"""


def render_index(modules: list[ModuleDoc]) -> str:
    """Render the API index page."""
    groups: dict[str, list[ModuleDoc]] = {}
    for module in modules:
        parts = module.module_name.split(".")
        group = parts[1] if len(parts) > 1 else "package"
        groups.setdefault(group, []).append(module)

    sections = []
    for group, group_modules in sorted(groups.items()):
        items = "\n".join(
            f'<li><a href="{page_name(mod.module_name)}">{html.escape(mod.module_name)}</a>'
            f'<span>{html.escape(str(mod.rel_path))}</span></li>'
            for mod in group_modules
        )
        sections.append(f"<section><h2>{html.escape(group)}</h2><ul>{items}</ul></section>")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>cup1d API Documentation</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body class="index">
  <main>
    <h1>cup1d API Documentation</h1>
    <p>Static documentation generated from Python docstrings.</p>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def write_styles() -> None:
    """Write the shared stylesheet."""
    (OUT_DIR / "styles.css").write_text(
        """
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --text: #1f2933;
  --muted: #667085;
  --line: #d9dee7;
  --accent: #2457a6;
  --code: #f0f3f8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.5;
}
aside {
  position: fixed;
  inset: 0 auto 0 0;
  width: 320px;
  overflow: auto;
  border-right: 1px solid var(--line);
  background: var(--panel);
  padding: 24px;
}
main {
  max-width: 980px;
  margin-left: 320px;
  padding: 40px 48px;
}
body.index main {
  margin: 0 auto;
}
h1, h2, h3, h4 { line-height: 1.2; }
h1 { margin-top: 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
nav ul, .index ul { list-style: none; padding: 0; }
nav li { margin: 0 0 8px; font-size: 14px; }
.index li {
  display: flex;
  gap: 12px;
  justify-content: space-between;
  border-bottom: 1px solid var(--line);
  padding: 9px 0;
}
.index li span, .path, .missing { color: var(--muted); }
pre {
  white-space: pre-wrap;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 14px 16px;
  overflow: auto;
}
code {
  display: block;
  background: var(--code);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px 12px;
  overflow: auto;
}
.classdoc, .member {
  border-top: 1px solid var(--line);
  padding-top: 18px;
  margin-top: 18px;
}
@media (max-width: 860px) {
  aside {
    position: static;
    width: auto;
    max-height: 280px;
  }
  main {
    margin-left: 0;
    padding: 28px 20px;
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Generate all API documentation pages."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    modules = [parse_module(path) for path in iter_package_files()]
    write_styles()
    (OUT_DIR / "index.html").write_text(render_index(modules), encoding="utf-8")
    for module in modules:
        (OUT_DIR / page_name(module.module_name)).write_text(
            render_module_page(module, modules),
            encoding="utf-8",
        )
    print(f"Wrote {len(modules)} module pages to {OUT_DIR}")


if __name__ == "__main__":
    main()
