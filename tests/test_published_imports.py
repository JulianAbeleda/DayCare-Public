"""Every `daycare.*` import in the published tree (module level or inside a function) names a published module."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _exists(name: str) -> bool:
    path = ROOT.joinpath(*name.split('.'))
    return path.with_suffix('.py').is_file() or (path / '__init__.py').is_file()


def _imports(path: Path):
    module = '.'.join(path.relative_to(ROOT).with_suffix('').parts)
    package = module if path.name == '__init__.py' else module.rsplit('.', 1)[0]
    package = package.removesuffix('.__init__')
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield node.lineno, [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ''
            if node.level:
                parts = package.split('.')[:len(package.split('.')) - node.level + 1]
                base = '.'.join(parts + ([node.module] if node.module else []))
            # `from pkg import name`: name is a submodule or an attribute of pkg; either way pkg must exist
            yield node.lineno, [base] + [f'{base}.{a.name}' for a in node.names if _exists(f'{base}.{a.name}')]


def test_every_daycare_import_resolves():
    missing = []
    for path in sorted([*ROOT.joinpath('daycare').rglob('*.py'), *ROOT.joinpath('tests').rglob('*.py')]):
        if 'vendor' in path.parts:
            continue
        for line, names in _imports(path):
            missing += [f'{path.relative_to(ROOT)}:{line}: {n}' for n in names
                        if n.split('.')[0] == 'daycare' and not _exists(n)]
    assert not missing, '\n'.join(missing)
