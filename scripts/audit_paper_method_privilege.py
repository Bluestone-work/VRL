"""Audit that paper-method controllers do not read simulator-truth attributes."""
from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path('marl/paper_navigation.py')
CONTROLLER_CLASSES = {'CBFQPFilter', 'MPPIController'}
FORBIDDEN_ATTRIBUTES = {
    'positions_mm', 'edges', 'masses', 'flow_model', 'solution',
    'clot_positions_mm', 'clot_stations', 'field', 'static_c', 'dyn',
}


def audit_source(source=SOURCE):
    tree = ast.parse(Path(source).read_text(), filename=str(source))
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in CONTROLLER_CLASSES:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr in FORBIDDEN_ATTRIBUTES:
                findings.append(f'{node.name}.{child.attr}')
    return sorted(set(findings))


def main():
    findings = audit_source()
    if findings:
        raise SystemExit('forbidden controller inputs: ' + ', '.join(findings))
    print('paper navigation privilege audit: PASS')
    print('allowed runtime inputs: image estimate, registered map geometry, detector tracks, nominal command')


if __name__ == '__main__':
    main()
