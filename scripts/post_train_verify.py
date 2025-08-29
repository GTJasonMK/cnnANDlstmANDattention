#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml  # type: ignore


def parse_args():
    ap = argparse.ArgumentParser(description="Verify that each YAML produced a unique exported best model file")
    ap.add_argument("--yaml-dir", default="./yamls", help="YAML directory")
    ap.add_argument("--pattern", default="*.yaml", help="Glob pattern for YAML files")
    ap.add_argument("--export-root", default="./getmodel/bestmodel", help="Unified export_best_dir root")
    ap.add_argument("--out", default="./post_train_verify.txt", help="Report output path")
    return ap.parse_args()


def load_export_dir(yaml_path: Path, default_root: Path) -> str:
    try:
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        d = (((cfg or {}).get("train") or {}).get("checkpoints") or {})
        e = d.get("export_best_dir")
        return e if isinstance(e, str) and e.strip() else str(default_root)
    except Exception:
        return str(default_root)


def safe_filename(name: str) -> str:
    """Apply the same character sanitization as trainer.py"""
    return name.replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '-')


def main():
    args = parse_args()
    ylist = sorted(Path(args.yaml_dir).glob(args.pattern))
    export_root = Path(args.export_root)
    lines: List[str] = []
    lines.append(f"YAML total: {len(ylist)}")
    uniq_files = set()
    missing_map: List[Tuple[str, str]] = []
    multi_map: Dict[str, List[str]] = {}

    for y in ylist:
        stem = y.stem
        safe_stem = safe_filename(stem)  # Apply same sanitization as trainer.py
        exp_dir = load_export_dir(y, export_root)
        pts = sorted([p for p in os.listdir(exp_dir) if p.endswith('.pt')]) if os.path.isdir(exp_dir) else []

        # Enhanced matching logic to handle various naming patterns
        cand = []
        for p in pts:
            # Pattern 1: Direct stem prefix (original naming)
            if p.startswith(f"{safe_stem}__"):
                cand.append(p)
            # Pattern 2: Patched stem prefix (multi_console_train naming)
            elif p.startswith(f"{safe_stem}_patched__"):
                cand.append(p)
            # Pattern 3: Contains stem as component (run_tag prefix cases)
            elif f"__{safe_stem}__" in p:
                cand.append(p)
            # Pattern 4: Fallback - contains stem anywhere (more permissive)
            elif safe_stem in p and len(safe_stem) > 8:  # Only for reasonably long stems to avoid false matches
                cand.append(p)

        # Remove duplicates while preserving order
        cand = list(dict.fromkeys(cand))

        if not cand:
            missing_map.append((stem, exp_dir))
        else:
            # Add unique files to global set
            for c in cand:
                uniq_files.add(os.path.join(exp_dir, c))
            # Check for multiple matches (should be rare with new naming)
            if len(cand) > 1:
                multi_map[stem] = cand

    lines.append(f"Exported unique files: {len(uniq_files)}")
    lines.append(f"YAML with no export match: {len(missing_map)}")
    lines.append(f"YAML with multiple exports (check duplicates): {len(multi_map)}")
    lines.append("")

    if missing_map:
        lines.append("[MISSING_EXPORT]")
        for stem, d in missing_map:
            lines.append(f"- {stem} => {d}")
        lines.append("")

    if multi_map:
        lines.append("[MULTIPLE_EXPORTS]")
        for stem, lst in multi_map.items():
            lines.append(f"- {stem}")
            for p in lst:
                lines.append(f"    - {p}")
        lines.append("")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

