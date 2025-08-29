#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml  # type: ignore

try:
    import torch  # type: ignore
except Exception:
    torch = None

EXPORT_RE = re.compile(r"(?:Export(?:ed)?\s+best.*?:\s*)(.*\.pt)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Scan YAMLs and report which ones did not produce an exported best model",
    )
    ap.add_argument("--yaml-dir", default="./yamls", help="Directory containing YAML files")
    ap.add_argument("--runs-root", default="./runs_all", help="Root directory where each YAML gets a subdir <stem>")
    ap.add_argument("--pattern", default="*.yaml", help="Glob pattern to match YAML files")
    ap.add_argument(
        "--report",
        default="./find_untrained_report.txt",
        help="Path to write a human-readable report",
    )
    ap.add_argument(
        "--csv",
        default="./find_untrained_report.csv",
        help="Optional CSV with details (stem,status,log_path,export_dir,export_pt)",
    )
    return ap.parse_args()


def load_yaml_export_dir(yaml_path: Path) -> Optional[str]:
    try:
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        d = (((cfg or {}).get("train") or {}).get("checkpoints") or {})
        exp = d.get("export_best_dir")
        if isinstance(exp, str) and exp.strip():
            return exp.strip()
    except Exception:
        pass
    return None


def find_export_from_log(log_path: Path) -> Optional[str]:
    try:
        txt = log_path.read_text(errors="ignore")
    except Exception:
        return None
    m = None
    for m in EXPORT_RE.finditer(txt):
        # use the last one if multiple
        pass
    return m.group(1).strip() if m else None


def scan(yaml_dir: Path, runs_root: Path, pattern: str) -> Tuple[Dict, List[Dict]]:
    rows: List[Dict] = []
    summary = {
        "total_yaml": 0,
        "missing_log": 0,
        "no_export_line": 0,
        "export_line_missing_file": 0,
        "export_dir_has_pt": 0,
        "export_ok": 0,
        "export_invalid_ckpt": 0,
    }
    yfiles = sorted(yaml_dir.glob(pattern))
    summary["total_yaml"] = len(yfiles)

    for y in yfiles:
        stem = y.stem
        run_dir = runs_root / stem
        log_path = run_dir / "training.log"
        export_dir_cfg = load_yaml_export_dir(y)

        status = ""
        export_pt_from_log: Optional[str] = None
        export_ok = False

        if not log_path.exists():
            status = "missing_log"
            summary["missing_log"] += 1
        else:
            export_pt_from_log = find_export_from_log(log_path)
            if export_pt_from_log:
                export_ok = os.path.exists(export_pt_from_log)
                if export_ok:
                    status = "export_ok"
                    summary["export_ok"] += 1
                else:
                    status = "export_line_missing_file"
                    summary["export_line_missing_file"] += 1
            else:
                status = "no_export_line"
                summary["no_export_line"] += 1

        # Fallback: if we have an export dir in YAML and it contains any pt, mark and try to validate
        export_dir_has_pt = False
        export_pt_found: Optional[str] = None
        if not export_ok and export_dir_cfg and os.path.isdir(export_dir_cfg):
            try:
                for fn in os.listdir(export_dir_cfg):
                    if fn.endswith('.pt'):
                        export_dir_has_pt = True
                        export_pt_found = str(Path(export_dir_cfg) / fn)
                        break
            except Exception:
                export_dir_has_pt = False
            if export_dir_has_pt and status in ("no_export_line", "export_line_missing_file", "missing_log"):
                # 验证 ckpt 是否可读（可选）
                valid = True
                if torch is not None and export_pt_found:
                    try:
                        torch.load(export_pt_found, map_location='cpu')
                    except Exception:
                        valid = False
                if valid:
                    status = "export_dir_has_pt"
                    summary["export_dir_has_pt"] += 1
                else:
                    status = "export_invalid_ckpt"
                    summary["export_invalid_ckpt"] += 1

        rows.append(
            {
                "stem": stem,
                "yaml_path": str(y),
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "export_dir_cfg": export_dir_cfg or "",
                "export_pt_from_log": export_pt_from_log or "",
                "status": status,
            }
        )
    return summary, rows


def write_reports(summary: Dict, rows: List[Dict], report_path: Path, csv_path: Path) -> None:
    # Text report
    lines: List[str] = []
    lines.append("=== Training/Export Coverage Report ===")
    for k in [
        "total_yaml",
        "missing_log",
        "no_export_line",
        "export_line_missing_file",
        "export_dir_has_pt",
        "export_invalid_ckpt",
        "export_ok",
    ]:
        lines.append(f"{k}: {summary.get(k, 0)}")
    lines.append("")

    def dump(title: str, filt: Optional[str]):
        lines.append(f"[{title}]")
        for r in rows:
            if filt is None or r["status"] == filt:
                lines.append(f"- {r['stem']}  ({r['status']})  log={r['log_path']}  export_dir={r['export_dir_cfg']}  export_pt={r['export_pt_from_log']}")
        lines.append("")

    dump("MISSING_LOG", "missing_log")
    dump("NO_EXPORT_LINE", "no_export_line")
    dump("EXPORT_LINE_MISSING_FILE", "export_line_missing_file")
    dump("EXPORT_DIR_HAS_PT (fallback treated as ok)", "export_dir_has_pt")
    dump("EXPORT_OK", "export_ok")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    # CSV
    try:
        import csv

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stem", "status", "yaml_path", "log_path", "export_dir_cfg", "export_pt_from_log"])
            for r in rows:
                w.writerow([
                    r["stem"],
                    r["status"],
                    r["yaml_path"],
                    r["log_path"],
                    r["export_dir_cfg"],
                    r["export_pt_from_log"],
                ])
    except Exception:
        pass


def main():
    args = parse_args()
    yaml_dir = Path(args.yaml_dir)
    runs_root = Path(args.runs_root)
    report_path = Path(args.report)
    csv_path = Path(args.csv)

    if not yaml_dir.exists():
        raise SystemExit(f"YAML directory not found: {yaml_dir}")

    summary, rows = scan(yaml_dir, runs_root, args.pattern)
    write_reports(summary, rows, report_path, csv_path)

    # 输出未训练/未导出列表，方便下一步补训
    missing = [r for r in rows if r["status"] in ("missing_log", "no_export_line", "export_line_missing_file")]
    missing_txt = Path("./missing_train.txt")
    missing_txt.write_text("\n".join(sorted(set(f"{m['stem']}.yaml" for m in missing))), encoding="utf-8")
    print(f"[INFO] Total YAML: {summary['total_yaml']}, missing/needs_attention: {len(missing)}")
    print(f"[INFO] Wrote report: {report_path}")
    print(f"[INFO] Wrote CSV: {csv_path}")
    print(f"[INFO] Wrote missing list: {missing_txt}")


if __name__ == "__main__":
    main()

