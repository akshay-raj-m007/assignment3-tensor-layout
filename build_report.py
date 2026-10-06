"""Refresh measured content in editable report.tex and compile its PDF."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                    "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def replace_block(source, name, content):
    pattern = rf"(?m)^% BEGIN GENERATED {name}\n.*?^% END GENERATED {name}$"
    replacement = f"% BEGIN GENERATED {name}\n{content}\n% END GENERATED {name}"
    result, count = re.subn(pattern, lambda _: replacement, source, flags=re.DOTALL)
    if count != 1:
        raise ValueError(f"Expected exactly one generated block: {name}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results.csv")
    parser.add_argument("--tex", type=Path, default=ROOT / "report.tex")
    parser.add_argument("--student", help="Update name; otherwise preserve the LaTeX value")
    parser.add_argument("--roll", help="Update roll number; otherwise preserve the LaTeX value")
    parser.add_argument("--compiler", help="Path/name of tectonic, pdflatex or xelatex")
    parser.add_argument("--tex-only", action="store_true", help="Refresh LaTeX without compiling")
    args = parser.parse_args()
    with args.results.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10 or any(r["verified"] != "PASS" for r in rows):
        raise ValueError("Report requires the complete verified ten-case suite")
    if any(float(r[key]) != 0 for r in rows for key in ("max_abs_error", "mae")):
        raise ValueError("Investigate nonzero errors before generating this report")
    digest = hashlib.sha256(args.results.read_bytes()).hexdigest()
    manifest = json.loads((args.results.parent / "run_manifest.json").read_text())
    if digest != manifest["results_sha256"]:
        raise ValueError("CSV differs from the measured run manifest")
    source = args.tex.read_text(encoding="utf-8")
    metadata = "\n".join([
        rf"\newcommand{{\ResultsSHA}}{{{digest}}}",
        rf"\newcommand{{\ElementCount}}{{{sum(int(r['elements']) for r in rows):,}}}",
        rf"\newcommand{{\PythonVersion}}{{{tex_escape(manifest['python'])}}}",
        rf"\newcommand{{\NumpyVersion}}{{{tex_escape(manifest['numpy'])}}}"])
    source = replace_block(source, "METADATA", metadata)
    keys = ("input", "B", "C", "H", "W", "max_abs_error", "mae")
    table = "\n".join(" & ".join(tex_escape(r[k]) for k in keys) + r" \\" for r in rows)
    source = replace_block(source, "RESULTS", table)
    for macro, value in (("StudentName", args.student), ("RollNumber", args.roll)):
        if value is not None:
            pattern = rf"(?m)^\\newcommand\{{\\{macro}\}}\{{.*\}}$"
            source, count = re.subn(pattern, lambda _: rf"\newcommand{{\{macro}}}{{{tex_escape(value)}}}", source)
            if count != 1:
                raise ValueError(f"Missing identity macro: {macro}")
    args.tex.write_text(source, encoding="utf-8", newline="\n")
    print(f"Updated {args.tex.name}: 10 measured rows; CSV SHA-256 {digest}", flush=True)
    if args.tex_only:
        return
    local_compiler = ROOT / ".tools" / "tectonic.exe"
    compiler = args.compiler or shutil.which("tectonic") or (str(local_compiler) if local_compiler.exists() else None) or shutil.which("pdflatex") or shutil.which("xelatex")
    if not compiler:
        parser.error("LaTeX compiler required: install Tectonic/TeX Live/MiKTeX, or use --tex-only and compile report.tex in Overleaf")
    compiler_name = Path(compiler).name.lower()
    tex = args.tex.resolve()
    if "tectonic" in compiler_name:
        command = [compiler, "--keep-logs", "--untrusted", str(tex)]
        passes = 1
    elif any(name in compiler_name for name in ("pdflatex", "xelatex")):
        command = [compiler, "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", tex.name]
        passes = 2
    else:
        parser.error("Supported compilers: tectonic, pdflatex, xelatex")
    logs = []
    for _ in range(passes):
        result = subprocess.run(command, cwd=tex.parent, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        logs.append(result.stdout)
        print(result.stdout, end="", flush=True)
        (tex.parent / "latex_build_output.txt").write_text("\n".join(logs), encoding="utf-8", newline="\n")
        result.check_returncode()
    if not tex.with_suffix(".pdf").is_file():
        raise RuntimeError("Compiler did not produce the corresponding PDF")
    print(f"Compiled {tex.name} -> {tex.with_suffix('.pdf').name}")


if __name__ == "__main__":
    main()
