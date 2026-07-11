"""Export the S1 complete streets-layer schema as a CSV supplementary table.

Scientific Data asks for tables that exceed one A4 page to be supplied as
csv/xlsx Supplementary Tables rather than embedded in the manuscript. This
script parses the S1 longtable in supplementary_sections.tex (the single
source of truth) and writes outputs/tables/supplementary_table_1_streets_schema.csv.
"""

import csv
import re
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
SOURCE = PAPER_DIR / "supplementary_sections.tex"
OUT = PAPER_DIR / "outputs" / "tables" / "supplementary_table_1_streets_schema.csv"

SKIP_PREFIXES = (
    "\\caption",
    "\\toprule",
    "\\midrule",
    "\\bottomrule",
    "\\endfirsthead",
    "\\endhead",
    "\\endfoot",
    "\\endlastfoot",
    "\\textbf{Attribute}",
)


def clean(cell: str) -> str:
    cell = cell.replace("\\\\", "").strip()
    cell = re.sub(r"\\texttt\{([^}]*)\}", r"\1", cell)
    cell = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
    cell = re.sub(r"\\emph\{([^}]*)\}", r"\1", cell)
    cell = cell.replace("\\textsuperscript{2}", "^2").replace("\\textsuperscript{3}", "^3")
    cell = cell.replace("\\_", "_").replace("\\{", "{").replace("\\}", "}").replace("\\%", "%")
    cell = cell.replace("\\ldots", "...").replace("$", "")
    cell = re.sub(r"\\text\{([^}]*)\}", r"\1", cell)
    cell = re.sub(r"\s+", " ", cell)
    return cell.strip()


def main() -> None:
    text = SOURCE.read_text()
    start = text.index("\\begin{longtable}")
    end = text.index("\\end{longtable}")
    block = text[start:end]

    rows = []
    group = ""
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if any(line.startswith(p) for p in SKIP_PREFIXES):
            continue
        multi = re.match(r"\\multicolumn\{4\}\{l\}\{\\textbf\{(.+)\}\}", line)
        if multi:
            group = clean(multi.group(1))
            continue
        if "\\multicolumn" in line or "&" not in line:
            continue
        cells = [clean(c) for c in line.split("&")]
        if len(cells) != 4 or cells[0] in ("", "Attribute"):
            continue
        rows.append([group, *cells])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "attribute", "type", "description", "source"])
        writer.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} attribute rows)")


if __name__ == "__main__":
    main()
