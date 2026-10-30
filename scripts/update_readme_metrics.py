"""Update the generated retrieval-results table in README from metrics JSON."""

import argparse
import json
from pathlib import Path


START = "<!-- retrieval-results:start -->"
END = "<!-- retrieval-results:end -->"


def write_table(metrics_path, readme_path):
    result = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    rows = [("Checkpoint", result["metrics"])]
    rows.extend((name.replace("_", " ").title(), values)
                for name, values in result.get("baselines", {}).items())

    def fmt(value):
        return "—" if value is None or value == "" else f"{value:.4f}"

    lines = [
        START,
        "### Retrieval results",
        "",
        "| Model | Recall@1 | Recall@5 | Recall@10 | Precision@5 | mAP |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in rows:
        lines.append(
            f"| {name} | {fmt(values.get('recall@1'))} | "
            f"{fmt(values.get('recall@5'))} | {fmt(values.get('recall@10'))} | "
            f"{fmt(values.get('precision@5'))} | {fmt(values.get('map'))} |"
        )
    latency = result.get("latency_ms", {})
    lines += [
        "",
        f"Query latency (embed + search): "
        f"{fmt(latency.get('embedding_plus_search_per_image'))} ms/image.",
        "",
        END,
    ]
    table = "\n".join(lines)

    readme_path = Path(readme_path)
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    if START in readme and END in readme:
        before, rest = readme.split(START, 1)
        _, after = rest.split(END, 1)
        readme = before.rstrip() + "\n\n" + table + after
    else:
        readme = readme.rstrip() + "\n\n" + table + "\n"
    readme_path.write_text(readme, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="results/test_metrics.json")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()
    write_table(args.metrics, args.readme)


if __name__ == "__main__":
    main()
