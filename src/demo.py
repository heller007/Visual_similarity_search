"""Create a local HTML page showing nearest CIFAR-100 image neighbors."""

import argparse
import base64
from html import escape
from io import BytesIO
from pathlib import Path

from src.data import get_loaders
from src.model import ResNet18
from src.retrieval import embed, search_embeddings
from src.utils import load_checkpoint, resolve_device


def _image_uri(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def create_demo(
    checkpoint_path,
    *,
    query_index=0,
    top_k=5,
    split="test",
    data_root="data",
    split_dir="splits",
    output="results/retrieval_demo.html",
    num_workers=0,
    device="auto",
    download=True,
):
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'.")
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    device = resolve_device(device)
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    config = checkpoint.get("config", {})
    P, K = int(config.get("P", 32)), int(config.get("K", 8))
    loaders = get_loaders(
        seed=int(config.get("seed", 42)),
        data_root=data_root,
        split_dir=split_dir,
        P=P,
        K=K,
        num_workers=num_workers,
        download=download,
    )
    model = ResNet18(d=int(config.get("embedding_dim", 128))).to(device)
    model.load_state_dict(checkpoint["model"])
    embeddings, labels = embed(
        model,
        loaders[split],
        device,
        use_backbone_features=(
            config.get("stage") == "cifar100_classification_pretrain"
        ),
    )
    if not 0 <= query_index < len(embeddings):
        raise IndexError(f"query-index must be in [0, {len(embeddings) - 1}].")

    subset = loaders[split].dataset
    pool = subset.pool
    class_names = pool.datasets[0].classes
    result_indices, distances = search_embeddings(
        embeddings[query_index].to(device),
        embeddings.to(device),
        labels.to(device),
        top_k=top_k,
        exclude_index=query_index,
    )
    result_indices = result_indices.cpu().tolist()
    distances = distances.cpu().tolist()

    query_global_index = int(subset.indices[query_index])
    query_image, _ = pool[query_global_index]
    query_label = int(labels[query_index])
    cards = []
    for rank, (index, distance) in enumerate(zip(result_indices, distances), start=1):
        global_index = int(subset.indices[index])
        image, _ = pool[global_index]
        label = int(labels[index])
        cards.append(
            "<article>"
            f"<img src=\"{_image_uri(image)}\" alt=\"{escape(class_names[label])}\">"
            f"<h2>#{rank} · {escape(class_names[label])}</h2>"
            f"<p>Distance: {distance:.4f} · dataset item: {global_index}</p>"
            "</article>"
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Visual similarity search</title>
  <style>
    body {{ font: 16px system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; color: #18212b; }}
    .query, .results {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }}
    article {{ border: 1px solid #d7dee7; border-radius: 12px; padding: 1rem; }}
    img {{ width: 100%; image-rendering: pixelated; border-radius: 8px; }}
    h2 {{ font-size: 1rem; margin-bottom: .4rem; }} p {{ color: #526170; font-size: .9rem; }}
  </style>
</head>
<body>
  <h1>Visual similarity search</h1>
  <p>Split: {escape(split)} · query item: {query_global_index} · class: {escape(class_names[query_label])}</p>
  <section class="query"><article><img src="{_image_uri(query_image)}" alt="Query image"></article></section>
  <h2>Nearest neighbors</h2>
  <section class="results">{''.join(cards)}</section>
</body>
</html>
"""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--query-index", type=int, default=0, help="Position in the selected split.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--output", default="results/retrieval_demo.html")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    try:
        output = create_demo(
            args.checkpoint,
            query_index=args.query_index,
            top_k=args.top_k,
            split=args.split,
            data_root=args.data_root,
            split_dir=args.split_dir,
            output=args.output,
            num_workers=args.num_workers,
            device=args.device,
            download=not args.no_download,
        )
    except (FileNotFoundError, IndexError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved retrieval gallery to {output.resolve()}")


if __name__ == "__main__":
    main()
