import argparse
import csv
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"

ZSRE_TRAIN_URL = "https://rome.baulab.info/data/dsets/zsre_mend_train.json"
ZSRE_EVAL_URL = "https://rome.baulab.info/data/dsets/zsre_mend_eval.json"
COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"
MQUAKE_REPO = "https://github.com/princeton-nlp/MQuAKE.git"
HALLUEDITBENCH_REPO = "https://github.com/baixianghuang/HalluEditBench.git"


def download_file(url: str, destination: Path, overwrite: bool):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def clone_repo(repo_url: str, destination: Path, overwrite: bool):
    if destination.exists():
        if overwrite:
            shutil.rmtree(destination)
        else:
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(destination)],
        check=True,
    )


def copy_if_exists(source: Path, destination: Path):
    if not source.exists():
        raise FileNotFoundError(f"Expected source file does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def flatten_hallueditbench(source_root: Path, destination: Path):
    candidate_dirs = [
        source_root / "data" / "questions" / "hallucination_final" / "meta_llama_3_8b_instruct",
        source_root / "data" / "questions" / "hallucination_final" / "mistral_7b_instruct_v0.3",
        source_root / "data" / "questions" / "hallucination_final" / "llama_2_7b_chat_hf",
    ]
    selected_dir = next((path for path in candidate_dirs if path.exists()), None)
    if selected_dir is None:
        raise FileNotFoundError(
            "Could not find a supported HalluEditBench question directory under "
            f"{source_root / 'data' / 'questions' / 'hallucination_final'}"
        )

    csv_files = sorted(selected_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {selected_dir}")

    rows = []
    fieldnames = []
    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            local_fields = reader.fieldnames or []
            for name in local_fields:
                if name not in fieldnames:
                    fieldnames.append(name)
            if "source_topic" not in fieldnames:
                fieldnames.append("source_topic")
            if "source_file" not in fieldnames:
                fieldnames.append("source_file")
            topic = csv_path.stem
            for row in reader:
                row = dict(row)
                row["source_topic"] = topic
                row["source_file"] = str(csv_path.relative_to(selected_dir))
                rows.append(row)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "source_dir": str(selected_dir),
        "csv_files": len(csv_files),
        "rows": len(rows),
        "output": str(destination),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT), type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep_sources", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "data_root": str(data_root.resolve()),
        "datasets": {},
    }

    download_file(ZSRE_TRAIN_URL, data_root / "zsre" / "zsre_mend_train.json", args.overwrite)
    download_file(ZSRE_EVAL_URL, data_root / "zsre" / "zsre_mend_eval.json", args.overwrite)
    manifest["datasets"]["ZsRE"] = {
        "train": str((data_root / "zsre" / "zsre_mend_train.json").resolve()),
        "eval": str((data_root / "zsre" / "zsre_mend_eval.json").resolve()),
    }

    download_file(COUNTERFACT_URL, data_root / "counterfact" / "counterfact.json", args.overwrite)
    manifest["datasets"]["CounterFact"] = {
        "main": str((data_root / "counterfact" / "counterfact.json").resolve()),
    }

    downloads_root = data_root / "_downloads"
    mquake_root = downloads_root / "MQuAKE"
    clone_repo(MQUAKE_REPO, mquake_root, args.overwrite)
    copy_if_exists(mquake_root / "datasets" / "MQuAKE-CF.json", data_root / "MQuAKE" / "MQuAKE-CF.json")
    optional_mquake = mquake_root / "datasets" / "MQuAKE-CF-3k-v2.json"
    if optional_mquake.exists():
        copy_if_exists(optional_mquake, data_root / "MQuAKE" / "MQuAKE-CF-3k-v2.json")
    manifest["datasets"]["MQuAKE"] = {
        "main": str((data_root / "MQuAKE" / "MQuAKE-CF.json").resolve()),
        "cf_3k_v2": str((data_root / "MQuAKE" / "MQuAKE-CF-3k-v2.json").resolve()) if optional_mquake.exists() else None,
    }

    hallu_root = downloads_root / "HalluEditBench"
    clone_repo(HALLUEDITBENCH_REPO, hallu_root, args.overwrite)
    hallu_info = flatten_hallueditbench(hallu_root, data_root / "Hallucination" / "hallucination-edit.csv")
    manifest["datasets"]["Hallucination"] = hallu_info

    if not args.keep_sources and downloads_root.exists():
        shutil.rmtree(downloads_root)

    manifest_path = data_root / "dataset_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"manifest_path": str(manifest_path.resolve()), **manifest}, indent=2))


if __name__ == "__main__":
    main()
