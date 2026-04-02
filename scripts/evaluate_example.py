import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, required=True)
    args = parser.parse_args()

    path = Path(args.metrics)
    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
