import sys

from transformers import AutoTokenizer


def main():
    model_path = sys.argv[1]
    max_length = int(sys.argv[2])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    lines = []
    words = []
    for _ in sys.stdin:
        _ = _.rstrip("\n")
        lines.append(_)
        if _:
            words.append(_.split("\t")[1])
            continue
        text = " ".join(words)
        tokens = tokenizer.enocde(text)
        if len(tokens) <= max_length:
            print(*lines, sep="\n")
        else:
            print(*lines, sep="\n", file=sys.stderr)
        lines = []
        words = []


if __name__ == "__main__":
    main()
