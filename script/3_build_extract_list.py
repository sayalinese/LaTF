"""
Step 3: 合并 train/val 注释文件，生成去重后的特征提取列表。
用法:
    python script/3_build_extract_list.py
    python script/3_build_extract_list.py --train annotation/train_v2.txt --val annotation/val_v2.txt --out annotation/extract_v2.txt
"""
import argparse
from pathlib import Path


def build_extract_list(ann_files: list[str], out_path: str) -> None:
    paths: set[str] = set()
    for f in ann_files:
        with open(f, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                # 兼容 tab 分隔和空格分隔的注释格式
                sep = "\t" if "\t" in line else " "
                img_path = line.rsplit(sep, 1)[0]
                paths.add(img_path)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fp:
        for p in sorted(paths):
            fp.write(p + "\t1\n")

    print(f"提取列表已写入: {out}  共 {len(paths)} 张")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成特征提取列表")
    parser.add_argument("--train", default="annotation/train_v2.txt")
    parser.add_argument("--val",   default="annotation/val_v2.txt")
    parser.add_argument("--out",   default="annotation/extract_v2.txt")
    args = parser.parse_args()

    build_extract_list([args.train, args.val], args.out)


if __name__ == "__main__":
    main()
