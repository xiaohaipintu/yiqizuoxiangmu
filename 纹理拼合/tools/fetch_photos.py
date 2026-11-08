"""从 picsum.photos 抓 100 张 600x600 的真实照片，存到 纹理拼合/照片/ 。"""
import concurrent.futures as cf
import os
import sys
import urllib.request
from pathlib import Path

OUT = Path("纹理拼合/照片")
OUT.mkdir(parents=True, exist_ok=True)
N = 100


def fetch(i: int) -> tuple[int, int, str]:
    dst = OUT / f"{i:03d}.jpg"
    if dst.exists() and dst.stat().st_size > 5000:
        return i, dst.stat().st_size, "skip"
    url = f"https://picsum.photos/seed/{i}/{600}/{600}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                data = r.read()
            if len(data) < 5000:
                raise RuntimeError("too small")
            dst.write_bytes(data)
            return i, len(data), "ok"
        except Exception as exc:                     # noqa: BLE001
            last = str(exc)
    return i, 0, "fail: " + last


def main() -> int:
    ok = skip = fail = 0
    total = 0
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, size, st in ex.map(fetch, range(1, N + 1)):
            if st == "ok":
                ok += 1
                total += size
            elif st == "skip":
                skip += 1
                total += size
            else:
                fail += 1
                print(f"  {i:03d} {st}", flush=True)
            if (ok + skip + fail) % 20 == 0:
                print(f"  进度 {ok + skip + fail}/{N}", flush=True)
    print(f"完成：新下载 {ok}，已存在 {skip}，失败 {fail}，合计 {total / 1024 / 1024:.1f} MB")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
