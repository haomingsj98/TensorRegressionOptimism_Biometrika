"""Draw the fixed 10,000-image UTKFace subsample, resize it to 64 x 64 x 3 and store it.

    python prepare_utkface.py --images <path to the UTKFace folder of *.jpg files>

UTKFace (Zhang, Song and Qi, 2017; non-commercial research use only) contains 23,708 aligned
and cropped faces named [age]_[gender]_[race]_[datetime].jpg.chip.jpg. A simple random sample
of --n images is drawn with --seed, converted to RGB and resized with Lanczos filtering.

Writes data/utkface_64.npz with X (uint8, n x 64 x 64 x 3), age, gender, race (-1 if
missing), files, pool and seed.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def parse(name):
    parts = name.split(".")[0].split("_")
    age = int(parts[0])
    gender = int(parts[1]) if len(parts) == 4 else -1
    race = int(parts[2]) if len(parts) == 4 else -1
    return age, gender, race


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2015)
    p.add_argument("--output", default=None, help="default data/utkface_{size}.npz")
    args = p.parse_args()
    pool = sorted(f for f in Path(args.images).iterdir() if f.name.lower().endswith(".jpg"))
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, args.n]))
    files = [pool[i] for i in np.sort(rng.choice(len(pool), size=args.n, replace=False))]
    X, meta = [], []
    for f in files:
        meta.append(parse(f.name))
        with Image.open(f) as im:
            X.append(np.asarray(im.convert("RGB").resize((args.size, args.size),
                                                         Image.Resampling.LANCZOS)))
    age, gender, race = (np.array(v) for v in zip(*meta))
    out = Path(args.output) if args.output else ROOT / "data" / f"utkface_{args.size}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=np.stack(X), age=age, gender=gender, race=race,
                        files=np.array([f.name for f in files]), pool=len(pool), seed=args.seed)
    print(f"{len(files)} of {len(pool)} images, age {age.min()}-{age.max()} -> {out}")


if __name__ == "__main__":
    main()
