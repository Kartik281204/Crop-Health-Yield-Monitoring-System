"""
Download a small, balanced demo subset of the PlantVillage dataset
(via the public GitHub mirror) into data/raw/<ClassName>/.

This is meant to get you training in minutes on a laptop/CPU. For real
results, swap in the full dataset -- see "Scaling up" in the README.

Usage:
    cd scripts
    python download_data.py
"""
import os
import subprocess
import sys
import urllib.request
import urllib.parse
import concurrent.futures

REPO = "spMohanty/PlantVillage-Dataset"
BRANCH = "master"
BASE_RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"

# 4 crops x (healthy + one disease) = a small, balanced starter set.
# Add more entries here (see the full 38-class list in the README) once
# you're ready to scale up.
CLASSES = {
    "Apple___healthy": "Apple_healthy",
    "Apple___Black_rot": "Apple_Black_rot",
    "Potato___healthy": "Potato_healthy",
    "Potato___Early_blight": "Potato_Early_blight",
    "Corn_(maize)___healthy": "Corn_healthy",
    "Corn_(maize)___Common_rust_": "Corn_Common_rust",
    "Tomato___healthy": "Tomato_healthy",
    "Tomato___Late_blight": "Tomato_Late_blight",
}

N_PER_CLASS = 260
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pv_repo")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")


def ensure_repo_metadata():
    """Blobless, no-checkout clone: gets directory/file listings without
    downloading any image content, so we can pick exactly which files we want."""
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
             f"https://github.com/{REPO}.git", REPO_DIR],
            check=True,
        )


def list_files(cls):
    out = subprocess.run(
        ["git", "-C", REPO_DIR, "ls-tree", "HEAD", "--name-only", f"raw/color/{cls}/"],
        capture_output=True, text=True, check=True,
    )
    return sorted(l for l in out.stdout.splitlines() if l.strip())


def download(url, dest):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dataset-fetch"})
        with urllib.request.urlopen(req, timeout=20) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        print(f"FAILED {url}: {e}", file=sys.stderr)
        return False


def main():
    ensure_repo_metadata()

    tasks = []
    for cls, safe_name in CLASSES.items():
        files = list_files(cls)[:N_PER_CLASS]
        class_dir = os.path.join(OUT_DIR, safe_name)
        os.makedirs(class_dir, exist_ok=True)
        for fpath in files:
            fname = os.path.basename(fpath)
            url = BASE_RAW + "/" + urllib.parse.quote(fpath, safe="/")
            dest = os.path.join(class_dir, fname)
            if not os.path.exists(dest):
                tasks.append((url, dest))

    print(f"Downloading {len(tasks)} images across {len(CLASSES)} classes...")
    ok = fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(download, u, d) for u, d in tasks]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            ok += 1 if fut.result() else 0
            if i % 200 == 0:
                print(f"{i}/{len(tasks)} done")

    fail = len(tasks) - ok
    print(f"Done. ok={ok} fail={fail}. Data at {OUT_DIR}")


if __name__ == "__main__":
    main()
