"""Run the chain of scripts in the order given by ORDER.csv.

    python3 run_all.py                    the whole chain
    python3 run_all.py --da 12 --a 30     steps 12 to 30 of ORDER.csv
    python3 run_all.py --elenco           print the steps without running them

Environment variables (all optional):
    CLUSTERING_DATA   folder of the PPMI data (default: data/ next to this file)
    CLUSTERING_BASE   working directory; receives outputs/, figures/, logs/ (default: this folder)
    N_PROCESSES        parallel processes for the scripts that use them

Step 0 prepares the data cache. Before script 104 the two branches with the alternative
DAT measures are built (02, 03, 08 with DAT_SET=mean and DAT_SET=striatum) in dat_variants/.
The chain stops at the first script that exits with an error; each script has its own log in logs/.
"""
import argparse, csv, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("CLUSTERING_BASE", HERE)
LOGS = os.path.join(BASE, "logs")


def path(name):
    if name.startswith("figure/"):
        return os.path.join(HERE, "figure", name.split("/")[1] + ".py")
    return os.path.join(HERE, "scripts", name + ".py")


def launch(name, env=None, arguments=(), label=None):
    for d in (LOGS, os.path.join(BASE, "outputs"), os.path.join(BASE, "figures")):
        os.makedirs(d, exist_ok=True)
    e = dict(os.environ, CLUSTERING_BASE=BASE, **(env or {}))
    for d in ("outputs", "figures"):                       # in the branches of the variants as well
        os.makedirs(os.path.join(e["CLUSTERING_BASE"], d), exist_ok=True)
    log = os.path.join(LOGS, (label or name.replace("/", "_")) + ".log")
    t0 = time.time()
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-B", path(name), *arguments], env=e, stdout=fh, stderr=subprocess.STDOUT,
                           cwd=os.path.dirname(path(name)))
    print(f'  {label or name:48s} exit {r.returncode}  {time.time() - t0:7.0f} s', flush=True)
    if r.returncode != 0:
        sys.exit(f'STOPPED: {name} exited with an error; see {log}')


def dat_variants():
    """The two branches with the alternative DAT measures, later read by 104."""
    for measure in ("mean", "striatum"):
        branch = os.path.join(BASE, "dat_variants", measure)
        os.makedirs(os.path.join(branch, "outputs"), exist_ok=True)
        shutil.copy(os.path.join(BASE, "outputs", "analysis_dataset.csv"), os.path.join(branch, "outputs"))
        for s in ("02_features_and_tendency", "03_clustering", "08_replication_150h"):
            launch(s, env={"CLUSTERING_BASE": branch, "DAT_SET": measure}, label=f'variants_dat_{measure}_{s}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fr', type=int, default=0); ap.add_argument("--a", type=int, default=10 ** 6)
    ap.add_argument('--listing', action="store_true")
    a = ap.parse_args()
    steps = [(0, "00_data_cache")] + [(int(r["step"]), r["script"]) for r in csv.DictReader(open(os.path.join(HERE, "ORDER.csv"), encoding="utf-8"))]
    for n, name in steps:
        if not a.fr <= n <= a.a:
            continue
        if a.listing:
            print(n, name); continue
        if name == "104_dat_variants_comparison":
            dat_variants()
        launch(name)
    print('chain finished')


if __name__ == "__main__":
    main()
