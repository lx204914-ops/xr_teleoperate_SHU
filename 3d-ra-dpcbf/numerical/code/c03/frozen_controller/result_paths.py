from pathlib import Path
ROOT=Path(__file__).resolve().parent

def result_root():
    # Never silently load historical experiments from another method/version.
    return ROOT.parent/'results/current_method/final'
