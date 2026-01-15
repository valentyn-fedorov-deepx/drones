import sys
import os
import shutil
from pathlib import Path

def clear_residue():
    dir_path = Path(__file__).resolve().parent
    for file in dir_path.iterdir():
        if file.is_dir() and file.name == "build":
            shutil.rmtree(file)
        elif file.is_file() and file.suffix == ".spec":
            file.unlink()
        elif file.is_file() and (not file.name.startswith("build")) and (not file.name == "clear_residue.py"):
            file.unlink()
            
        
            
if __name__ == "__main__":
    clear_residue()