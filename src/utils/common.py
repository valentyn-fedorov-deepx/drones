import os, sys

# Wrapper to unify resource paths for built standalone app
def resource_path(path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, path)
    return path