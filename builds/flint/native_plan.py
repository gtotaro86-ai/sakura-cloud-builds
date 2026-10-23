"""Run the unchanged native full-model metadata planner, without reading payloads."""
import ctypes, sys
from pathlib import Path
lib=ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
lib.sage_dryrun.argtypes=[ctypes.c_char_p,ctypes.c_int]
raise SystemExit(lib.sage_dryrun(str(Path(sys.argv[2]).resolve()).encode(),int(sys.argv[3])))
