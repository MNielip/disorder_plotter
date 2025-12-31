import os
import sys
import webbrowser
import threading
import multiprocessing 
from uvicorn import run
import gui_app 

# --- CRITICAL FIX FOR WINDOWED MODE ---
# PyInstaller --windowed sets sys.stdout/stderr to None. 
# Uvicorn needs .isatty() methods, so we redirect them to devnull.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

def start_browser():
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # We delay the browser slightly to ensure the server is ready
    threading.Timer(1.5, start_browser).start()
    
    # log_config=None prevents Uvicorn from trying to configure the 
    # default fancy console logger that crashes in windowed mode.
    run(gui_app.app, host="127.0.0.1", port=8000, workers=1, reload=False, log_config=None)
