# run.py
import sys
import threading
import webbrowser
import multiprocessing # <--- ADD THIS
from uvicorn import run
import gui_app 

def start_browser():
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    # Necessary for PyInstaller on Windows to handle process forking
    multiprocessing.freeze_support() # <--- ADD THIS
    
    threading.Timer(1.5, start_browser).start()
    
    # Pass the app OBJECT, not the string, to avoid import errors in frozen state
    # Also, force workers=1 and reload=False
    run(gui_app.app, host="127.0.0.1", port=8000, workers=1, reload=False)
