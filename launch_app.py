import subprocess
import sys
import os
import threading
import webview
import time
import requests

def resource_path(relative_path):
    try:
        base_path =sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
        
    return os.path.join(base_path,relative_path)
    



def start_streamlit():
    npath = os.path.join(os.path.dirname(__file__),"Model_builder.py")
    patha = resource_path("pages\01_Model_Builder.py")
    subprocess.Popen(["streamlit", "run",npath ])


def wait_for_streamlit():
    retries = 15
    for _ in range(retries):
        try:
            response = requests.get("http://localhost:8501")
            if response.status_code == 200:
                return True
        except:
            time.sleep(1)
    return False


if __name__ == "__main__":
    threading.Thread(target=start_streamlit).start()
    
    if wait_for_streamlit():
        webview.create_window("TrainFlow", "http://localhost:8501", width=1400, height=900)
        webview.start()
    else:
        print("Failed to connect to Streamlit server. Check if it's running.")
