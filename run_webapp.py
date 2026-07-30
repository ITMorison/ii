import subprocess
import sys
import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

REQUIRED_PACKAGES = [
    "numpy",
    "opencv-python",
    "torch",
    "torchvision", 
    "streamlit",
    "pyyaml",
    "requests",
    "streamlit-autorefresh",
    "protobuf<4.0.0"
]

def check_and_install_packages():
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            if package == "opencv-python":
                import cv2
            else:
                __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n[+] Installing missing packages: {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing)
        return True
    return False

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 50)
    print("[*] Security Monitoring System - Starting")
    print("=" * 50)
    
    print("\n[*] Checking dependencies...")
    if check_and_install_packages():
        print("  [+] All dependencies installed")
    
    print("\n[*] Starting Streamlit application...")
    print("    Open in browser: http://localhost:8501")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except Exception as e:
        print(f"\n[-] Error: {e}")
    finally:
        print("\n[*] Press Enter to exit...")
        input()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[-] Critical error: {e}")
        print("\n[*] Press Enter to exit...")
        input()