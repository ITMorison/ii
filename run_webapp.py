import os
import subprocess
import re

# 👉 ЖЁСТКО УКАЗЫВАЕМ PYTHON 3.11
PYTHON_PATH = r"C:\Users\пк\AppData\Local\Programs\Python\Python311\python.exe"

PACKAGE_TO_IMPORT_NAME = {
    "pyyaml": "yaml",
    "protobuf": "google.protobuf",
    "streamlit-autorefresh": "streamlit_autorefresh",
    "opencv-python": "cv2",
}

def get_import_name(package_name: str) -> str:
    cleaned = re.split(r"[<>=!~]", package_name, maxsplit=1)[0].strip()
    return PACKAGE_TO_IMPORT_NAME.get(cleaned.lower(), cleaned)

CLIP_GIT_URL = "git+https://github.com/openai/CLIP.git"

def check_and_install_packages(packages):
    missing = []
    for pkg in packages:
        import_name = get_import_name(pkg)
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Отсутствуют пакеты: {missing}. Устанавливаем...")
        subprocess.check_call([PYTHON_PATH, "-m", "pip", "install"] + missing)

def check_and_install_clip():
    try:
        __import__("clip")
    except ImportError:
        print("CLIP не найден. Ставим с GitHub...")
        subprocess.check_call([PYTHON_PATH, "-m", "pip", "install", CLIP_GIT_URL])

REQUIRED_PACKAGES = [
    "streamlit",
    "opencv-python",
    "pyyaml",
    "protobuf<4.0.0",
    "streamlit-autorefresh",
    "torch",
    "pillow",
    "numpy",
    "pandas",
    "requests",
    "mediapipe",  # 👈 ДОБАВИЛИ
]

if __name__ == "__main__":
    check_and_install_packages(REQUIRED_PACKAGES)
    check_and_install_clip()

    # запуск Streamlit через Python 3.11
    script_path = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run([PYTHON_PATH, "-m", "streamlit", "run", script_path])