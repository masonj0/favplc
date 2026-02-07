import PyInstaller.__main__
import os
import sys

def build_exe():
    print("Preparing to build Fortuna Intelligence Monolith EXE...")

    # Path to the main script
    script_path = "fortuna.py"

    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found.")
        return

    # PyInstaller arguments
    # We use --onefile for a single executable monolith
    # We collect all sub-packages for complex dependencies
    args = [
        script_path,
        "--onefile",
        "--name=FortunaMonolith",
        "--clean",
        # Web and Scraper dependencies
        "--collect-all=scrapling",
        "--collect-all=browserforge",
        "--collect-all=curl_cffi",
        "--collect-all=fastapi",
        "--collect-all=uvicorn",
        "--collect-all=webview",
        "--collect-all=selectolax",
        # Ensure async sqlite and pydantic are bundled
        "--hidden-import=aiosqlite",
        "--hidden-import=pydantic",
        "--hidden-import=pydantic_core",
        # Metadata
        "--version-file=version_info.txt" if os.path.exists("version_info.txt") else None,
    ]

    # Filter out None values
    args = [arg for arg in args if arg is not None]

    print(f"Running PyInstaller with arguments: {' '.join(args)}")

    try:
        PyInstaller.__main__.run(args)
        print("\nBuild complete! Check the 'dist' folder for FortunaMonolith.")
    except Exception as e:
        print(f"Build failed: {e}")

if __name__ == "__main__":
    build_exe()
