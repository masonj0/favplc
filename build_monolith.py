import PyInstaller.__main__
import os
import sys

def create_version_info():
    """Creates a basic version_info.txt for the Windows EXE."""
    version_content = """VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
    ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'Fortuna Intelligence'),
        StringStruct(u'FileDescription', u'Fortuna All-In-One Intelligence Monolith'),
        StringStruct(u'FileVersion', u'1.0.0'),
        StringStruct(u'InternalName', u'FortunaMonolith'),
        StringStruct(u'LegalCopyright', u'Copyright (c) 2026'),
        StringStruct(u'OriginalFilename', u'FortunaMonolith.exe'),
        StringStruct(u'ProductName', u'Fortuna Intelligence'),
        StringStruct(u'ProductVersion', u'1.0.0')])
      ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
    with open("version_info.txt", "w") as f:
        f.write(version_content)
    print("Created version_info.txt")

def build_exe():
    print("Preparing to build Fortuna Intelligence Monolith EXE...")

    # Path to the main script
    script_path = "fortuna.py"

    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found.")
        return

    if not os.path.exists("version_info.txt"):
        create_version_info()

    # PyInstaller arguments
    # We use --onefile for a single executable monolith
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
        "--collect-all=camoufox",
        "--collect-all=msgspec",
        # Ensure async sqlite and pydantic are bundled
        "--hidden-import=aiosqlite",
        "--hidden-import=pydantic",
        "--hidden-import=pydantic_core",
        "--hidden-import=pydantic_settings",
        "--hidden-import=pandas",
        "--hidden-import=structlog",
        "--hidden-import=tenacity",
        "--hidden-import=uvicorn.logging",
        "--hidden-import=uvicorn.protocols",
        "--hidden-import=uvicorn.protocols.http",
        "--hidden-import=uvicorn.protocols.http.auto",
        "--hidden-import=uvicorn.protocols.websockets",
        "--hidden-import=uvicorn.protocols.websockets.auto",
        "--hidden-import=uvicorn.lifespan",
        "--hidden-import=uvicorn.lifespan.on",
        # Metadata
        "--version-file=version_info.txt",
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
