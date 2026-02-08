import PyInstaller.__main__
import os
import sys
from datetime import datetime

def create_version_info():
    """Creates a basic version_info.txt for the Windows EXE."""
    year = datetime.now().year  # Dynamically use current year (2026)
    version_content = f"""VSVersionInfo(
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
        StringStruct(u'LegalCopyright', u'Copyright (c) {year}'),
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


def get_data_files():
    """Collect data files that need to be bundled."""
    data_files = []

    # Add static assets if they exist
    if os.path.exists("static"):
        data_files.append(("--add-data", "static;static"))
    if os.path.exists("templates"):
        data_files.append(("--add-data", "templates;templates"))
    if os.path.exists("config"):
        data_files.append(("--add-data", "config;config"))

    return data_files


def build_exe(console_mode: bool = True, debug: bool = False):
    """
    Build the Fortuna Monolith executable.

    Args:
        console_mode: If True, show console window. False for GUI-only.
        debug: If True, add debug flags for troubleshooting.
    """
    print("Preparing to build Fortuna Intelligence Monolith EXE...")

    script_path = "fortuna.py"

    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found.")
        sys.exit(1)

    if not os.path.exists("version_info.txt"):
        create_version_info()

    # Base arguments
    args = [
        script_path,
        "--onefile",
        "--name=FortunaMonolith",
        "--clean",
        "--version-file=version_info.txt",
    ]

    # Console or windowed mode
    if not console_mode:
        args.append("--noconsole")

    # Debug mode
    if debug:
        args.extend([
            "--debug=all",
            "--log-level=DEBUG",
        ])

    # Icon if available
    if os.path.exists("assets/icon.ico"):
        args.append("--icon=assets/icon.ico")

    # Collect-all for complex packages
    collect_all_packages = [
        "scrapling",
        "browserforge",
        "curl_cffi",
        "fastapi",
        "uvicorn",
        "webview",
        "selectolax",
        "camoufox",
        "msgspec",
        "pydantic",
        "starlette",
        "rich",
        "win10toast_py3",
    ]

    for pkg in collect_all_packages:
        args.append(f"--collect-all={pkg}")

    # Hidden imports for packages that need explicit inclusion
    hidden_imports = [
        # Async & DB
        "aiosqlite",
        "sqlite3",
        "asyncio",

        # Pydantic ecosystem
        "pydantic",
        "pydantic_core",
        "pydantic_settings",
        "pydantic.deprecated.decorator",

        # Data processing
        "pandas",
        "numpy",

        # Logging & resilience
        "structlog",
        "tenacity",

        # Uvicorn internals
        "uvicorn.logging",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.http.httptools_impl",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.protocols.websockets.wsproto_impl",
        "uvicorn.protocols.websockets.websockets_impl",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "uvicorn.lifespan.off",

        # HTTP clients
        "httpx",
        "httpx._transports",
        "httpx._transports.default",
        "h11",
        "anyio",
        "anyio._backends",
        "anyio._backends._asyncio",
        "sniffio",

        # Encodings (critical for --onefile)
        "encodings",
        "encodings.utf_8",
        "encodings.ascii",
        "encodings.latin_1",
        "encodings.idna",

        # Multiprocessing
        "multiprocessing",
        "concurrent.futures",

        # JSON
        "json",
        "orjson",
    ]

    for imp in hidden_imports:
        args.append(f"--hidden-import={imp}")

    # Add data files
    for flag, value in get_data_files():
        args.append(f"{flag}={value}")

    # Exclude unnecessary packages to reduce size
    excludes = [
        "matplotlib",
        "PIL",
        "tkinter",
        "scipy",
        "pytest",
        "hypothesis",
        "setuptools",
        "wheel",
        "pip",
    ]

    for exc in excludes:
        args.append(f"--exclude-module={exc}")

    print(f"\nRunning PyInstaller with {len(args)} arguments...")
    print("=" * 60)

    try:
        PyInstaller.__main__.run(args)

        exe_path = os.path.join("dist", "FortunaMonolith.exe")
        if os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            print("\n" + "=" * 60)
            print(f"[SUCCESS] Build complete!")
            print(f"   Output: {exe_path}")
            print(f"   Size: {size_mb:.1f} MB")
            print("=" * 60)
        else:
            print("\n[WARNING] Build completed but EXE not found at expected path")

    except Exception as e:
        print(f"\n[ERROR] Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Fortuna Monolith EXE")
    parser.add_argument("--gui", action="store_true", help="Hide console window")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    build_exe(console_mode=not args.gui, debug=args.debug)
