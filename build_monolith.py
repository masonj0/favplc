import PyInstaller.__main__
import os
import sys
import platform
import json
import subprocess
from datetime import datetime

def create_version_info():
    """Creates version_info.txt for the Windows EXE metadata."""
    year = datetime.now().year
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
        StringStruct(u'FileDescription', u'Fortuna Faucet Portable App'),
        StringStruct(u'FileVersion', u'1.0.0'),
        StringStruct(u'InternalName', u'FortunaFaucetPortableApp'),
        StringStruct(u'LegalCopyright', u'Copyright (c) {year}'),
        StringStruct(u'OriginalFilename', u'FortunaFaucetPortableApp.exe'),
        StringStruct(u'ProductName', u'Fortuna Intelligence'),
        StringStruct(u'ProductVersion', u'1.0.0')])
      ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
    try:
        with open("version_info.txt", "w") as f:
            f.write(version_content)
        print("Created version_info.txt")
    except IOError as e:
        print(f"Warning: Could not create version_info.txt: {e}")
        print("Continuing without version metadata...")


def get_data_files():
    """Collect data files to bundle. Returns flat list of strings for args.extend()."""
    data_files = []
    # Platform-specific separator
    sep = ';' if platform.system() == 'Windows' else ':'

    # Directories
    for dirname in ["static", "templates", "config"]:
        if os.path.exists(dirname):
            data_files.extend(["--add-data", f"{dirname}{sep}{dirname}"])

    # Root files
    for filename in ["VERSION", "config.toml"]:
        if os.path.exists(filename):
            data_files.extend(["--add-data", f"{filename}{sep}."])

    # Optional reports
    report_files = [
        "summary_grid.txt",
        "goldmine_report.txt",
        "analytics_report.txt",
        "fortuna_report.html",
        "race_data.json"
    ]
    for f in report_files:
        if os.path.exists(f):
            data_files.extend(["--add-data", f"{f}{sep}."])

    return data_files


def create_build_metadata():
    """Creates build_info.json with build metadata."""
    metadata = {
        "build_date": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    # Try to get git info
    try:
        metadata["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        metadata["git_branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        metadata["git_commit"] = "unknown"
        metadata["git_branch"] = "unknown"

    try:
        with open("build_info.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print("Created build_info.json")
        return ["--add-data", f"build_info.json{';' if platform.system() == 'Windows' else ':'}."]
    except Exception as e:
        print(f"Warning: Could not create build_info.json: {e}")
        return []


def verify_exe(exe_path):
    """Verify the EXE can launch without errors."""
    print("\nVerifying EXE...")
    if platform.system() != 'Windows' and not exe_path.endswith('.exe'):
        # On non-windows, the output might not have .exe extension
        # but for this repo we are focused on Windows EXE.
        pass

    try:
        # Try to run with --help (should be fast)
        # Note: This might not work on the build machine if it's Linux building for Windows
        # but in GHA we are on Windows.
        result = subprocess.run(
            [exe_path, "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print("[PASS] EXE verification passed")
            return True
        else:
            print(f"[FAIL] EXE returned error code: {result.returncode}")
            print(f"stderr: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("[FAIL] EXE timed out (might be stuck)")
        return False
    except Exception as e:
        print(f"[FAIL] Could not verify EXE: {e}")
        # If we are on Linux but built a Windows EXE, verification will fail here.
        if platform.system() != 'Windows':
            print("  (This is expected when cross-compiling or building on Linux)")
            return True
        return False


def build_exe(console_mode: bool = True, debug: bool = False):
    """Build the Fortuna Monolith executable."""
    print("Preparing to build Fortuna Intelligence Monolith EXE...")
    print("=" * 60)

    script_path = "fortuna.py"
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found.")
        sys.exit(1)

    if not os.path.exists("version_info.txt"):
        create_version_info()

    # Base arguments
    args = [
        str(script_path),
        "--onefile",
        "--name=FortunaFaucetPortableApp",
        "--clean",
    ]

    if os.path.exists("version_info.txt"):
        args.append("--version-file=version_info.txt")

    if not console_mode:
        args.append("--noconsole")

    if debug:
        args.extend(["--debug=all", "--log-level=DEBUG"])

    if os.path.exists("assets/icon.ico"):
        args.append("--icon=assets/icon.ico")

    # Packages that NEED their data files (use --collect-all)
    collect_all_packages = [
        "browserforge",
        "scrapling",
        "curl_cffi",
        "camoufox",
        "selectolax",
        "rich",
        "tomli",
    ]
    for pkg in collect_all_packages:
        args.append(f"--collect-all={pkg}")

    # Packages that need submodules but not data (use --collect-submodules)
    collect_submodules = [
        "uvicorn",
        "fastapi",
        "starlette",
        "pydantic",
    ]
    for pkg in collect_submodules:
        args.append(f"--collect-submodules={pkg}")

    # Individual hidden imports
    hidden_imports = [
        # Async & DB
        "aiosqlite", "sqlite3", "asyncio",
        # Data processing
        "pandas", "numpy", "structlog", "tenacity",
        # Notifications
        "winotify", "win10toast_py3",
        # Build tools (needed by pkg_resources)
        "setuptools", "pkg_resources",
        # HTTP clients
        "httpx", "httpx._transports", "httpx._transports.default",
        "h11", "anyio", "anyio._backends", "anyio._backends._asyncio", "sniffio",
        # Encodings
        "encodings", "encodings.utf_8", "encodings.ascii",
        "encodings.latin_1", "encodings.idna",
        # Misc
        "multiprocessing", "concurrent.futures", "json", "orjson", "msgspec",
        # Webview
        "webview",
    ]
    for imp in hidden_imports:
        args.append(f"--hidden-import={imp}")

    # Add data files
    args.extend(get_data_files())

    # Add build metadata
    args.extend(create_build_metadata())

    # Exclude bloat
    excludes = [
        "matplotlib", "PIL", "tkinter", "scipy",
        "pytest", "hypothesis",
        "wheel", "pip",
        "IPython", "jupyter", "notebook",
        "pandas.tests", "numpy.tests",
        "tornado", "sphinx", "docutils", "jedi", "parso",
        # Additional bloat
        "cv2", "opencv", "torch", "tensorflow",
        "PIL.ImageQt", "PyQt5", "setuptools._distutils",
    ]
    for exc in excludes:
        args.append(f"--exclude-module={exc}")

    # FINAL HARNESS: Ensure all args are strings and fail loudly if not
    errors = []
    final_args = []
    for i, arg in enumerate(args):
        if not isinstance(arg, str):
            errors.append(f"  [{i}] {arg!r} (type: {type(arg).__name__})")
        else:
            final_args.append(arg)

    if errors:
        print("CRITICAL ERROR: Non-string arguments detected:")
        print("\n".join(errors))
        print("\nThis indicates a bug in build_monolith.py")
        sys.exit(1)

    print(f"\nRunning PyInstaller with {len(final_args)} arguments...")
    print("=" * 60)

    # Run PyInstaller
    PyInstaller.__main__.run(final_args)

    # Verify output
    exe_name = "FortunaFaucetPortableApp.exe" if platform.system() == 'Windows' else "FortunaFaucetPortableApp"
    exe_path = os.path.join("dist", exe_name)

    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print("\n" + "=" * 60)
        print(f"[SUCCESS] Build complete!")
        print(f"   Output: {exe_path}")
        print(f"   Size:   {size_mb:.1f} MB")

        if verify_exe(exe_path):
             print("=" * 60)
        else:
             print("[WARN] EXE built but verification failed")
             sys.exit(1)
    else:
        print("\n[ERROR] Build finished but EXE not found")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Fortuna Monolith EXE")
    parser.add_argument("--gui", action="store_true", help="Hide console window")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    build_exe(console_mode=not args.gui, debug=args.debug)
