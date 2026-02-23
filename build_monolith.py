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
    # PyInstaller uses ';' on Windows, ':' on POSIX
    sep = ';' if platform.system() == 'Windows' else ':'

    # Directories
    for dirname in ["static", "templates", "config"]:
        if os.path.exists(dirname):
            data_files.extend(["--add-data", f"{dirname}{sep}{dirname}"])

    # Root files
    for filename in ["VERSION", "config.toml"]:
        if os.path.exists(filename):
            data_files.extend(["--add-data", f"{filename}{sep}."])

    # Optional reports bundled at build time
    report_files = [
        "summary_grid.txt",
        "goldmine_report.txt",
        "analytics_report.txt",
        "fortuna_report.html",
        "race_data.json",
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

    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        metadata["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:
        metadata["git_commit"] = "unknown"
        metadata["git_branch"] = "unknown"

    try:
        with open("build_info.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print("Created build_info.json")
        sep = ';' if platform.system() == 'Windows' else ':'
        return ["--add-data", f"build_info.json{sep}."]
    except Exception as e:
        print(f"Warning: Could not create build_info.json: {e}")
        return []


def verify_exe(exe_path):
    """Verify the EXE can launch without errors."""
    print("\nVerifying EXE...")

    # Remove Mark-of-the-Web so Windows doesn't block the freshly-built artifact
    if platform.system() == "Windows":
        try:
            subprocess.run(
                ["powershell.exe", "-Command", f"Unblock-File -Path '{exe_path}'"],
                check=False,
                capture_output=True
            )
            print("   Unblocked EXE (PowerShell)")
        except Exception:
            pass

    try:
        result = subprocess.run(
            [exe_path, "--help"],
            capture_output=True,
            text=True,
            # Onefile EXEs must fully self-extract before Python starts.
            # For large bundles (scrapling, playwright, curl_cffi data) this
            # can take well over 60s on first run.  120s gives safe headroom.
            timeout=120,
        )
        if result.returncode == 0:
            print("[PASS] EXE verification passed")
            return True
        else:
            print(f"[FAIL] EXE returned error code: {result.returncode}")
            print(f"stderr: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("[FAIL] EXE timed out during self-extraction or startup")
        return False
    except Exception as e:
        print(f"[FAIL] Could not verify EXE: {e}")
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

    # ── Base arguments ─────────────────────────────────────────────────────────
    args = [
        str(script_path),
        "--onefile",
        "--name=FortunaFaucetPortableApp",
        "--clean",
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │  DO NOT add --runtime-tmpdir=.                                      │
        # │                                                                     │
        # │  Omitting --runtime-tmpdir lets PyInstaller use the system %TEMP%   │
        # │  directory (C:\Users\<user>\AppData\Local\Temp on Windows), which   │
        # │  is ALWAYS user-writable and expected by Windows Defender.           │
        # │                                                                     │
        # │  Setting it to '.' causes "Cannot create temporary directory!"      │
        # │  whenever CWD requires elevated permissions (e.g. root-level dirs   │
        # │  like C:\Temp, network shares, or the Downloads folder on locked-   │
        # │  down enterprise machines).                                         │
        # └─────────────────────────────────────────────────────────────────────┘
        #
        # UPX compression is a major trigger for SmartScreen "Unknown Publisher"
        # flags and antivirus false positives on unsigned EXEs.
        "--noupx",
    ]

    if os.path.exists("version_info.txt"):
        args.append("--version-file=version_info.txt")

    if not console_mode:
        args.append("--noconsole")

    if debug:
        args.extend(["--debug=all", "--log-level=DEBUG"])

    if os.path.exists("assets/icon.ico"):
        args.append("--icon=assets/icon.ico")

    # ── --collect-all: packages whose data files are required at runtime ───────
    collect_all_packages = [
        "browserforge",
        "scrapling",
        "curl_cffi",
        "camoufox",
        "selectolax",
        "playwright",
        "rich",
        "webview",
        "pydantic",
        "pydantic_core",
        "structlog",
        "tomli",
    ]
    for pkg in collect_all_packages:
        args.append(f"--collect-all={pkg}")

    # ── --collect-submodules: packages that register plugins/backends lazily ───
    collect_submodules = [
        "uvicorn",
        "fastapi",
        "starlette",
    ]
    for pkg in collect_submodules:
        args.append(f"--collect-submodules={pkg}")

    # ── Hidden imports ─────────────────────────────────────────────────────────
    hidden_imports = [
        # ── Fortuna modules (imported dynamically) ─────────────────────────────
        "fortuna_analytics",

        # ── Async & DB ─────────────────────────────────────────────────────────
        "aiosqlite",
        "sqlite3",
        "asyncio",
        "aiofiles",

        # ── Timezone (CRITICAL on Windows) ─────────────────────────────────────
        "zoneinfo",
        "tzdata",

        # ── Data processing ────────────────────────────────────────────────────
        "pandas",
        "numpy",
        "tenacity",
        "orjson",
        "msgspec",

        # ── Notifications (Windows-only) ───────────────────────────────────────
        "winotify",
        "win10toast_py3",

        # ── pkg_resources / importlib.metadata ─────────────────────────────────
        "setuptools",
        "pkg_resources",
        "importlib.metadata",
        "importlib.resources",

        # ── HTTP client stack ──────────────────────────────────────────────────
        "httpx",
        "httpx._transports",
        "httpx._transports.default",
        "httpcore",
        "httpcore._async",
        "httpcore._sync",
        "h11",

        # ── anyio / sniffio (async backends) ──────────────────────────────────
        "anyio",
        "anyio._backends",
        "anyio._backends._asyncio",
        "anyio._backends._trio",
        "sniffio",

        # ── Encodings ─────────────────────────────────────────────────────────
        "encodings",
        "encodings.utf_8",
        "encodings.ascii",
        "encodings.latin_1",
        "encodings.idna",

        # ── Stdlib ─────────────────────────────────────────────────────────────
        "multiprocessing",
        "concurrent.futures",
        "json",
    ]
    for imp in hidden_imports:
        args.append(f"--hidden-import={imp}")

    # ── Data files ─────────────────────────────────────────────────────────────
    args.extend(get_data_files())
    args.extend(create_build_metadata())

    # ── Exclusions (reduce EXE bloat) ──────────────────────────────────────────
    excludes = [
        "matplotlib", "PIL", "scipy",
        "cv2", "opencv", "torch", "tensorflow",
        "PIL.ImageQt",
        "tkinter", "PyQt5",
        "pytest", "hypothesis",
        "IPython", "jupyter", "notebook",
        "sphinx", "docutils", "jedi", "parso",
        "wheel", "pip",
        "setuptools._distutils",
        "pandas.tests", "numpy.tests",
        "tornado",
    ]
    for exc in excludes:
        args.append(f"--exclude-module={exc}")

    # ── Argument sanity check ──────────────────────────────────────────────────
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

    PyInstaller.__main__.run(final_args)

    # ── Post-build verification ────────────────────────────────────────────────
    exe_name = (
        "FortunaFaucetPortableApp.exe"
        if platform.system() == "Windows"
        else "FortunaFaucetPortableApp"
    )
    exe_path = os.path.join("dist", exe_name)

    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print("\n" + "=" * 60)
        print("[SUCCESS] Build complete!")
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
    parser.add_argument("--gui", action="store_true", help="Hide console window (--noconsole)")
    parser.add_argument("--debug", action="store_true", help="Enable PyInstaller debug mode")

    parsed = parser.parse_args()
    build_exe(console_mode=not parsed.gui, debug=parsed.debug)
