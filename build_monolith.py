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


def smartscreen_autopilot():
    """
    Experimental: Uses PowerShell to automate clicking 'Run Anyway' if
    the SmartScreen dialog appears during verification.
    Not recommended for production.
    """
    ps_script = """
    $shell = New-Object -ComObject WScript.Shell
    for($i=0; $i -lt 10; $i++) {
        if($shell.AppActivate("Windows protected your PC")) {
            Sleep 1
            $shell.SendKeys("{TAB}") # More info
            $shell.SendKeys(" ")
            Sleep 1
            $shell.SendKeys("{TAB}") # Run anyway
            $shell.SendKeys(" ")
            break
        }
        Sleep 2
    }
    """
    if platform.system() == "Windows":
        subprocess.Popen(["powershell.exe", "-Command", ps_script])


def verify_exe(exe_path):
    """Verify the EXE can launch without errors."""
    print("\nVerifying EXE...")

    # Attempt to remove "Mark of the Web" on Windows to bypass some SmartScreen checks
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
        # Start the autopilot in the background if on Windows
        if platform.system() == "Windows":
            smartscreen_autopilot()

        result = subprocess.run(
            [exe_path, "--help"],
            capture_output=True,
            text=True,
            timeout=60,  # onefile EXEs need time to self-extract before running
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
        # Extract to a predictable temp location; avoids permission issues when
        # the EXE is run from a read-only directory (e.g. Downloads on some
        # Windows configs).
        "--runtime-tmpdir=.",
        # UPX compression is a major trigger for SmartScreen "Unknown Publisher"
        # flags and antivirus false positives. (Council/Jules Fix)
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
    # Rule of thumb: if the package loads JSON/YAML/browser assets from its own
    # directory (rather than the OS or a URL), it needs --collect-all.
    collect_all_packages = [
        # Scraping / browser-impersonation stack
        "browserforge",     # fingerprint headers/data JSON files
        "scrapling",        # internal JS assets and adapters
        "curl_cffi",        # pre-compiled .dll/.so curl binaries
        "camoufox",         # browser profile data and configs
        "selectolax",       # compiled Modest/Lexbor parser
        # Playwright: the _impl/_json/ protocol specs are loaded at import time;
        # hidden imports alone are NOT enough to make it work frozen.
        "playwright",
        # UI / display
        "rich",             # color themes and markup definitions
        # pywebview ships JS bridge assets + platform-specific DLLs
        "webview",
        # Pydantic v2 uses pydantic_core (compiled) + JSON schema data files;
        # --collect-submodules misses the data, --collect-all covers everything.
        "pydantic",
        "pydantic_core",
        # structlog has lazy-loaded processors discovered via inspect
        "structlog",
        # tomli / tomllib: pure-Python but some builds ship a compiled C extension
        "tomli",
    ]
    for pkg in collect_all_packages:
        args.append(f"--collect-all={pkg}")

    # ── --collect-submodules: packages that register plugins/backends lazily ───
    collect_submodules = [
        "uvicorn",      # server implementations, protocols, loops
        "fastapi",      # routing, dependency injection internals
        "starlette",    # middleware and exception handler registrations
    ]
    for pkg in collect_submodules:
        args.append(f"--collect-submodules={pkg}")

    # ── Hidden imports ─────────────────────────────────────────────────────────
    hidden_imports = [
        # ── Fortuna modules (imported dynamically via sys.path tricks) ─────────
        "fortuna_analytics",    # imported at runtime in diagnostic/auditor paths

        # ── Async & DB ─────────────────────────────────────────────────────────
        "aiosqlite",
        "sqlite3",
        "asyncio",
        "aiofiles",

        # ── Timezone (CRITICAL on Windows) ────────────────────────────────────
        # Windows has no built-in IANA timezone database, so zoneinfo falls back
        # to the `tzdata` package. Without this the EXE crashes with
        # ZoneInfoNotFoundError("America/New_York") on any Windows machine that
        # hasn't manually installed tzdata — i.e. all of them.
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

        # ── pkg_resources / importlib.metadata ────────────────────────────────
        "setuptools",
        "pkg_resources",
        "importlib.metadata",
        "importlib.resources",

        # ── HTTP client stack ──────────────────────────────────────────────────
        # httpcore is the actual transport backend for httpx; omitting it causes
        # "No module named httpcore" at runtime even though httpx is collected.
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
        # The trio backend import is attempted and suppressed at anyio startup;
        # if it's missing entirely from the frozen archive PyInstaller can emit
        # noisy tracebacks at startup.
        "anyio._backends._trio",
        "sniffio",

        # ── Encodings (belt-and-suspenders for frozen builds) ─────────────────
        "encodings",
        "encodings.utf_8",
        "encodings.ascii",
        "encodings.latin_1",
        "encodings.idna",

        # ── Stdlib / multiprocessing ───────────────────────────────────────────
        "multiprocessing",
        "concurrent.futures",
        "json",
    ]
    for imp in hidden_imports:
        args.append(f"--hidden-import={imp}")

    # ── Data files ─────────────────────────────────────────────────────────────
    args.extend(get_data_files())
    args.extend(create_build_metadata())

    # ── Exclusions (reduce EXE bloat) ─────────────────────────────────────────
    excludes = [
        # Visualisation / ML (never used at runtime)
        "matplotlib", "PIL", "scipy",
        "cv2", "opencv", "torch", "tensorflow",
        "PIL.ImageQt",
        # GUI toolkits (webview uses its own; tkinter would pull in Tcl/Tk DLLs)
        "tkinter", "PyQt5",
        # Dev/test tooling
        "pytest", "hypothesis",
        "IPython", "jupyter", "notebook",
        "sphinx", "docutils", "jedi", "parso",
        # Package management (no reason to ship pip inside the EXE)
        "wheel", "pip",
        "setuptools._distutils",
        # Test sub-packages of bundled libraries
        "pandas.tests", "numpy.tests",
        # tornado conflicts with uvicorn's event loop management when frozen
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
