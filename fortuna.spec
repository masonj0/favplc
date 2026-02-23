# -*- mode: python ; coding: utf-8 -*-
#
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  NOTE: The canonical build method is `python build_monolith.py`.   ║
# ║  This spec file is provided for local/manual builds only.          ║
# ║  Keep it in sync with build_monolith.py or regenerate as needed.   ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = [
    # ── Fortuna modules ────────────────────────────────────────────────
    'fortuna_analytics',

    # ── Async & DB ─────────────────────────────────────────────────────
    'aiosqlite', 'sqlite3', 'asyncio', 'aiofiles',

    # ── Timezone (CRITICAL on Windows — no built-in IANA DB) ───────────
    'zoneinfo', 'tzdata',

    # ── Pydantic ───────────────────────────────────────────────────────
    'pydantic', 'pydantic_core', 'tomli',

    # ── Data processing ────────────────────────────────────────────────
    'tenacity', 'orjson', 'msgspec',

    # ── HTTP client stack ──────────────────────────────────────────────
    'httpx', 'httpx._transports', 'httpx._transports.default',
    'httpcore', 'httpcore._async', 'httpcore._sync', 'h11',

    # ── anyio / sniffio ────────────────────────────────────────────────
    'anyio', 'anyio._backends', 'anyio._backends._asyncio',
    'anyio._backends._trio', 'sniffio',

    # ── Encodings ──────────────────────────────────────────────────────
    'encodings', 'encodings.utf_8', 'encodings.ascii',
    'encodings.latin_1', 'encodings.idna',

    # ── Stdlib ─────────────────────────────────────────────────────────
    'multiprocessing', 'concurrent.futures', 'json',
    'setuptools', 'pkg_resources',
    'importlib.metadata', 'importlib.resources',
]

# ── collect_all: packages whose data files are required at runtime ─────
_collect_all_packages = [
    'scrapling', 'browserforge', 'curl_cffi', 'camoufox',
    'selectolax', 'playwright', 'rich', 'webview',
    'pydantic', 'pydantic_core', 'structlog', 'tomli',
]
for _pkg in _collect_all_packages:
    try:
        _ret = collect_all(_pkg)
        datas += _ret[0]; binaries += _ret[1]; hiddenimports += _ret[2]
    except Exception:
        pass  # Package not installed; skip gracefully

# ── collect_submodules equivalent (add submodule hiddenimports) ────────
_collect_sub_packages = ['uvicorn', 'fastapi', 'starlette']
for _pkg in _collect_sub_packages:
    try:
        _ret = collect_all(_pkg)
        datas += _ret[0]; binaries += _ret[1]; hiddenimports += _ret[2]
    except Exception:
        pass

a = Analysis(
    ['fortuna.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'PIL', 'scipy', 'cv2', 'opencv',
        'torch', 'tensorflow', 'PIL.ImageQt',
        'tkinter', 'PyQt5',
        'pytest', 'hypothesis', 'IPython', 'jupyter', 'notebook',
        'sphinx', 'docutils', 'jedi', 'parso',
        'wheel', 'pip', 'setuptools._distutils',
        'pandas.tests', 'numpy.tests', 'tornado',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FortunaFaucetPortableApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                # UPX triggers AV false positives — match --noupx
    upx_exclude=[],
    runtime_tmpdir=None,      # None = system %TEMP%; NEVER use '.' (breaks on
                              # root-level dirs and read-only locations)
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='version_info.txt',
)
