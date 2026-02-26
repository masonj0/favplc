# 🔐 Signing the Fortuna Monolith EXE

To prevent Windows Defender SmartScreen from blocking the portable app and forcing users to click "Run Anyway", the executable must be signed with a valid Code Signing Certificate.

## 📋 Prerequisites

1.  **Code Signing Certificate**: You must obtain a certificate from a trusted Certificate Authority (CA) such as DigiCert, Sectigo, or GlobalSign.
    -   **Standard Code Signing**: Bypasses SmartScreen after gaining "reputation".
    -   **EV (Extended Validation) Code Signing**: Provides immediate reputation and bypasses SmartScreen instantly.
2.  **Windows SDK**: Requires `signtool.exe`, which is included with the Windows SDK.

## 🛠️ Signing Process (Manual)

If you have a `.pfx` certificate file, run the following command in a Windows Terminal (as Administrator):

```powershell
signtool sign /f "C:\path\to\your\certificate.pfx" /p YourPassword /tr http://timestamp.digicert.com /td sha256 /fd sha256 "dist\FortunaFaucetPortableApp.exe"
```

## 🤖 Automated Signing (Recommended)

You can update `build_monolith.py` to automatically sign the EXE if the certificate is available.

Add the following function to `build_monolith.py` and call it after the build process:

```python
def sign_exe(exe_path):
    """Sign the EXE using signtool.exe if a certificate is configured."""
    cert_path = os.environ.get("CODE_SIGN_CERT_PATH")
    cert_pass = os.environ.get("CODE_SIGN_CERT_PASS")

    if not cert_path or not cert_pass:
        print("\n[SKIP] Code signing skipped (ENV vars not set)")
        return

    print("\nSigning EXE...")
    try:
        subprocess.run([
            "signtool", "sign",
            "/f", cert_path,
            "/p", cert_pass,
            "/tr", "http://timestamp.digicert.com",
            "/td", "sha256",
            "/fd", "sha256",
            exe_path
        ], check=True)
        print("[PASS] EXE signed successfully")
    except Exception as e:
        print(f"[FAIL] Signing failed: {e}")
```

## ⚡ Alternative: Self-Signing (Not Recommended for Public Distribution)

If you just want to sign it for your own machines, you can create a self-signed certificate, but you must manually install that certificate into the "Trusted Root Certification Authorities" store on every machine that runs the EXE.
