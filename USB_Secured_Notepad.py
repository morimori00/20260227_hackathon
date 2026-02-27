from pathlib import Path
import datetime as dt
import hashlib
import sys
import re
import os
import getpass
import platform

# ========= CONFIG =========
USB_NAME   = "C-KILLER1"          # <-- change if you rename your USB
NOTES_FILE = "notes.txt"
HASH_FILE  = "notes.sha256"
TAMPER_LOG = "tamper.log"
LOG_ROTATE_BYTES = 100_000        # ~100 KB
# ==========================

USB_PATH   = Path("/Volumes") / USB_NAME
NOTES_PATH = USB_PATH / NOTES_FILE
HASH_PATH  = USB_PATH / HASH_FILE
LOG_PATH   = USB_PATH / TAMPER_LOG

def ensure_usb() -> bool:
    if USB_PATH.exists():
        return True
    vols = ", ".join(p.name for p in Path("/Volumes").iterdir())
    print(f"❌ USB not found at {USB_PATH}")
    print(f"   Mounted volumes: {vols}")
    return False

def ensure_files():
    if not NOTES_PATH.exists():
        NOTES_PATH.touch()
        print(f"🆕 Created {NOTES_PATH}")
    if not HASH_PATH.exists():
        write_hash()
    if not LOG_PATH.exists():
        LOG_PATH.touch()

def timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def sha256_of_path(p: Path, chunk=65536) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk_bytes in iter(lambda: f.read(chunk), b""):
            h.update(chunk_bytes)
    return h.hexdigest()

def write_hash():
    digest = sha256_of_path(NOTES_PATH)
    with HASH_PATH.open("w", encoding="utf-8") as f:
        f.write(digest + "\n")

def rotate_log_if_needed():
    try:
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > LOG_ROTATE_BYTES:
            backup = LOG_PATH.with_suffix(LOG_PATH.suffix + ".1")
            if backup.exists():
                backup.unlink(missing_ok=True)
            LOG_PATH.rename(backup)
            LOG_PATH.touch()
    except Exception:
        # If rotation fails, don't crash the app
        pass

def file_meta(p: Path) -> dict:
    try:
        st = p.stat()
        mtime = dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        return {"size": st.st_size, "mtime": mtime}
    except Exception:
        return {"size": "?", "mtime": "?"}

def log_tamper(saved_hash: str, current_hash: str, action: str):
    """
    action: 'kept_old_hash' | 'updated_hash'
    """
    rotate_log_if_needed()
    meta = file_meta(NOTES_PATH)
    user = getpass.getuser()
    host = platform.node()
    line = (
        f"{timestamp()} | user={user}@{host} | action={action} | "
        f"saved={saved_hash} | current={current_hash} | "
        f"notes_size={meta['size']} | notes_mtime={meta['mtime']}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)

def append_note():
    text = input("Type your note (blank cancels): ").strip()
    if not text:
        print("↩️  Cancelled.")
        return
    line = f"[{timestamp()}] {text}\n"
    with NOTES_PATH.open("a", encoding="utf-8") as f:
        f.write(line)
    write_hash()
    print(f"✅ Saved to {NOTES_PATH.name}")

def read_all_notes() -> list[str]:
    with NOTES_PATH.open("r", encoding="utf-8") as f:
        return f.readlines()

def view_notes():
    lines = read_all_notes()
    if not lines:
        print("ℹ️ No notes yet.")
        return
    print(f"\n—— {NOTES_PATH.name} ({len(lines)} lines) ——")
    for line in lines:
        print(line.rstrip())
    print("—— end ——\n")

def search_notes():
    query = input("Keyword (leave blank to skip): ").strip()
    date_str = input("Date YYYY-MM-DD (leave blank to skip): ").strip()

    date_filter = None
    if date_str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
            print("❌ Date must be YYYY-MM-DD (e.g., 2025-09-07).")
            return
        date_filter = date_str

    lines = read_all_notes()
    if not lines:
        print("ℹ️ No notes to search.")
        return

    hits = []
    for line in lines:
        match_keyword = (query.lower() in line.lower()) if query else True
        match_date = (date_filter in line[:12]) if date_filter else True  # "[YYYY-MM-DD"
        if match_keyword and match_date:
            hits.append(line.rstrip())

    if not hits:
        print("🔍 No matches.")
        return

    print(f"\n🔎 Matches ({len(hits)}):")
    for h in hits:
        print(h)
    print("—— end ——\n")

def verify_integrity():
    if not HASH_PATH.exists():
        print("ℹ️ No hash file yet – generating one now.")
        write_hash()
        print("✅ Baseline hash created. Run verify again if needed.")
        return
    current = sha256_of_path(NOTES_PATH)
    saved = HASH_PATH.read_text(encoding="utf-8").strip()
    if current == saved:
        print("🛡️  Integrity OK: notes.txt matches saved hash.")
    else:
        print("⚠️  WARNING: notes.txt hash does NOT match saved hash (file changed).")
        print(f"Saved:   {saved}")
        print(f"Current: {current}")
        choice = input("Update saved hash to current? (y/N): ").strip().lower()
        if choice == "y":
            write_hash()
            log_tamper(saved, current, action="updated_hash")
            print("✅ Hash updated.")
        else:
            log_tamper(saved, current, action="kept_old_hash")
            print("↩️  Kept old hash (mismatch recorded in tamper.log).")

def view_tamper_log():
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        print("ℹ️ No tamper events recorded.")
        return
    print(f"\n—— {LOG_PATH.name} ——")
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            print(line.rstrip())
    print("—— end ——\n")

def menu():
    print("\n=== USB Secure Notepad ===")
    print(f"Drive: {USB_PATH}")
    print("[1] Add note")
    print("[2] View all notes")
    print("[3] Search notes")
    print("[4] Verify integrity")
    print("[5] View tamper log")
    print("[6] Quit")
    return input("Select: ").strip()

def main():
    if not ensure_usb():
        sys.exit(1)
    ensure_files()

    while True:
        c = menu()
        if c == "1":
            append_note()
        elif c == "2":
            view_notes()
        elif c == "3":
            search_notes()
        elif c == "4":
            verify_integrity()
        elif c == "5":
            view_tamper_log()
        elif c == "6":
            print("👋 Bye!")
            break
        else:
            print("❓ Invalid option.")

if __name__ == "__main__":
    main()
