"""
Ghi song song stdout ra màn hình và ra file UTF-8.

Dùng để tạo evidence log thay cho `| tee` của PowerShell — PowerShell 5.1
ghi file ra UTF-16LE khiến log hiển thị sai trên GitHub.
"""
import sys
from pathlib import Path


class _Tee:
    def __init__(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(p, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


def enable_utf8():
    """Ép stdout sang UTF-8 để PowerShell không lỗi UnicodeEncodeError."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def start_log(path):
    """Bắt đầu ghi toàn bộ stdout vào file. Trả về đối tượng tee để đóng lại."""
    enable_utf8()
    tee = _Tee(path)
    sys.stdout = tee
    return tee


def stop_log(tee):
    """Ngừng ghi log và trả stdout về mặc định."""
    if tee is None:
        return
    sys.stdout = tee.stdout
    tee.close()