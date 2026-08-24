"""
Bước 4 — Guardrails AI Validators
===================================
1. PIIDetector  - phát hiện và che 4 loại PII bằng regex
2. JSONFormatter - kiểm tra và tự sửa JSON lỗi từ đầu ra LLM

Lưu ý quan trọng về Guardrails:
  - on_fail phải truyền vào CONSTRUCTOR của validator:
        Guard().use(PIIDetector(on_fail=OnFailAction.FIX))    ĐÚNG
        Guard().use(PIIDetector(), on_fail=OnFailAction.FIX)  SAI
  - Muốn FIX thực sự thay thế được output thì validate() phải trả về
        FailResult(error_message=..., fix_value=<gia tri da sua>)
    Trả PassResult(value_override=...) sẽ không đổi validated_output.

Cách dùng:
    python 04_guardrails_validator.py            # chạy cả 2 demo, ghi 2 file log
    python 04_guardrails_validator.py --demo pii
    python 04_guardrails_validator.py --demo json
"""
import re
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from guardrails import Guard, settings
from guardrails.validators import Validator, register_validator, PassResult, FailResult
from guardrails.validator_base import OnFailAction

from utils.tee import enable_utf8, start_log, stop_log

# Tắt telemetry của Guardrails để log evidence không lẫn lỗi OTLP
settings.disable_tracing = True

EVIDENCE_DIR = Path(__file__).parent.parent / "evidence"


# ── 1. PII Detector ────────────────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Phát hiện thông tin cá nhân bằng regex và thay bằng placeholder an toàn.

    Bốn loại PII được nhận diện: email, số điện thoại, SSN, số thẻ tín dụng.
    Thứ tự thay thế có ý nghĩa: SSN và thẻ tín dụng phải chạy trước số điện
    thoại, nếu không pattern điện thoại sẽ ăn mất một phần của chúng.
    """

    PII_PATTERNS = [
        ("EMAIL",       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        ("CREDIT_CARD", r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
        ("SSN",         r"\b\d{3}-\d{2}-\d{4}\b"),
        ("PHONE",       r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    ]

    def validate(self, value: str, metadata: dict) -> PassResult | FailResult:
        redacted = value
        found    = []

        for label, pattern in self.PII_PATTERNS:
            redacted, n = re.subn(pattern, f"[{label}_REDACTED]", redacted)
            if n:
                found.append(f"{label} x{n}")

        if found:
            return FailResult(
                error_message=f"Phat hien PII: {', '.join(found)}",
                fix_value=redacted,
            )
        return PassResult()


# ── 2. JSON Formatter ──────────────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Kiểm tra chuỗi có parse được thành JSON không, nếu không thì tự sửa.

    Ba phép sửa được áp dụng theo thứ tự:
      1. Gỡ markdown code fences (```json ... ```)
      2. Đổi nháy đơn thành nháy kép
      3. Xóa dấu phẩy thừa trước } hoặc ]

    Nếu vẫn không parse được thì trả về JSON dự phòng thay vì ném lỗi.
    """

    FALLBACK = {"error": "invalid_json", "message": "Khong the sua chuoi thanh JSON hop le"}

    def _repair(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)   # gỡ fence mở
        text = re.sub(r"\s*```$", "", text).strip()     # gỡ fence đóng
        text = text.replace("'", '"')                   # nháy đơn -> nháy kép
        text = re.sub(r",\s*([}\]])", r"\1", text)      # xóa dấu phẩy thừa
        return text

    def validate(self, value: str, metadata: dict) -> PassResult | FailResult:
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass

        repaired = self._repair(value)
        try:
            parsed = json.loads(repaired)
            return FailResult(
                error_message="JSON loi da duoc tu sua",
                fix_value=json.dumps(parsed, ensure_ascii=False, indent=2),
            )
        except json.JSONDecodeError as e:
            return FailResult(
                error_message=f"Khong the sua duoc: {e}",
                fix_value=json.dumps(self.FALLBACK, ensure_ascii=False, indent=2),
            )


# ── 3. Demo PII Detector ───────────────────────────────────────────────────
PII_TEST_CASES = [
    ("Dau vao sach, khong co PII",
     "Machine learning models learn patterns from training data."),
    ("Email",
     "Please contact our support team at support@example.com for help."),
    ("So dien thoai",
     "Call me back at 555-123-4567 tomorrow morning."),
    ("SSN",
     "The customer SSN is 123-45-6789 in our records."),
    ("The tin dung",
     "Charge the card 4532 1234 5678 9010 for this order."),
    ("Nhieu loai PII cung luc",
     "Reach John at john.doe@corp.com or 555-987-6543, SSN 987-65-4321."),
]


def demo_pii():
    print("=" * 70)
    print("  DEMO 1: PII Detector")
    print("=" * 70)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    for i, (label, text) in enumerate(PII_TEST_CASES, 1):
        result = guard.validate(text)
        output = result.validated_output
        changed = output != text

        print(f"\n[Test {i}] {label}")
        print(f"  Input : {text}")
        print(f"  Output: {output}")
        print(f"  Trang thai: {'DA CHE PII' if changed else 'SACH - giu nguyen'}")

    print(f"\nTong cong {len(PII_TEST_CASES)} test case.")


# ── 4. Demo JSON Formatter ─────────────────────────────────────────────────
JSON_TEST_CASES = [
    ("JSON hop le",
     '{"name": "Alice", "age": 30}'),
    ("Bi boc trong markdown fences",
     '```json\n{"city": "Hanoi", "country": "Vietnam"}\n```'),
    ("Dung nhay don",
     "{'model': 'gpt-oss-20b', 'provider': 'groq'}"),
    ("Co dau phay thua",
     '{"a": 1, "b": 2,}'),
    ("Ket hop fences + nhay don + phay thua",
     "```json\n{'x': 1, 'y': 2,}\n```"),
    ("Hoan toan sai dinh dang",
     "This is definitely not JSON at all {]"),
]


def demo_json():
    print("=" * 70)
    print("  DEMO 2: JSON Formatter")
    print("=" * 70)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    for i, (label, text) in enumerate(JSON_TEST_CASES, 1):
        result = guard.validate(text)
        output = result.validated_output

        try:
            json.loads(output)
            parseable = True
        except Exception:
            parseable = False

        if output == text:
            status = "HOP LE - giu nguyen"
        elif "invalid_json" in output:
            status = "KHONG SUA DUOC - tra ve JSON du phong"
        else:
            status = "DA TU SUA"

        print(f"\n[Test {i}] {label}")
        print(f"  Input : {text!r}")
        print(f"  Output: {output}")
        print(f"  Trang thai: {status} | parse duoc: {parseable}")

    print(f"\nTong cong {len(JSON_TEST_CASES)} test case.")


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Guardrails AI validators")
    parser.add_argument("--demo", choices=["pii", "json"], help="Chi chay mot demo")
    args, _ = parser.parse_known_args()   # bo qua --step cua run_all.py
    
    enable_utf8()

    if args.demo in (None, "pii"):
        log = start_log(EVIDENCE_DIR / "04_pii_demo_log.txt")
        try:
            demo_pii()
        finally:
            stop_log(log)

    if args.demo in (None, "json"):
        log = start_log(EVIDENCE_DIR / "04_json_demo_log.txt")
        try:
            demo_json()
        finally:
            stop_log(log)

    print("\nBuoc 4 hoan thanh. Log da ghi vao evidence/")


if __name__ == "__main__":
    main()