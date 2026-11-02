"""
Bo Theo Doi Trang Thai Xu Ly Du Lieu & Khong Trung Lap (Incremental State Tracker).
================================================================================
Kiem soat trang thai dau vao va dau ra cho moi module trong he thong AI Banking Assistant:
1. Tinh toan SHA-256 hash cua file, text hoac object dict dau vao.
2. So khanh voi so cai trang thai (State Ledger) da luu trong thu muc output.
3. Loai bo 100% viec tinh toan / goi API trung lap voi nhung gi da chay.
4. Chi thuc thi nhung gi moi hoac bi thay doi.

NGUYEN TAC BAT BUOC:
1. KHONG DUNG __INIT__.PY: Import truc tiep file module.
2. KHONG DUNG THU VIEN TYPING: Thuan tuy dung kieu nguyen ban cua Python.
3. ZERO TERNARY: Bat buoc dung if/else tuong minh, khong dung ?: hoac toan tu 3 ngoi.
4. 0% REGEX LAW: Tuyet doi khong dung regex.
"""

from __future__ import annotations
import os
import json
import time
import hashlib
from pathlib import Path


class StateTracker:
    """
    Bo theo doi trang thai xu ly du lieu va khu trung lap (0% Regex, Zero Ternary).
    """

    def __init__(self, state_file_path: str | Path):
        self.state_file_path = Path(state_file_path)
        self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict = {}
        self.load()

    def load(self) -> None:
        """Doc trang thai tu file JSON."""
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception:
                self.records = {}
        else:
            self.records = {}

    def save(self) -> None:
        """Ghi trang thai ra file JSON."""
        try:
            with open(self.state_file_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[StateTracker] Canh bao khong the luu state file {self.state_file_path}: {e}")

    @staticmethod
    def compute_file_hash(file_path: str | Path) -> str:
        """Tinh SHA-256 cua file tren dia."""
        p = Path(file_path)
        if not p.exists():
            return ""
        if not p.is_file():
            return ""

        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Tinh SHA-256 cua chuoi van ban."""
        if text is not None:
            clean = str(text).strip()
        else:
            clean = ""
        return hashlib.sha256(clean.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_dict_hash(data: dict) -> str:
        """Tinh SHA-256 cua doi tuong dict/json."""
        serialized = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def is_processed(self, item_id: str, content_hash: str) -> bool:
        """Kiem tra item_id da duoc xu ly voi content_hash nay chua."""
        if item_id in self.records:
            recorded = self.records[item_id]
            rec_hash = recorded.get("hash")
            if rec_hash == content_hash:
                return True
            else:
                return False
        else:
            return False

    def are_all_items_processed(self, items_dict: dict) -> bool:
        """
        Kiem tra mot danh sach cac cap {item_id: content_hash} da duoc xu ly het chua.
        Tra ve True neu tat ca deu da xu ly va hash khong thay doi, nguoc lai False.
        """
        if not items_dict:
            return False
        for item_id, content_hash in items_dict.items():
            if not self.is_processed(item_id, content_hash):
                return False
        return True

    def record_processed(self, item_id: str, content_hash: str, metadata: dict | None = None) -> None:
        """Ghi nhan item_id da xu ly thanh cong kem metadata."""
        if metadata is not None:
            meta = dict(metadata)
        else:
            meta = {}
        meta["hash"] = content_hash
        meta["processed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.records[item_id] = meta

    def get_record(self, item_id: str) -> dict | None:
        """Lay thong tin da ghi nhan cua item_id."""
        if item_id in self.records:
            return self.records[item_id]
        else:
            return None

    def clear(self) -> None:
        """Xoa toan bo trang thai (phuc vu chay force tu dau)."""
        self.records = {}
        self.save()
