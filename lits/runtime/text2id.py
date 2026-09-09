"""Map C++ G2P token lines to the acoustic model's numeric input contract.

This module intentionally depends only on the Python standard library. It must
remain safe to import in deployment without loading the legacy cleaner, G2P, or
dictionary stack under :mod:`lits.text`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EncodedPhonemes:
    phonemes: str
    token_ids: list[int]
    tone_ids: list[int] | None

    def to_record(self) -> dict[str, Any]:
        return {
            "phonemes": self.phonemes,
            "token_ids": self.token_ids,
            "tone_ids": self.tone_ids,
        }

    @classmethod
    def from_record(cls, payload: Any, *, line_no: int) -> "EncodedPhonemes":
        if not isinstance(payload, dict):
            raise ValueError(f"token ID JSONL line {line_no} must be an object")
        phonemes = payload.get("phonemes", "")
        token_ids = payload.get("token_ids")
        tone_ids = payload.get("tone_ids")
        if not isinstance(phonemes, str):
            raise ValueError(f"token ID JSONL line {line_no} phonemes must be a string")
        if not isinstance(token_ids, list) or not token_ids or not all(
            isinstance(value, int) and value >= 0 for value in token_ids
        ):
            raise ValueError(
                f"token ID JSONL line {line_no} token_ids must be a non-empty "
                "list of non-negative integers"
            )
        if tone_ids is not None:
            if not isinstance(tone_ids, list) or len(tone_ids) != len(token_ids) or not all(
                isinstance(value, int) and value >= 0 for value in tone_ids
            ):
                raise ValueError(
                    f"token ID JSONL line {line_no} tone_ids must be null or match token_ids"
                )
        return cls(phonemes=phonemes, token_ids=token_ids, tone_ids=tone_ids)


class Text2Id:
    """Immutable token inventory loaded from a versioned JSON resource."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported Text2Id schema_version in {self.config_path}: "
                f"{payload.get('schema_version')!r}"
            )

        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or not tokens or not all(isinstance(x, str) for x in tokens):
            raise ValueError(f"Text2Id tokens must be a non-empty string list: {self.config_path}")
        if len(tokens) != len(set(tokens)):
            raise ValueError(f"Text2Id tokens contain duplicates: {self.config_path}")
        configured_vocab = payload.get("n_vocab")
        if configured_vocab != len(tokens):
            raise ValueError(
                f"Text2Id n_vocab={configured_vocab!r} does not match "
                f"tokens={len(tokens)}: {self.config_path}"
            )

        self.model_key = str(payload.get("model_key", ""))
        self.tokens = tuple(tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
        self.n_vocab = len(tokens)
        self.blank_id = self._required_token_id(payload.get("blank_token"), "blank_token")
        self.unknown_id = self._required_token_id(payload.get("unknown_token"), "unknown_token")
        self.silence_id = self.token_to_id.get("<sil>")
        self.emit_tone_ids = bool(payload.get("emit_tone_ids", False))

        raw_tone_marks = payload.get("tone_marks", {})
        if not isinstance(raw_tone_marks, dict):
            raise ValueError(f"Text2Id tone_marks must be an object: {self.config_path}")
        self.tone_marks: dict[str, int] = {}
        for mark, tone_id in raw_tone_marks.items():
            if not isinstance(mark, str) or not isinstance(tone_id, int) or tone_id <= 0:
                raise ValueError(f"Invalid Text2Id tone mark {mark!r}: {tone_id!r}")
            self.tone_marks[mark] = tone_id

    def _required_token_id(self, token: Any, field: str) -> int:
        if not isinstance(token, str) or token not in self.token_to_id:
            raise ValueError(
                f"Text2Id {field}={token!r} is missing from tokens: {self.config_path}"
            )
        return self.token_to_id[token]

    @staticmethod
    def _intersperse(values: list[int], separator: int) -> list[int]:
        output = [separator]
        for value in values:
            output.extend((value, separator))
        return output

    def encode(
        self,
        phonemes: str,
        *,
        add_blank: bool = False,
        prepend_sil: bool = False,
    ) -> EncodedPhonemes:
        clean = phonemes.strip()
        token_ids: list[int] = []
        tone_ids: list[int] | None = [] if self.emit_tone_ids else None

        for token in clean.split():
            token_ids.append(self.token_to_id.get(token, self.unknown_id))
            if tone_ids is not None:
                if token in self.tone_marks and tone_ids:
                    tone_ids[-1] = self.tone_marks[token]
                tone_ids.append(0)

        if not token_ids:
            raise ValueError("C++ G2P produced an empty phoneme token sequence")
        if prepend_sil:
            if self.silence_id is None:
                raise ValueError("Text2Id inventory has no <sil> token")
            if token_ids[0] != self.silence_id:
                token_ids.insert(0, self.silence_id)
                clean = f"<sil> {clean}"
                if tone_ids is not None:
                    tone_ids.insert(0, 0)
        if add_blank:
            token_ids = self._intersperse(token_ids, self.blank_id)
            if tone_ids is not None:
                tone_ids = self._intersperse(tone_ids, 0)
        return EncodedPhonemes(clean, token_ids, tone_ids)

    def validate_model_vocab(self, model_n_vocab: int) -> None:
        if model_n_vocab != self.n_vocab:
            raise ValueError(
                f"Text2Id inventory has n_vocab={self.n_vocab}, but model has "
                f"n_vocab={model_n_vocab}"
            )


def write_jsonl(path: str | Path, rows: list[EncodedPhonemes]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(row.to_record(), ensure_ascii=False, separators=(",", ":"))
        for row in rows
    )
    output_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[EncodedPhonemes]:
    input_path = Path(path)
    rows: list[EncodedPhonemes] = []
    for line_no, raw in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid token ID JSONL at line {line_no}: {exc}") from exc
        rows.append(EncodedPhonemes.from_record(payload, line_no=line_no))
    if not rows:
        raise ValueError(f"token ID JSONL contains no records: {input_path}")
    return rows
