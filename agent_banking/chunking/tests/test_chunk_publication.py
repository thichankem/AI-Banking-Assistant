import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from docx import Document
from tokenizers import Tokenizer, models, pre_tokenizers

from agent_banking.chunking.run_chunking import load_published_chunks, run_auto_chunking


def make_tokenizer():
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


class ChunkPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.incoming = self.root / "input"
        self.incoming.mkdir()
        self.output = self.root / "output"
        self.options = {
            "cache_db_path": self.root / "cache.db",
            "require_ready": False,
            "use_llm": False,
            "use_decision": False,
            "tokenizer": make_tokenizer(),
            "retry_delays": (),
        }

    def write_document(self, body):
        source = self.incoming / "Hạn mức.docx"
        document = Document()
        document.add_paragraph(body)
        document.save(source)
        return source

    def test_clean_keeps_previous_snapshot_when_entire_rebuild_fails(self):
        source = self.write_document("Hạn mức là 10 triệu đồng.")
        expected = run_auto_chunking(self.incoming, self.output, **self.options)
        original_manifest = (self.output / "chunking_manifest.json").read_bytes()
        source.write_bytes(b"Not a Word package")
        with self.assertRaisesRegex(RuntimeError, "previous publication retained"):
            run_auto_chunking(self.incoming, self.output, clean_output=True, **self.options)
        self.assertEqual((self.output / "chunking_manifest.json").read_bytes(), original_manifest)
        self.assertEqual(load_published_chunks(self.output, require_ready=False), expected)

    def test_partial_clean_publishes_successes_and_keeps_old_artifacts(self):
        self.write_document("Hạn mức là 10 triệu đồng.")
        run_auto_chunking(self.incoming, self.output, **self.options)
        _, old = load_published_chunks(self.output, require_ready=False, return_manifest=True)
        self.write_document("Hạn mức là 15 triệu đồng.")
        (self.incoming / "Broken.pdf").write_bytes(b"Not a PDF")
        expected = run_auto_chunking(self.incoming, self.output, clean_output=True, **self.options)
        published, manifest = load_published_chunks(self.output, require_ready=False, return_manifest=True)
        self.assertEqual(published, expected)
        self.assertEqual(manifest["failed_files"], 1)
        self.assertTrue((self.output / old["active_aggregate_file"]).is_file())
        self.assertTrue((self.output / old["records"][0]["output_file"]).is_file())

    def test_clean_collects_obsolete_files_after_commit_and_keeps_runtime_and_manual_edits(self):
        self.write_document("Hạn mức là 10 triệu đồng.")
        run_auto_chunking(self.incoming, self.output, **self.options)
        _, old = load_published_chunks(self.output, require_ready=False, return_manifest=True)
        for name in ("runtime", "manual_edits", "assets"):
            directory = self.output / name
            directory.mkdir(exist_ok=True)
            (directory / "notes.txt").write_text("user file", encoding="utf-8")
        self.write_document("Hạn mức là 15 triệu đồng.")
        expected = run_auto_chunking(self.incoming, self.output, clean_output=True, **self.options)
        self.assertEqual(load_published_chunks(self.output, require_ready=False), expected)
        self.assertFalse((self.output / old["active_aggregate_file"]).exists())
        self.assertFalse((self.output / old["records"][0]["output_file"]).exists())
        for name in ("runtime", "manual_edits", "assets"):
            self.assertTrue((self.output / name / "notes.txt").is_file())
        with patch("agent_banking.chunking.run_chunking._load_snapshot",
                   side_effect=[FileNotFoundError("old snapshot"), expected]) as load:
            manifest_path = self.output / "chunking_manifest.json"
            with patch.object(Path, "read_bytes",
                              side_effect=[json.dumps(old).encode(), manifest_path.read_bytes()]):
                self.assertEqual(load_published_chunks(self.output, require_ready=False), expected)
            self.assertEqual(load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
