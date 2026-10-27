"""bench/lib/promote.py: turning a measured variant into production."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "lib"))

from promote import PromoteError, load_measured  # noqa: E402
from stack import Variant  # noqa: E402

Q8V = Variant("q8V", entries={"m": {"extra_args": ["--cache-type-v q8_0"]}})


def row(model="m", variant=Q8V, status="OK", digest="d" * 64, cmd="/opt/e/llama-server -m m.gguf", **extra):
    return {"model": model, "variant": variant.name, "depth": "4096", "status": status,
            "patch": json.dumps(variant.to_dict(), sort_keys=True), "rendered_cmd": cmd,
            "production": "active", "base_digest": digest, **extra}


class LoadMeasuredTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def tsv(self, rows, name="sweep.tsv"):
        path = self.root / name
        columns = list(dict.fromkeys(k for r in rows for k in r))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def kv_json(self, provenance, model="m", records=1, name="kv.json"):
        path = self.root / name
        path.write_text(json.dumps({"tag": "t", "model": model, "provenance": provenance, "summary": {},
                                    "records": [{"depth": 4096, "rep": i + 1} for i in range(records)]}))
        return path

    def test_tsv_variant_rows_agree(self):
        path = self.tsv([row(variant=Variant()), row(), row(depth="65536")])
        measured = load_measured(path, "q8V")
        self.assertEqual((measured.model, measured.variant.entries, measured.base_digest, measured.rows),
                         ("m", Q8V.entries, "d" * 64, 2))
        self.assertEqual(measured.rendered_cmd, "/opt/e/llama-server -m m.gguf")

    def test_failed_rows_do_not_count(self):
        path = self.tsv([row(status="LOAD_FAIL"), row(status="REQUEST_FAIL")])
        with self.assertRaisesRegex(PromoteError, "no OK rows for variant q8V"):
            load_measured(path, "q8V")

    def test_unknown_variant_names_the_ones_present(self):
        path = self.tsv([row(variant=Variant()), row()])
        with self.assertRaisesRegex(PromoteError, r"no variant 'fast'.*production, q8V"):
            load_measured(path, "fast")

    def test_rows_that_disagree_are_refused(self):
        for field, value in (("base_digest", "e" * 64), ("rendered_cmd", "/other"),
                             ("patch", json.dumps(Variant("q8V", entries={"m": {"ctx": 1}}).to_dict()))):
            with self.subTest(field=field):
                path = self.tsv([row(), {**row(), field: value}])
                with self.assertRaisesRegex(PromoteError, f"disagree on {field}"):
                    load_measured(path, "q8V")

    def test_results_without_base_digest_are_refused(self):
        old = {k: v for k, v in row().items() if k != "base_digest"}
        for path in (self.tsv([old]), self.tsv([row(digest="")], name="empty.tsv")):
            with self.subTest(path=path.name), self.assertRaisesRegex(PromoteError, "no base digest.*re-run"):
                load_measured(path, "q8V")

    def test_two_models_need_model(self):
        path = self.tsv([row(model="m"), row(model="n")])
        with self.assertRaisesRegex(PromoteError, r"several models.*--model.*m, n"):
            load_measured(path, "q8V")
        self.assertEqual(load_measured(path, "q8V", model="n").model, "n")
        with self.assertRaisesRegex(PromoteError, "no OK rows for variant q8V of model x"):
            load_measured(path, "q8V", model="x")

    def test_kv_probe_json(self):
        provenance = {"mode": "stack", "variant": "q8V", "patch": json.dumps(Q8V.to_dict(), sort_keys=True),
                      "rendered_cmd": "/opt/e/llama-server", "production": "active", "base_digest": "d" * 64}
        measured = load_measured(self.kv_json(provenance, records=2), "q8V")
        self.assertEqual((measured.model, measured.rows, measured.variant.entries), ("m", 2, Q8V.entries))
        with self.assertRaisesRegex(PromoteError, "no variant 'other'.*q8V"):
            load_measured(self.kv_json(provenance), "other")
        with self.assertRaisesRegex(PromoteError, "no OK rows"):
            load_measured(self.kv_json(provenance, records=0), "q8V")

    def test_standalone_results_are_refused(self):
        path = self.kv_json({"mode": "standalone", "server": "http://127.0.0.1:8080"})
        with self.assertRaisesRegex(PromoteError, "standalone.*re-measure.*--variant"):
            load_measured(path, "production")


if __name__ == "__main__":
    unittest.main()
