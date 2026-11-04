"""Regression guards for altered source bytes, paper values and invalid responses."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import csv
import hashlib
import json
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'survey_code'))
from reproducibility import check_inputs, verify_outputs
from collect_api import prompts, validate
import pandas as pd


class ReproducibilityTests(unittest.TestCase):
    def test_changed_input_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'survey_data').mkdir()
            p = root/'survey_data/source.csv'
            p.write_bytes(b'value\n1\n')
            manifest = {'sha256': {'survey_data/source.csv': hashlib.sha256(p.read_bytes()).hexdigest()}}
            (root/'survey_data/input_manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(check_inputs(root)['files_verified'], 1)
            p.write_bytes(b'value\n2\n')
            with self.assertRaisesRegex(ValueError, 'source.csv'):
                check_inputs(root)

    def test_original_windows_prompt_and_invalid_response(self):
        train = pd.read_csv(ROOT/'survey_data/training.csv')
        with ZipFile(ROOT/'survey_data/api_raw/original_responses.zip') as z:
            record = json.loads(z.read('main/0039dd51c086424a0070e445bb6e4b27a3afdecc1c56902815e4f8decdd6180d.json'))
            self.assertEqual(prompts(record['mode'], record['role'], record['profile'], train), record['prompt'])
            self.assertIn('\r\n', record['prompt'])
            bad = json.loads(z.read('main/b5f38fe3000ec3443bf379ffcf8feeeaa2a755de9f0e80df9793cc3ec2a31e96.json'))
            with self.assertRaises(KeyError):
                validate(bad['response']['choices'][0]['message']['content'], bad['mode'], bad['profile'])

    def test_changed_paper_cell_is_rejected(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name in [f'table_{i:02d}.csv' for i in range(1, 15)] + ['正文数值.csv', 'figure_05_data.csv'] + [f'figure_{i:02d}.png' for i in range(1, 6)]:
                shutil.copyfile(ROOT/'paper_results'/name, out/name)
            self.assertEqual(verify_outputs(out)['status'], 'passed')
            p = out/'table_05.csv'
            with p.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.reader(stream))
            rows[1][2] = '9.9999'
            with p.open('w', encoding='utf-8-sig', newline='') as stream:
                csv.writer(stream).writerows(rows)
            with self.assertRaisesRegex(ValueError, '核对未通过'):
                verify_outputs(out)
            report = json.loads((out/'logs/verification.json').read_text())
            self.assertTrue(any(x['file'] == 'table_05.csv' and x['actual'] == '9.9999' for x in report['differences']))


if __name__ == '__main__':
    unittest.main()
