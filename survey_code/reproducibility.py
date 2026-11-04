"""Audit original inputs, API responses and manuscript-level results offline."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def check_inputs(root=ROOT):
    manifest = json.loads((root/'survey_data/input_manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['sha256'].items():
        path = root/relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError('原始输入校验失败：' + relative)
    return {'files_verified': len(manifest['sha256']),
            'manifest_sha256': sha256(root/'survey_data/input_manifest.json')}


def audit_original_api(output, root=ROOT):
    """Reparse actual responses; do not call a model or repair failed responses."""
    from collect_api import validate, prompts, payload
    output, data = Path(output), root/'survey_data'
    metadata = json.loads((data/'api_metadata.json').read_text(encoding='utf-8-sig'))
    index = json.loads((data/'api_raw/response_index.json').read_text(encoding='utf-8'))
    train, expected = pd.read_csv(data/'training.csv'), pd.read_csv(data/'api_records.csv')
    rows, counts, times = [], Counter(), {'main': [], 'pilot': []}
    phase_hashes = {'main': set(), 'pilot': set()}
    with ZipFile(data/'api_raw/original_responses.zip') as archive:
        if len(index) != 8740 or set(archive.namelist()) != {x['file'] for x in index}:
            raise ValueError('原始响应文件集合与清单不一致。')
        for entry in index:
            raw = archive.read(entry['file'])
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('原始响应校验失败：' + entry['file'])
            record = json.loads(raw.decode('utf-8-sig'))
            phase = entry['file'].split('/')[0]
            if record['hash'] in phase_hashes[phase]:
                raise ValueError('原始响应存在重复请求。')
            phase_hashes[phase].add(record['hash'])
            if record['hash'] != entry['hash'] or record['status'] != entry['status']:
                raise ValueError('响应索引与原始记录不一致。')
            prompt = prompts(record['mode'], record['role'], record['profile'], train)
            if prompt != record['prompt']:
                raise ValueError('无法重建原始提示：' + record['hash'])
            body = payload(metadata['model'], prompt, metadata['thinking'], metadata['temperature'],
                           metadata['max_tokens'], metadata['reasoning_effort'])
            identity = {'body': body, 'decode': record['decode'], 'record': record['profile_id'],
                        'mode': record['mode'], 'role': record['role']}
            if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != record['hash']:
                raise ValueError('原始请求身份哈希不一致。')
            try:
                parsed = validate(record['response']['choices'][0]['message']['content'],
                                  record['mode'], record['profile'])
                status = 'ok'
            except (ValueError, KeyError, TypeError):
                parsed, status = {}, 'invalid_output'
            if status != record['status'] or (status == 'ok' and parsed != record['parsed']):
                raise ValueError('重新解析与原始状态不一致：' + record['hash'])
            counts[(phase, status)] += 1
            times[phase].append(record['timestamp_utc'])
            if phase == 'main':
                row = {key: record.get(key) for key in
                       ['hash', 'mode', 'role', 'decode', 'profile_id', 'status', 'returned_model', 'seconds']}
                row.update(parsed)
                rows.append(row)
    if counts != Counter({('main', 'ok'): 8707, ('main', 'invalid_output'): 1, ('pilot', 'ok'): 32}):
        raise ValueError('原始请求数量或状态与论文不一致。')
    for phase in ['main', 'pilot']:
        plan = json.loads((data/f'api_raw/{phase}_api_plan.json').read_text(encoding='utf-8-sig'))
        if phase_hashes[phase] != set(plan['planned_hashes']):
            raise ValueError('原始请求与' + phase + '计划不一致。')
    if phase_hashes['main'] != set(metadata['planned_hashes']):
        raise ValueError('分析用请求清单与原始正式请求不一致。')
    rebuilt = pd.DataFrame(rows).set_index('hash').reindex(expected.hash).reset_index()[expected.columns]
    pd.testing.assert_frame_equal(rebuilt, expected, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    (output/'details').mkdir(parents=True, exist_ok=True)
    rebuilt.to_csv(output/'details/rebuilt_api_records.csv', index=False, encoding='utf-8-sig')
    report = {'status': 'passed', 'network_requests': 0, 'main_requests': 8708, 'main_valid': 8707,
              'main_invalid': 1, 'pilot_requests': 32, 'pilot_valid': 32,
              'raw_prompts_and_request_hashes_reconstructed': True,
              'parsed_records_match_analysis_csv': True,
              'timestamps_utc': {k: {'first': min(v), 'last': max(v)} for k, v in times.items()}}
    write_json(output/'logs/api_replay.json', report)
    return report


def environment():
    packages = {name: importlib.metadata.version(name) for name in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib']}
    return {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
            'platform': platform.platform(), 'machine': platform.machine(), 'packages': packages,
            'rng': 'numpy.random.default_rng / PCG64', 'seed_registry': 'docs/randomness.md',
            'api_mode': 'original cached responses; no requests'}


def normalized(value):
    return re.sub(r'\s+', '', value).replace('−', '-').replace('合成', '生成')


def verify_outputs(output, root=ROOT):
    """Exact displayed table cells and full-precision body-value regression."""
    reference = json.loads((root/'survey_data/paper_reference.json').read_text(encoding='utf-8'))
    differences, missing = [], []
    for name, expected in reference['tables'].items():
        path = output/name
        if not path.is_file():
            missing.append(name)
            continue
        with path.open(encoding='utf-8-sig', newline='') as stream:
            actual = list(csv.reader(stream))
        if len(actual) != len(expected) or any(len(a) != len(b) for a, b in zip(actual, expected)):
            differences.append({'file': name, 'type': 'table_shape'})
            continue
        for i, (a, b) in enumerate(zip(actual, expected), 1):
            for j, (value, wanted) in enumerate(zip(a, b), 1):
                if normalized(value) != normalized(wanted):
                    differences.append({'file': name, 'row': i, 'column': j, 'reference': wanted, 'actual': value})
    body = pd.read_csv(output/'正文数值.csv')
    expected_body = pd.DataFrame(reference['body_values'])
    try:
        for col in ['部分', '指标', '条件', '单位']:
            pd.testing.assert_series_equal(body[col].fillna('').map(normalized),
                                           expected_body[col].fillna('').map(normalized), check_names=False)
        pd.testing.assert_series_equal(body['数值'], expected_body['数值'], check_names=False,
                                       check_exact=False, rtol=1e-12, atol=1e-12)
    except AssertionError:
        differences.append({'file': '正文数值.csv', 'type': 'body_value_mismatch'})
    try:
        pd.testing.assert_frame_equal(pd.read_csv(output/'figure_05_data.csv'),
                                      pd.DataFrame(reference['figure_05_data']), check_dtype=False,
                                      check_exact=False, rtol=1e-12, atol=1e-12)
    except AssertionError:
        differences.append({'file': 'figure_05_data.csv', 'type': 'plot_data_mismatch'})
    for name in [f'figure_{i:02d}.png' for i in range(1, 6)]:
        path = output/name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
    report = {'status': 'passed' if not differences and not missing else 'failed',
              'tables_checked': 14, 'figures_checked': 5, 'body_values_checked': len(body),
              'figure_05_data_checked': True,
              'table_comparison': 'exact displayed cells, ignoring whitespace and terminology-only changes',
              'body_numeric_tolerance': {'rtol': 1e-12, 'atol': 1e-12},
              'reference': reference['manuscript'], 'differences': differences, 'missing': missing,
              'figure_check': 'file existence; numerical sources mapped in docs/result_map.md'}
    write_json(output/'logs/verification.json', report)
    if report['status'] != 'passed':
        raise ValueError('论文结果核对未通过，详见 logs/verification.json。')
    return report
