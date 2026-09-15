"""One-time migration of early pilot scores to the exact reference window count."""
import json
from concurrent.futures import ThreadPoolExecutor

from quality_metrics import DNSMOS, audio16
from quality_worker import ROOT, records


def main():
    model = DNSMOS()
    calibration_path = ROOT / 'quality/calibration.json'
    calibration = json.loads(calibration_path.read_text())
    calibration = [{**row, **model(audio16(row['path']))} for row in calibration]
    calibration_path.write_text(json.dumps(calibration, ensure_ascii=False, indent=2))
    generated = {r['id']: r for r in records(ROOT / 'logs/synthesis.jsonl') if r.get('status') == 'ok'}
    previous = {r['id']: r for r in records(ROOT / 'quality/metrics.jsonl')}
    pending = [r for r in previous.values() if r.get('dnsmos_protocol') != 'dns-challenge-window-count-v1']

    def score(row):
        return {**row, **model(audio16(generated[row['id']]['output_24k']))}

    with ThreadPoolExecutor(8) as pool, (ROOT / 'quality/metrics.jsonl').open('a', buffering=1) as stream:
        for index, row in enumerate(pool.map(score, pending), 1):
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            if index % 100 == 0:
                print(json.dumps({'refreshed': index, 'target': len(pending)}), flush=True)
    print(json.dumps({'status': 'complete', 'refreshed': len(pending)}), flush=True)


if __name__ == '__main__':
    main()
