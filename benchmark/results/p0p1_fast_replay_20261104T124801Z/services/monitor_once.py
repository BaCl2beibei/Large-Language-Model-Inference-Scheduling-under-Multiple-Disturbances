from pathlib import Path
import json
import subprocess

run_dir = Path(__file__).resolve().parents[1]
meta = json.loads((run_dir / 'controller_supervisor.json').read_text())
for name in ('run_state.json', 'episode_ledger.jsonl'):
    path = run_dir / name
    print(f"{name}:\n{path.read_text()[-4000:] if path.exists() else 'not created'}")
for cmd in (['supervisorctl', 'status', meta['name']],
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader']):
    result = subprocess.run(cmd, capture_output=True, text=True)
    print('\n$', ' '.join(cmd), '\n', result.stdout, result.stderr)
