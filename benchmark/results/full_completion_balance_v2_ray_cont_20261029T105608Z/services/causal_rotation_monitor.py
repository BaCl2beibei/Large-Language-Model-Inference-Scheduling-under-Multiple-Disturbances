import sys
from pathlib import Path
from datetime import datetime, timezone
import gzip, hashlib, json, os, signal, time, urllib.request

run = Path(sys.argv[1]).resolve()
runs = run / 'runs'
causal_dir = Path(json.loads((run / 'service_session.json').read_text())['causal_dir']).resolve()
manifest = run / 'causal_rotation_manifest.jsonl'
base_url = 'http://127.0.0.1:18080'
rotations = 0


def runner_pid():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = Path(f'/proc/{p.name}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        if 'benchmark/run_full_completion_balance.py' in cmd and str(run) in cmd:
            return int(p.name)
    return None


def is_stopped(pid):
    try:
        state = next(x for x in Path(f'/proc/{pid}/status').read_text().splitlines() if x.startswith('State:')).split()[1]
        return state in ('T', 't')
    except (OSError, StopIteration):
        return False


def episode_count():
    try:
        return len((run / 'episodes.jsonl').read_text().splitlines())
    except OSError:
        return 0


def append_manifest(row):
    with manifest.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _basic_ack_ok(ack):
    return (isinstance(ack, dict) and ack.get('status') == 'passed'
            and ack.get('durable') is True and ack.get('pending_delivery') == 0)


def post_flush():
    last_summary = {}
    for attempt in range(3):
        request = urllib.request.Request(base_url + '/real_control/flush', data=b'{}',
            headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read())
        cluster = payload.get('cluster') or {}
        api = payload.get('api') or {}
        manager = cluster.get('manager') or {}
        instances = list(cluster.get('instances') or [])
        instance_summaries = []
        instance_ok = bool(instances) and len(instances) == 4
        for instance in instances:
            local = instance.get('local') or {}
            workers = list(instance.get('workers') or [])
            state = instance.get('state') or {}
            this_ok = (_basic_ack_ok(local) and bool(workers)
                       and all(_basic_ack_ok(worker) for worker in workers)
                       and state.get('pending_delivery') == 0)
            instance_ok = instance_ok and this_ok
            instance_summaries.append({
                'local': {k: local.get(k) for k in ('status', 'durable', 'pending_delivery', 'reason')},
                'workers': [{k: w.get(k) for k in ('status', 'durable', 'pending_delivery', 'reason')} for w in workers],
                'state_pending_delivery': state.get('pending_delivery'),
            })
        ok = (_basic_ack_ok(manager) and instance_ok and _basic_ack_ok(api))
        last_summary = {
            'manager': {k: manager.get(k) for k in ('status', 'durable', 'pending_delivery', 'reason')},
            'instances': instance_summaries,
            'api': {k: api.get(k) for k in ('status', 'durable', 'pending_delivery', 'reason')},
            'attempt': attempt + 1,
        }
        if ok:
            return True, last_summary
        time.sleep(0.5)
    return False, last_summary


def open_writers(paths):
    wanted = {str(p): p.stat().st_ino for p in paths}
    found = set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        fdroot = proc / 'fd'
        try:
            fds = list(fdroot.iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target not in wanted:
                continue
            try:
                if (fdroot / fd.name).stat().st_ino == wanted[target]:
                    found.add(int(proc.name))
            except OSError:
                pass
    return found


def safe_prepared_boundary():
    active = [d for d in runs.iterdir() if d.is_dir() and not (d / 'episode.json').exists()]
    if not active:
        return True, 'between episodes'
    for d in active:
        journal = d / 'client_events.jsonl'
        if (journal.is_file() and journal.stat().st_size == 0
                and not (d / 'initial_flush.json').exists()):
            continue
        return False, 'next episode has advanced beyond pre-prepare'
    return True, 'next episode is before prepare'


def archive_and_truncate(paths, boundary_episode, writer_pids):
    staged = []
    for path in paths:
        size = path.stat().st_size
        if not size:
            continue
        st = path.stat()
        with path.open('rb') as source:
            source.seek(-1, 2)
            if source.read(1) != b'\n':
                raise RuntimeError('incomplete trailing JSONL record: ' + path.name)
        stem = path.name[:-len('.jsonl')]
        indexes = []
        for old in causal_dir.glob(stem + '-*.jsonl.gz'):
            try:
                indexes.append(int(old.name[len(stem) + 1:-len('.jsonl.gz')]))
            except ValueError:
                pass
        index = max(indexes, default=-1) + 1
        target = causal_dir / f'{stem}-{index:06d}.jsonl.gz'
        temporary = target.with_name(target.name + '.tmp')
        if target.exists() or temporary.exists():
            raise RuntimeError('archive name collision: ' + target.name)
        digest = hashlib.sha256()
        total = lines = 0
        sequences = []
        with path.open('rb') as source, temporary.open('wb') as raw_target:
            with gzip.GzipFile(fileobj=raw_target, mode='wb', compresslevel=1, mtime=0) as zipped:
                for line in source:
                    if not line.endswith(b'\n'):
                        raise RuntimeError('partial JSONL line: ' + path.name)
                    record = json.loads(line)
                    sequence = record.get('event_sequence')
                    if isinstance(sequence, int):
                        sequences.append(sequence)
                    digest.update(line)
                    total += len(line)
                    lines += 1
                    zipped.write(line)
            raw_target.flush()
            os.fsync(raw_target.fileno())
        uncompressed_sha = digest.hexdigest()
        restored_hash = hashlib.sha256()
        restored_bytes = 0
        with gzip.open(temporary, 'rb') as restored:
            for block in iter(lambda: restored.read(1024 * 1024), b''):
                restored_hash.update(block)
                restored_bytes += len(block)
        if restored_bytes != total or restored_hash.hexdigest() != uncompressed_sha:
            raise RuntimeError('gzip verification failed: ' + path.name)
        now = path.stat()
        if now.st_dev != st.st_dev or now.st_ino != st.st_ino or now.st_size != size:
            raise RuntimeError('source changed during archive: ' + path.name)
        staged.append({'path': path, 'target': target, 'temporary': temporary, 'size': size,
            'sha': uncompressed_sha, 'compressed_bytes': temporary.stat().st_size,
            'compressed_sha': hashlib.sha256(temporary.read_bytes()).hexdigest(), 'lines': lines,
            'sequence_min': min(sequences) if sequences else None,
            'sequence_max': max(sequences) if sequences else None,
            'dev': st.st_dev, 'ino': st.st_ino})
    try:
        for item in staged:
            item['temporary'].replace(item['target'])
        for item in staged:
            with item['path'].open('r+b', buffering=0) as source:
                current = os.fstat(source.fileno())
                if (current.st_dev != item['dev'] or current.st_ino != item['ino']
                        or current.st_size != item['size']):
                    raise RuntimeError('source identity changed before truncate: ' + item['path'].name)
                os.ftruncate(source.fileno(), 0)
                os.fsync(source.fileno())
            if item['path'].stat().st_size != 0:
                raise RuntimeError('truncate not confirmed: ' + item['path'].name)
        for item in staged:
            append_manifest({'recorded_utc': datetime.now(timezone.utc).isoformat(),
                'boundary_episode': boundary_episode, 'source_path': str(item['path'].relative_to(run)),
                'archive_path': str(item['target'].relative_to(run)), 'source_bytes': item['size'],
                'compressed_bytes': item['compressed_bytes'], 'source_sha256': item['sha'],
                'compressed_sha256': item['compressed_sha'], 'line_count': item['lines'],
                'event_sequence_min': item['sequence_min'], 'event_sequence_max': item['sequence_max'],
                'writer_pids': sorted(writer_pids), 'source_inode_preserved_and_truncated': True})
    except BaseException:
        for item in staged:
            item['temporary'].unlink(missing_ok=True)
            try:
                if item['path'].stat().st_size != 0:
                    item['target'].unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return staged


last_count = episode_count()
print('boundary rotation monitor active; completed episodes:', last_count, flush=True)
while True:
    if (run / 'service_cleanup.json').exists() or (run / 'screening_summary.json').exists():
        print('run finished; rotations:', rotations, flush=True)
        break
    count = episode_count()
    if count <= last_count:
        time.sleep(0.005)
        continue
    last_count = count
    pid = runner_pid()
    if pid is None:
        print('boundary observed but runner PID was not found; will wait for next boundary', flush=True)
        continue
    paused = []
    runner_paused = False
    try:
        os.kill(pid, signal.SIGSTOP)
        runner_paused = True
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and not is_stopped(pid):
            time.sleep(0.002)
        if not is_stopped(pid):
            raise RuntimeError('runner stop not confirmed')
        safe, reason = safe_prepared_boundary()
        if not safe:
            print('rotation skipped:', reason, 'episode-count', count, flush=True)
            continue
        flush_ok, flush_summary = post_flush()
        if not flush_ok:
            print('rotation skipped: producer flush or pending-delivery audit failed', json.dumps(flush_summary), flush=True)
            continue
        paths = sorted(causal_dir.glob('events_*.jsonl'))
        writers = open_writers(paths)
        if not paths or not writers:
            print('rotation skipped: causal streams or open writers not found', flush=True)
            continue
        for writer_pid in sorted(writers):
            try:
                os.kill(writer_pid, signal.SIGSTOP)
                paused.append(writer_pid)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and any(not is_stopped(x) for x in paused if Path(f'/proc/{x}').exists()):
            time.sleep(0.002)
        if any(not is_stopped(x) for x in paused if Path(f'/proc/{x}').exists()):
            raise RuntimeError('causal writer stop not confirmed')
        rows = archive_and_truncate(paths, f'episode_count_{count}', paused)
        reclaimed = sum(row['size'] - row['compressed_bytes'] for row in rows)
        rotations += 1
        print('rotated', len(rows), 'streams; reclaimed', reclaimed, 'bytes at completed-episode boundary',
            count, flush=True)
    except BaseException as exc:
        print('rotation failed safely:', type(exc).__name__, str(exc), flush=True)
    finally:
        for writer_pid in reversed(paused):
            try:
                os.kill(writer_pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        if runner_paused:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
    time.sleep(0.005)
