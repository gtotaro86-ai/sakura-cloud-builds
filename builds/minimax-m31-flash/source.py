"""Pinned public sources. Model bytes only through exact bounded HTTP ranges."""
import collections, fcntl, hashlib, json, os, shutil, struct, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = 'MiniMaxAI/MiniMax-M3.1-Flash'
REV = '54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e'
LLAMA = '970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc'
ROOT = Path(__file__).resolve().parent
CHUNK = 64 * 1024**2

def digest(data):
    return hashlib.sha256(data).hexdigest()

def inventory():
    return json.loads((ROOT/'source-inventory.json').read_text())

def fetch(path, start, count, total, work, budget=100_000_000):
    if not 0 <= start < start+count <= total or count > CHUNK:
        raise ValueError('Exact bounded range required')
    journal = Path(work)/'range-budget.json'
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        state = json.loads(journal.read_text()) if journal.exists() else {'reserved_bytes': 0, 'requests': 0}
        state['reserved_bytes'] += count
        state['requests'] += 1
        if state['reserved_bytes'] > budget:
            raise RuntimeError('Source byte budget exhausted')
        temp = journal.with_suffix('.tmp')
        temp.write_text(json.dumps(state)); os.replace(temp, journal)
    end = start+count-1
    url = f'https://huggingface.co/{REPO}/resolve/{REV}/{path}?sakura_minimax={start}-{end}'
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(req, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f'HTTP {response.status}: whole response refused before body')
        if response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
            raise RuntimeError('Content-Range mismatch')
        if response.headers.get('Content-Encoding', 'identity') != 'identity':
            raise RuntimeError('Encoded response refused')
        if response.headers.get('Content-Length') not in (None, str(count)):
            raise RuntimeError('Content-Length mismatch')
        raw = response.read(count+1)
        if len(raw) != count:
            raise RuntimeError('Short/excess range')
        return raw

def metadata(work):
    work = Path(work); model = work/'model'; model.mkdir(parents=True, exist_ok=True)
    inv = inventory()
    for spec in inv['metadata']:
        dst = model/spec['path']
        if not dst.exists():
            url = f"https://huggingface.co/{REPO}/resolve/{REV}/{spec['path']}"
            with urllib.request.urlopen(url, timeout=120) as r:
                raw = r.read(spec['size']+1)
            if len(raw) != spec['size'] or digest(raw) != spec['sha256']:
                raise RuntimeError('Pinned metadata size/hash mismatch: '+spec['path'])
            dst.write_bytes(raw)
        if digest(dst.read_bytes()) != spec['sha256']:
            raise RuntimeError('Changed local metadata')
    return model

def headers(work):
    work = Path(work); folder = work/'headers'; folder.mkdir(parents=True, exist_ok=True)
    specs = {}
    index = json.loads((metadata(work)/'model.safetensors.index.json').read_text())['weight_map']
    for spec in inventory()['weights']:
        name = spec['path']; dst = folder/(name+'.json')
        if dst.exists():
            raw = dst.read_bytes()
        else:
            first = fetch(name, 0, 8, spec['size'], work)
            n = struct.unpack('<Q', first)[0]
            if not 2 <= n <= 5*1024**2:
                raise RuntimeError('Unbounded safetensors header')
            raw = fetch(name, 8, n, spec['size'], work)
            dst.write_bytes(raw)
        if digest(raw) != spec['header_sha256'] or len(raw) != spec['header_size']:
            raise RuntimeError('Frozen safetensors header mismatch: '+name)
        head = json.loads(raw); offset = 8+len(raw)
        for tensor, entry in head.items():
            if tensor == '__metadata__': continue
            a,b = entry['data_offsets']
            if tensor in specs or index.get(tensor) != name or not 0 <= a < b <= spec['size']-offset:
                raise RuntimeError('Invalid/duplicate source tensor')
            if entry['dtype'] not in ('BF16','F32','F8_E4M3','U8'):
                raise RuntimeError('Original 16-bit/F32 source required')
            size = 1
            for dim in entry['shape']: size *= dim
            if b-a != size*{'BF16':2,'F32':4,'F8_E4M3':1,'U8':1}[entry['dtype']]:
                raise RuntimeError('Tensor byte count mismatch')
            specs[tensor] = dict(entry, path=name, start=offset+a, size=b-a)
    if set(specs) != set(index): raise RuntimeError('Index/header coverage mismatch')
    return specs

class VerifiedShardCache:
    """One full verified source shard on the GitHub runner, never on the user PC."""
    def __init__(self, work):
        if (os.environ.get('GITHUB_ACTIONS') != 'true' or os.environ.get('RUNNER_ENVIRONMENT') != 'github-hosted'
                or os.environ.get('RUNNER_OS') != 'Linux' or os.environ.get('REPO_VISIBILITY') != 'public'):
            raise RuntimeError('Full source reads require an actual public GitHub-hosted Linux runner')
        self.work = Path(work); self.current = None; self.verified = {}; self.loads = 0
        self.small = {}
        self.small_specs = collections.defaultdict(list)
        for item in headers(work).values():
            if item['size'] <= 65536:self.small_specs[item['path']].append(item)
        self.files = {x['path']: x for x in inventory()['weights']}
        self.directory = self.work/'shard-cache'; self.directory.mkdir(exist_ok=True)
        self.file = self.directory/'source.safetensors'

    def read(self, spec):
        name = spec['path'];key=(name,spec['start'],spec['size'])
        if key in self.small:return bytearray(self.small[key])
        if self.current != name:
            self.current = None; self.file.unlink(missing_ok=True)
            item = self.files[name]
            if shutil.disk_usage(self.directory).free < item['size']+512*1024**2:
                raise RuntimeError('Insufficient runner disk for one bounded source shard')
            h = hashlib.sha256(); tick = time.perf_counter()
            def read_chunk(start):
                count=min(CHUNK,item['size']-start)
                for attempt in range(3):
                    try:return start,fetch(name,start,count,item['size'],self.work,750_000_000_000)
                    except (OSError,TimeoutError):
                        if attempt==2:raise
                        time.sleep(2*(attempt+1))
            starts=iter(range(0,item['size'],CHUNK))
            with self.file.open('wb') as out, ThreadPoolExecutor(max_workers=4) as pool:
                pending=collections.deque(pool.submit(read_chunk,s) for s in [next(starts,None) for _ in range(4)] if s is not None)
                while pending:
                    start,raw=pending.popleft().result();count=len(raw)
                    out.write(raw); h.update(raw)
                    if start//CHUNK % 16 == 0:
                        print(json.dumps({'source_shard': name, 'bytes': start+count, 'size': item['size']}), file=__import__('sys').stderr, flush=True)
                    next_start=next(starts,None)
                    if next_start is not None:pending.append(pool.submit(read_chunk,next_start))
            if h.hexdigest() != item['sha256']:
                self.file.unlink(); raise RuntimeError('Complete original safetensors SHA256 mismatch')
            self.current = name; self.verified[name] = h.hexdigest(); self.loads += 1
            # Global scales and norms are often placed in a different shard
            # from expert matrices. Keep only these small verified tensors in RAM.
            with self.file.open('rb') as small_file:
                for item in self.small_specs[name]:
                    small_file.seek(item['start']);value=small_file.read(item['size'])
                    if len(value)!=item['size']:raise RuntimeError('Small verified tensor length mismatch')
                    self.small[(name,item['start'],item['size'])]=value
            print(json.dumps({'source_verified': name, 'seconds': time.perf_counter()-tick}), file=__import__('sys').stderr, flush=True)
        with self.file.open('rb') as f:
            f.seek(spec['start']); raw = f.read(spec['size'])
        if len(raw) != spec['size']: raise RuntimeError('Cached tensor length mismatch')
        return bytearray(raw)
