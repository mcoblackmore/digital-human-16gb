"""Small ComfyUI HTTP client shared by qwen_edit.py, infinitetalk_run.py and check_setup.py.

COMFY_URL (default http://127.0.0.1:8188) may point at ComfyUI on another machine: inputs go up
through /upload/image and results come back through /view, so no shared folder is needed.
Standard library only.
"""
import json, os, time, uuid, mimetypes, urllib.request, urllib.parse, urllib.error

URL = os.environ.get('COMFY_URL', 'http://127.0.0.1:8188').rstrip('/')
CLIENT = str(uuid.uuid4())
SUB = 'digital_human'          # input/output subfolder, keeps this skill's files apart from others


def call(path, data=None, timeout=60):
    req = urllib.request.Request(URL + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', 'replace')[:2000]
        raise RuntimeError(f'{path} -> HTTP {e.code}: {body}') from None


def upload(path):
    """Copy a local image, video or audio file into ComfyUI's input/<SUB>/; returns the name loaders take."""
    name = os.path.basename(path)
    ctype = mimetypes.guess_type(name)[0] or 'application/octet-stream'
    b = uuid.uuid4().hex
    with open(path, 'rb') as f:
        data = f.read()
    body = b''.join([
        f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
        f'Content-Type: {ctype}\r\n\r\n'.encode() + data + b'\r\n',
        f'--{b}\r\nContent-Disposition: form-data; name="subfolder"\r\n\r\n{SUB}\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'.encode(),
        f'--{b}--\r\n'.encode()])
    req = urllib.request.Request(URL + '/upload/image', data=body,
                                 headers={'Content-Type': f'multipart/form-data; boundary={b}'})
    with urllib.request.urlopen(req, timeout=600) as r:
        res = json.loads(r.read())
    return f"{res['subfolder']}/{res['name']}" if res.get('subfolder') else res['name']


def models(folder):
    """File names ComfyUI lists for a model folder (subfolders included, e.g. 'WanVideo/x.safetensors')."""
    try:
        return call(f'/models/{folder}')
    except Exception:
        return []


def find_model(folder, filename):
    """The name ComfyUI uses for `filename`, wherever it sits inside `folder`; the bare name if not found."""
    for m in models(folder):
        if m.replace('\\', '/').split('/')[-1] == filename:
            return m
    return filename


def run(graph, poll=3):
    """Queue an API-format graph and wait for it; returns the history entry."""
    pid = call('/prompt', {'prompt': graph, 'client_id': CLIENT})['prompt_id']
    while True:
        time.sleep(poll)
        try:
            h = call(f'/history/{pid}')
        except Exception:          # the probe can time out while a big model loads; the job keeps running
            continue
        if pid in h:
            return h[pid]


def error(entry):
    errs = [m for m in entry.get('status', {}).get('messages', []) if m[0] == 'execution_error']
    return errs[0][1].get('exception_message', '')[:800] if errs else None


def download(entry, out_dir):
    """Save every output file of a finished job into out_dir; returns the local paths."""
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for o in entry.get('outputs', {}).values():
        for k in ('images', 'videos', 'gifs'):
            for it in o.get(k, []):
                if it.get('type', 'output') != 'output':
                    continue
                q = urllib.parse.urlencode({'filename': it['filename'], 'subfolder': it.get('subfolder', ''),
                                            'type': 'output'})
                dst = os.path.join(out_dir, it['filename'])
                with urllib.request.urlopen(f'{URL}/view?{q}', timeout=600) as r, open(dst, 'wb') as f:
                    f.write(r.read())
                files.append(dst)
    return files
