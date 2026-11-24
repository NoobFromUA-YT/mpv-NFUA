#!/usr/bin/env python3
"""Download MSYS2 CLANG64 packages (with dependencies) and unpack them into sysroot/.
usage: msys2_fetch.py pkg [pkg...]   (names without the mingw-w64-clang-x86_64- prefix)"""
import io, os, re, subprocess, sys, tarfile, urllib.request
BASE = 'https://mirror.msys2.org/mingw/clang64/'
PRE = 'mingw-w64-clang-x86_64-'
here = os.path.dirname(os.path.abspath(__file__))
dl, root = os.path.join(here, 'dl', 'msys2'), os.path.join(here, 'sysroot')
os.makedirs(dl, exist_ok=True); os.makedirs(root, exist_ok=True)
def get(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path + '.part'); os.rename(path + '.part', path)
    return path
db = get(BASE + 'clang64.db', os.path.join(dl, 'clang64.db'))
raw = subprocess.run(['bsdtar', '-xOf', db, '*/desc'], capture_output=True).stdout.decode() if False else None
# the db is a (zstd or gzip) tar of <pkg>/desc files
data = open(db, 'rb').read()
if data[:4] == b'\x28\xb5\x2f\xfd':
    data = subprocess.run(['zstd', '-dc', db], capture_output=True).stdout
    tf = tarfile.open(fileobj=io.BytesIO(data))
else:
    tf = tarfile.open(db)
pkgs, provides = {}, {}
for m in tf.getmembers():
    if not m.name.endswith('/desc'): continue
    txt = tf.extractfile(m).read().decode()
    f = {k: v.strip().split('\n') for k, v in re.findall(r'%(\w+)%\n(.*?)(?:\n\n|\Z)', txt, re.S)}
    name = f['NAME'][0]
    pkgs[name] = f
    for p in f.get('PROVIDES', []): provides.setdefault(re.split(r'[<>=]', p)[0], name)
todo, done = [PRE + a for a in sys.argv[1:]], []
while todo:
    n = todo.pop()
    n = n if n in pkgs else provides.get(n)
    if not n or n in done: continue
    done.append(n)
    for d in pkgs[n].get('DEPENDS', []): todo.append(re.split(r'[<>=]', d)[0])
print(len(done), 'packages')
for n in sorted(done):
    fn = pkgs[n]['FILENAME'][0]
    path = get(BASE + fn, os.path.join(dl, fn))
    subprocess.run(f'zstd -dc "{path}" | tar -x -C "{root}" --exclude=.BUILDINFO --exclude=.MTREE --exclude=.PKGINFO --exclude=.INSTALL', shell=True, check=True)
    print(' ', n[len(PRE):], pkgs[n]['VERSION'][0])
