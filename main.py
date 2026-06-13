from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading, time, os, math, random, heapq
import numpy as np
import requests
from typing import List, Dict, Any
from dataclasses import dataclass, field

APP_PORT = 8080

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VecIn(BaseModel):
    metadata: str
    category: str
    embedding: List[float]


class DocIn(BaseModel):
    title: str
    text: str


def parse_vec_param(v: str) -> List[float]:
    try:
        return [float(x) for x in v.split(',') if x.strip()!='']
    except Exception:
        return []


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def get_dist_fn(metric: str):
    if metric == 'cosine':
        return cosine_distance
    if metric == 'manhattan':
        return manhattan_distance
    return euclidean_distance


@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float]


class BruteForce:
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, item: VectorItem):
        self.items.append(item)

    def knn(self, q: List[float], k: int, dist_fn):
        scored = [(dist_fn(np.array(q, dtype=np.float32), np.array(v.emb, dtype=np.float32)), v.id) for v in self.items]
        scored.sort(key=lambda x: x[0])
        return scored[:k]

    def remove(self, vid: int):
        self.items = [v for v in self.items if v.id != vid]


class KDNode:
    def __init__(self, item: VectorItem):
        self.item = item
        self.left = None
        self.right = None


class KDTree:
    def __init__(self, dims: int):
        self.root = None
        self.dims = dims

    def _destroy(self, node):
        if not node:
            return
        self._destroy(node.left)
        self._destroy(node.right)

    def _insert(self, node, item: VectorItem, depth: int):
        if node is None:
            return KDNode(item)
        axis = depth % self.dims
        if item.emb[axis] < node.item.emb[axis]:
            node.left = self._insert(node.left, item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def insert(self, item: VectorItem):
        self.root = self._insert(self.root, item, 0)

    def rebuild(self, items: List[VectorItem]):
        self.root = None
        for item in items:
            self.insert(item)

    def _knn(self, node, q: List[float], k: int, depth: int, dist_fn, heap):
        if node is None:
            return
        qv = np.array(q, dtype=np.float32)
        dv = np.array(node.item.emb, dtype=np.float32)
        dn = dist_fn(qv, dv)
        if len(heap) < k:
            heapq.heappush(heap, (-dn, node.item.id))
        elif dn < -heap[0][0]:
            heapq.heapreplace(heap, (-dn, node.item.id))
        axis = depth % self.dims
        diff = q[axis] - node.item.emb[axis]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left
        self._knn(closer, q, k, depth + 1, dist_fn, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist_fn, heap)

    def knn(self, q: List[float], k: int, dist_fn):
        heap = []
        self._knn(self.root, q, k, 0, dist_fn, heap)
        out = [(-d, vid) for d, vid in heap]
        out.sort(key=lambda x: x[0])
        return out[:k]


class HNSW:
    class Node:
        def __init__(self, item: VectorItem, max_layer: int):
            self.item = item
            self.max_layer = max_layer
            self.nbrs: List[List[int]] = [[] for _ in range(max_layer + 1)]

    @dataclass
    class GraphInfo:
        topLayer: int
        nodeCount: int
        nodesPerLayer: List[int] = field(default_factory=list)
        edgesPerLayer: List[int] = field(default_factory=list)
        nodes: List[Dict[str, Any]] = field(default_factory=list)
        edges: List[Dict[str, int]] = field(default_factory=list)

    def __init__(self, m: int = 16, ef_build: int = 200, seed: int = 42):
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m)) if m > 1 else 1.0
        self.top_layer = -1
        self.entry_pt = -1
        self.rng = random.Random(seed)
        self.graph: Dict[int, HNSW.Node] = {}

    def _rand_level(self) -> int:
        u = max(self.rng.random(), 1e-12)
        return int(math.floor(-math.log(u) * self.mL))

    def _distance(self, q: List[float], item: VectorItem, dist_fn):
        return dist_fn(np.array(q, dtype=np.float32), np.array(item.emb, dtype=np.float32))

    def _search_layer(self, q: List[float], ep: int, ef: int, lyr: int, dist_fn):
        if ep not in self.graph:
            return []
        visited = {ep}
        candidates = []
        found = []
        d0 = self._distance(q, self.graph[ep].item, dist_fn)
        heapq.heappush(candidates, (d0, ep))
        heapq.heappush(found, (-d0, ep))
        while candidates:
            cd, cid = heapq.heappop(candidates)
            if len(found) >= ef and cd > -found[0][0]:
                break
            if lyr >= len(self.graph[cid].nbrs):
                continue
            for nid in self.graph[cid].nbrs[lyr]:
                if nid in visited or nid not in self.graph:
                    continue
                visited.add(nid)
                nd = self._distance(q, self.graph[nid].item, dist_fn)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(candidates, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)
        res = [(-d, vid) for d, vid in found]
        res.sort(key=lambda x: x[0])
        return res

    def _select_nbrs(self, cands, max_m: int):
        return [vid for _, vid in cands[:max_m]]

    def insert(self, item: VectorItem, dist_fn):
        lvl = self._rand_level()
        self.graph[item.id] = HNSW.Node(item, lvl)
        if self.entry_pt == -1:
            self.entry_pt = item.id
            self.top_layer = lvl
            return
        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if ep in self.graph and lc < len(self.graph[ep].nbrs):
                w = self._search_layer(item.emb, ep, 1, lc, dist_fn)
                if w:
                    ep = w[0][1]
        for lc in range(min(self.top_layer, lvl), -1, -1):
            w = self._search_layer(item.emb, ep, self.ef_build, lc, dist_fn)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(w, max_m)
            self.graph[item.id].nbrs[lc] = list(sel)
            for nid in sel:
                if nid not in self.graph:
                    continue
                if len(self.graph[nid].nbrs) <= lc:
                    self.graph[nid].nbrs.extend([[] for _ in range(lc + 1 - len(self.graph[nid].nbrs))])
                conn = self.graph[nid].nbrs[lc]
                conn.append(item.id)
                if len(conn) > max_m:
                    ds = []
                    for c in conn:
                        if c in self.graph:
                            ds.append((dist_fn(np.array(self.graph[nid].item.emb, dtype=np.float32), np.array(self.graph[c].item.emb, dtype=np.float32)), c))
                    ds.sort(key=lambda x: x[0])
                    self.graph[nid].nbrs[lc] = [c for _, c in ds[:max_m]]
            if w:
                ep = w[0][1]
        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = item.id

    def knn(self, q: List[float], k: int, ef: int, dist_fn):
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if ep in self.graph and lc < len(self.graph[ep].nbrs):
                w = self._search_layer(q, ep, 1, lc, dist_fn)
                if w:
                    ep = w[0][1]
        w = self._search_layer(q, ep, max(ef, k), 0, dist_fn)
        return w[:k]

    def remove(self, vid: int):
        if vid not in self.graph:
            return
        for nd in self.graph.values():
            for layer in nd.nbrs:
                while vid in layer:
                    layer.remove(vid)
        if self.entry_pt == vid:
            self.entry_pt = -1
            for nid in self.graph:
                if nid != vid:
                    self.entry_pt = nid
                    break
        del self.graph[vid]

    def info(self):
        max_l = max(self.top_layer + 1, 1)
        gi = HNSW.GraphInfo(topLayer=self.top_layer, nodeCount=len(self.graph))
        gi.nodesPerLayer = [0] * max_l
        gi.edgesPerLayer = [0] * max_l
        for vid, nd in self.graph.items():
            gi.nodes.append({'id': vid, 'metadata': nd.item.metadata, 'category': nd.item.category, 'maxLyr': nd.max_layer})
            for lc in range(0, min(nd.max_layer, max_l - 1) + 1):
                gi.nodesPerLayer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if vid < nid:
                            gi.edgesPerLayer[lc] += 1
                            gi.edges.append({'src': vid, 'dst': nid, 'lyr': lc})
        return gi


class VectorDB:
    def __init__(self, dims=16):
        self.lock = threading.Lock()
        self.next_id = 1
        self.store: Dict[int, VectorItem] = {}
        self.dims = dims
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)

    def insert(self, metadata: str, category: str, emb: List[float], dist_metric: str = 'cosine') -> int:
        with self.lock:
            vid = self.next_id
            self.next_id += 1
            item = VectorItem(vid, metadata, category, list(emb))
            self.store[vid] = item
            self.bf.insert(item)
            self.kdt.insert(item)
            self.hnsw.insert(item, get_dist_fn(dist_metric))
            return vid

    def remove(self, vid: int) -> bool:
        with self.lock:
            if vid not in self.store:
                return False
            del self.store[vid]
            self.bf.remove(vid)
            self.hnsw.remove(vid)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def all(self):
        with self.lock:
            return [
                {'id': v.id, 'metadata': v.metadata, 'category': v.category, 'embedding': v.emb}
                for v in self.store.values()
            ]

    def _algo_knn(self, q: List[float], k: int, metric: str, algo: str):
        dist_fn = get_dist_fn(metric)
        start = time.perf_counter()
        if algo == 'bruteforce':
            raw = self.bf.knn(q, k, dist_fn)
        elif algo == 'kdtree':
            raw = self.kdt.knn(q, k, dist_fn)
        else:
            raw = self.hnsw.knn(q, k, 50, dist_fn)
        us = int((time.perf_counter() - start) * 1_000_000)
        return raw, us

    def knn(self, q: List[float], k: int, metric: str, algo: str = 'hnsw'):
        with self.lock:
            if not self.store:
                return {'results': [], 'latencyUs': 0, 'algo': algo, 'metric': metric}
            raw, us = self._algo_knn(q, k, metric, algo)
            res = []
            for d, vid in raw[:k]:
                v = self.store.get(vid)
                if v is not None:
                    res.append({'id': v.id, 'metadata': v.metadata, 'category': v.category, 'distance': float(d), 'embedding': v.emb})
            return {'results': res, 'latencyUs': us, 'algo': algo, 'metric': metric}

    def benchmark(self, q: List[float], k: int, metric: str):
        with self.lock:
            if not self.store:
                return {'bruteforceUs': 0, 'kdtreeUs': 0, 'hnswUs': 0, 'itemCount': 0}
            bf_raw, bf_us = self._algo_knn(q, k, metric, 'bruteforce')
            kd_raw, kd_us = self._algo_knn(q, k, metric, 'kdtree')
            hs_raw, hs_us = self._algo_knn(q, k, metric, 'hnsw')
            return {'bruteforceUs': bf_us, 'kdtreeUs': kd_us, 'hnswUs': hs_us, 'itemCount': len(self.store)}

    def hnsw_info(self):
        with self.lock:
            return self.hnsw.info()

    def size(self):
        with self.lock:
            return len(self.store)


class OllamaClient:
    def __init__(self, host='127.0.0.1', port=11434):
        self.base = f'http://{host}:{port}'
        self.embed_model = 'nomic-embed-text'
        self.gen_model = 'llama3.2'

    def is_available(self) -> bool:
        try:
            r = requests.get(self.base + '/api/tags', timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> List[float]:
        try:
            body = {"model": self.embed_model, "prompt": text}
            r = requests.post(self.base + '/api/embeddings', json=body, timeout=30)
            if r.status_code != 200:
                return []
            data = r.json()
            # Expecting {"embedding":[...]} or similar
            if isinstance(data, dict) and 'embedding' in data:
                return data['embedding']
            # Some Ollama installs wrap differently — try to find key
            for v in data.values():
                if isinstance(v, dict) and 'embedding' in v:
                    return v['embedding']
            return []
        except Exception:
            return []

    def generate(self, prompt: str) -> str:
        try:
            body = {"model": self.gen_model, "prompt": prompt, "stream": False}
            r = requests.post(self.base + '/api/generate', json=body, timeout=180)
            if r.status_code != 200:
                return 'ERROR: Ollama unavailable. Run: ollama serve'
            data = r.json()
            if isinstance(data, dict) and 'response' in data:
                return data['response']
            return str(data)
        except Exception:
            return 'ERROR: Ollama unavailable. Run: ollama serve'


ollama = OllamaClient()


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30):
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    step = chunk_words - overlap_words
    for i in range(0, len(words), step):
        end = min(i + chunk_words, len(words))
        chunk = ' '.join(words[i:end])
        chunks.append(chunk)
        if end == len(words):
            break
    return chunks


class DocumentDB:
    def __init__(self):
        self.lock = threading.Lock()
        self.next_id = 1
        self.store: Dict[int, Dict[str, Any]] = {}
        self.dims = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)
            did = self.next_id
            self.next_id += 1
            self.store[did] = {'id': did, 'title': title, 'text': text, 'embedding': np.array(emb, dtype=np.float32)}
            return did

    def remove(self, did: int) -> bool:
        with self.lock:
            if did in self.store:
                del self.store[did]
                return True
            return False

    def all(self):
        with self.lock:
            out = []
            for v in self.store.values():
                preview = v['text'][:120] + ('…' if len(v['text'])>120 else '')
                words = len(v['text'].split())
                out.append({'id': v['id'], 'title': v['title'], 'preview': preview, 'words': words})
            return out

    def search(self, q_emb: List[float], k: int, max_dist: float = 0.7):
        if not self.store:
            return []
        qv = np.array(q_emb, dtype=np.float32)
        with self.lock:
            items = list(self.store.values())
        dists = []
        for v in items:
            d = cosine_distance(qv, v['embedding'])
            dists.append((d, v))
        dists.sort(key=lambda x: x[0])
        out = []
        for d, v in dists[:k]:
            if d <= max_dist:
                out.append((float(d), {'id': v['id'], 'title': v['title'], 'text': v['text']}))
        return out

    def size(self):
        with self.lock:
            return len(self.store)


DATA_DIR = os.path.dirname(__file__)

vec_db = VectorDB(dims=16)
doc_db = DocumentDB()

# Load demo vectors roughly ported from the original C++ demo
def load_demo():
    # We'll keep this small set to mimic the demo behavior
    demo = [
        ("Linked List: nodes connected by pointers","cs",[0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
        ("Sushi: vinegared rice raw fish and nori rolls","food",[0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
        ("Basketball: fast-paced shooting dribbling slam dunks","sports",[0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
    ]
    for meta, cat, emb in demo:
        vec_db.insert(meta, cat, emb)


load_demo()


@app.get('/search')
def search(v: str, k: int = 5, metric: str = 'cosine', algo: str = 'hnsw'):
    q = parse_vec_param(v)
    if len(q) != vec_db.dims:
        return JSONResponse({'error': f'need {vec_db.dims}D vector'}, status_code=400)
    return vec_db.knn(q, k, metric, algo)


@app.post('/insert')
def insert(vec: VecIn):
    if len(vec.embedding) != vec_db.dims:
        return JSONResponse({'error': 'invalid body'}, status_code=400)
    vid = vec_db.insert(vec.metadata, vec.category, vec.embedding)
    return {'id': vid}


@app.delete('/delete/{vid}')
def delete(vid: int):
    ok = vec_db.remove(vid)
    return {'ok': ok}


@app.get('/items')
def items():
    return vec_db.all()


@app.get('/benchmark')
def benchmark(v: str, k: int = 5, metric: str = 'cosine'):
    q = parse_vec_param(v)
    if len(q) != vec_db.dims:
        return JSONResponse({'error': f'need {vec_db.dims}D vector'}, status_code=400)
    return vec_db.benchmark(q, k, metric)


@app.get('/hnsw-info')
def hnsw_info():
    gi = vec_db.hnsw_info()
    return {
        'topLayer': gi.topLayer,
        'nodeCount': gi.nodeCount,
        'nodesPerLayer': gi.nodesPerLayer,
        'edgesPerLayer': gi.edgesPerLayer,
        'nodes': gi.nodes,
        'edges': gi.edges,
    }


@app.post('/doc/insert')
def doc_insert(d: DocIn):
    title = d.title.strip(); text = d.text.strip()
    if not title or not text:
        return JSONResponse({'error': 'need title and text'}, status_code=400)
    chunks = chunk_text(text, 250, 30)
    ids = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return JSONResponse({'error':'Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2'}, status_code=500)
        chunk_title = f"{title} [{i+1}/{len(chunks)}]" if len(chunks)>1 else title
        ids.append(doc_db.insert(chunk_title, chunk, emb))
    # Also insert a 16D fake vector for visualization (approximate)
    # Use a short embedding heuristic to map to 16D
    emb16 = ollama.embed(title + ' ' + text)[:16]
    if emb16 and len(emb16) >= 16:
        vec_db.insert(title, 'doc', emb16[:16])
    return {'ids': ids, 'chunks': len(chunks), 'dims': doc_db.dims}


@app.delete('/doc/delete/{did}')
def doc_delete(did: int):
    ok = doc_db.remove(did)
    return {'ok': ok}


@app.get('/doc/list')
def doc_list():
    return doc_db.all()


@app.post('/doc/search')
def doc_search(body: Dict[str, Any]):
    question = body.get('question', '')
    k = int(body.get('k', 3))
    if not question:
        return JSONResponse({'error': 'need question'}, status_code=400)
    qemb = ollama.embed(question)
    if not qemb:
        return JSONResponse({'error': 'Ollama unavailable'}, status_code=500)
    hits = doc_db.search(qemb, k)
    contexts = [{'id': h[1]['id'], 'title': h[1]['title'], 'distance': h[0]} for h in hits]
    return {'contexts': contexts}


@app.post('/doc/ask')
def doc_ask(body: Dict[str, Any]):
    question = body.get('question', '')
    k = int(body.get('k', 3))
    if not question:
        return JSONResponse({'error': 'need question'}, status_code=400)
    qemb = ollama.embed(question)
    if not qemb:
        return JSONResponse({'error': 'Ollama unavailable'}, status_code=500)
    hits = doc_db.search(qemb, k)
    ctxs = ''
    contexts = []
    for i, (d, item) in enumerate(hits):
        ctxs += f"[{i+1}] {item['title']}:\n{item['text']}\n\n"
        contexts.append({'id': item['id'], 'title': item['title'], 'text': item['text'], 'distance': d})
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
        "Just answer the question naturally.\n\n"
        "Context:\n" + ctxs + "Question: " + question + "\n\nAnswer:"
    )
    answer = ollama.generate(prompt)
    return {'answer': answer, 'model': ollama.gen_model, 'contexts': contexts, 'docCount': doc_db.size()}


@app.get('/status')
def status():
    up = ollama.is_available()
    return {'ollamaAvailable': up, 'embedModel': ollama.embed_model, 'genModel': ollama.gen_model, 'docCount': doc_db.size(), 'docDims': doc_db.dims, 'demoDims': vec_db.dims, 'demoCount': vec_db.size()}


@app.get('/stats')
def stats():
    return {'count': vec_db.size(), 'dims': vec_db.dims, 'algorithms': ['bruteforce','kdtree','hnsw'], 'metrics': ['euclidean','cosine','manhattan']}


@app.get('/')
def root():
    index_path = os.path.join(DATA_DIR, 'index.html')
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404)
    return FileResponse(index_path, media_type='text/html')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=APP_PORT, reload=False)
