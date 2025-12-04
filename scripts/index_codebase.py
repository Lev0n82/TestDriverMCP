#!/usr/bin/env python3
"""Index repository source files into Qdrant using sentence-transformers.

Usage:
  python scripts/index_codebase.py --collection code_index_384 --max-files 200

This script:
- Finds code files under the repo (extensions: .py, .js, .ts, .java, .md, .html)
- Chunks each file into text windows
- Optionally summarizes chunk via Ollama (disabled by default)
- Encodes chunks with `sentence-transformers` and upserts into Qdrant
"""

import argparse
import glob
import os
import uuid
import json
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
from sentence_transformers import SentenceTransformer


EXTENSIONS = ['.py', '.js', '.ts', '.java', '.md', '.html', '.json']


def find_files(root: str, max_files: int = 0) -> List[str]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip virtualenvs and .git
        if any(x in dirpath for x in ('.venv', '.git', 'node_modules', '__pycache__')):
            continue
        for fn in filenames:
            if any(fn.endswith(ext) for ext in EXTENSIONS):
                files.append(os.path.join(dirpath, fn))
                if max_files and len(files) >= max_files:
                    return files
    return files


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap if end < length else end
    return chunks


def make_collection(client: QdrantClient, collection_name: str, dim: int = 384):
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        print(f"Collection {collection_name} already exists")
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance="Cosine")
    )
    print(f"Created collection {collection_name} (dim={dim})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost', help='Qdrant host')
    parser.add_argument('--port', default=6333, type=int, help='Qdrant port')
    parser.add_argument('--collection', default='code_index_384')
    parser.add_argument('--model', default='all-MiniLM-L6-v2')
    parser.add_argument('--chunk-size', default=2000, type=int)
    parser.add_argument('--overlap', default=200, type=int)
    parser.add_argument('--max-files', default=100, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    args = parser.parse_args()

    repo_root = os.getcwd()
    print("Scanning files...")
    files = find_files(repo_root, max_files=args.max_files)
    print(f"Found {len(files)} files to index (max {args.max_files})")

    print(f"Loading embedding model '{args.model}'...")
    model = SentenceTransformer(args.model)

    client = QdrantClient(host=args.host, port=args.port)
    make_collection(client, args.collection, dim=model.get_sentence_embedding_dimension())

    points = []
    total = 0
    for path in files:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception:
            continue

        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        for idx, chunk in enumerate(chunks):
            preview = chunk[:200].replace('\n', ' ')
            vec = model.encode(preview).tolist()
            pid = f"{uuid.uuid4()}"
            payload = {
                'path': os.path.relpath(path, repo_root),
                'chunk_index': idx,
                'preview': preview,
                'language': os.path.splitext(path)[1].lstrip('.')
            }
            points.append({'id': pid, 'vector': vec, 'payload': payload})
            total += 1

            if len(points) >= args.batch_size:
                client.upsert(collection_name=args.collection, points=points)
                print(f"Upserted {len(points)} points (total {total})")
                points = []

    if points:
        client.upsert(collection_name=args.collection, points=points)
        print(f"Upserted {len(points)} points (total {total})")

    print(f"Indexing complete. Total points: {total}")


if __name__ == '__main__':
    main()
