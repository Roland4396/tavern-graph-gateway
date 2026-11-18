#!/usr/bin/env python3
"""Seed public plugin assets from exact GitHub revisions, never over a GPU tunnel."""
import argparse
import asyncio
import json
import mimetypes
import subprocess
from pathlib import Path
from urllib.parse import quote

from gateway import DiskCache

ROOTS = {
    "was-node-suite-comfyui": "web",
    "rgthree-comfy": "web/comfyui",
    "comfyui-easy-use": "web_version/v2",
    "comfyui-kjnodes": "web",
    "comfyui_essentials": "js",
    "comfyui_ultimatesdupscale": "js",
}


async def warm(manifest, source_root, cache_root):
    cache = DiskCache(cache_root)
    files = size = 0
    # Validate all revisions before modifying the live cache.
    nodes = json.loads(manifest.read_text())["nodes"]
    for node in nodes:
        repo = source_root / node["name"]
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        assert revision == node["commit"], (node["name"], revision)
        assert not subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--stat"], text=True).strip(), node["name"]
    for node in nodes:
        name = node["name"]
        repo = source_root / name
        mounts = [("/extensions/" + name + "/", repo / ROOTS[name])]
        if name == "rgthree-comfy":
            mounts += [("/rgthree/" + d + "/", repo / "web" / d) for d in ("common", "lib")]
        if name == "comfyui-kjnodes":
            mounts.append(("/kjweb_async/", repo / "kjweb_async"))
        for prefix, root in mounts:
            for path in root.rglob("*"):
                url = prefix + quote(path.relative_to(root).as_posix())
                if not path.is_file() or not DiskCache.is_extension_asset(url):
                    continue
                data = path.read_bytes()
                content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
                await cache.put(url, data, {"Content-Type": content_type})
                files += 1
                size += len(data)
    return {"files": files, "bytes": size, "source": "revision-checked public GitHub checkouts"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(warm(args.manifest, args.source_root, args.cache_root))))
