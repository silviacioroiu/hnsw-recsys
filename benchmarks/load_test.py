"""Latency of GET /recommend/{user_id} under concurrent load.

Start the service first (see the Makefile target `serve` or compose.yaml).

Usage: python benchmarks/load_test.py [--url http://127.0.0.1:8000] --server "..."
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

from recsys.runinfo import collect_runinfo, set_seed

WARMUP = 200


async def run_load(url: str, users: list[int], k: int, concurrency: int) -> dict:
    """Send one request per user, `concurrency` at a time; time each one."""
    queue = asyncio.Queue()
    for user in users:
        queue.put_nowait(user)
    latencies_ms, errors = [], 0

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal errors
        while not queue.empty():
            user = queue.get_nowait()
            start = time.perf_counter()
            response = await client.get(f"{url}/recommend/{user}", params={"k": k})
            latencies_ms.append(1000 * (time.perf_counter() - start))
            errors += response.status_code != 200

    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=30) as client:
        start = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        seconds = time.perf_counter() - start
    return {
        "concurrency": concurrency, "requests": len(users), "errors": errors,
        "requests_per_second": round(len(users) / seconds, 1),
        **{f"latency_ms_p{p}": round(float(np.percentile(latencies_ms, p)), 2)
           for p in (50, 95, 99)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=Path,
                        default=Path("data/artifacts/serving/seen_users.npy"))
    parser.add_argument("--requests", type=int, default=3000)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--server", required=True,
                        help="how the service was started, recorded with the result")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    health = httpx.get(f"{args.url}/health", timeout=30).json()
    params = {"url": args.url, "server": args.server, "k": args.k,
              "requests_per_level": args.requests, "warmup_requests": WARMUP,
              "users": "known users sampled uniformly with replacement",
              "client": "httpx.AsyncClient on the same machine as the server",
              "health": health}
    runinfo = collect_runinfo(params, args.seed, packages=("httpx", "fastapi", "uvicorn"))

    all_users = np.load(args.users)
    rng = np.random.default_rng(args.seed)

    def sample(n: int) -> list[int]:
        return rng.choice(all_users, size=n).tolist()

    asyncio.run(run_load(args.url, sample(WARMUP), args.k, 4))
    rows = []
    for concurrency in args.concurrency:
        row = asyncio.run(run_load(args.url, sample(args.requests), args.k, concurrency))
        rows.append(row)
        print(f"concurrency={concurrency:<3} p50={row['latency_ms_p50']}ms  "
              f"p95={row['latency_ms_p95']}ms  p99={row['latency_ms_p99']}ms  "
              f"{row['requests_per_second']} req/s  errors={row['errors']}")

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "m6_latency.json"
    out_file.write_text(json.dumps({"runinfo": runinfo, "results": rows}, indent=2))
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
