"""
MIT – private scheduler for Torrent-Tokens
Emmanuel Dessallien 2024
"""
import asyncio, json, time, math, random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

@dataclass
class Peer:
    peer_id: str
    rtc_conn: object              # WebRTC conn (can be None for unit-test)
    sram_mb: int
    upload_mbps: float
    watts: float
    last_ping: float = field(default_factory=time.time)

@dataclass
class TileJob:
    tile_id: int
    model_id: int
    act_blob: bytes
    assigned: List[Tuple[str,int]] = field(default_factory=list)
    results: Dict[str, bytes] = field(default_factory=dict)
    committed: Optional[bytes] = None

class RingScheduler:
    def __init__(self, max_stall_ms=150, min_votes=2):
        self.max_stall = max_stall_ms / 1000.0
        self.min_votes = min_votes
        self.peers: Dict[str, Peer] = {}
        self.jobs = deque()
        self.credits: Dict[str, float] = {}

    # ---------- public API ----------
    async def add_job(self, tile_id: int, model_id: int, act_blob: bytes):
        job = TileJob(tile_id, model_id, act_blob)
        self.jobs.append(job)
        await self._schedule()

    async def on_result(self, peer_id: str, tile_id: int, result: bytes):
        job = self._find_job(tile_id)
        if not job: return
        job.results[peer_id] = result
        await self._try_commit(job)

    async def on_peer_join(self, peer: Peer):
        self.peers[peer.peer_id] = peer
        await self._schedule()

    # ---------- internals ----------
    async def _schedule(self):
        while self.jobs and len(self.peers) >= 3:
            job = self.jobs[0]
            if len(job.assigned) >= 3: break
            peers = self._pick_peers(3 - len(job.assigned))
            for i, p in enumerate(peers):
                job.assigned.append((p.peer_id, i))
                await self._send_tile(p, job)
            asyncio.create_task(self._stall_guard(job))

    def _pick_peers(self, n: int) -> List[Peer]:
        # credit-weighted random sample
        total = sum(self._credit(p.peer_id) for p in self.peers.values())
        if total == 0: return list(self.peers.values())[:n]
        picked = []
        for _ in range(n):
            r = random.random() * total
            cum = 0.0
            for p in self.peers.values():
                cum += self._credit(p.peer_id)
                if cum >= r:
                    picked.append(p)
                    break
        return picked

    def _credit(self, peer_id: str) -> float:
        p = self.peers.get(peer_id)
        if not p: return 0.0
        return p.sram_mb * p.upload_mbps / max(p.watts, 0.1)

    async def _send_tile(self, peer: Peer, job: TileJob):
        msg = {"type": "RUN_TILE", "tile_id": job.tile_id, "act_blob": job.act_blob.hex()}
        if peer.rtc_conn:
            peer.rtc_conn.send(json.dumps(msg))
        else:
            print(f"[stub] send to {peer.peer_id}: {msg}")

    async def _stall_guard(self, job: TileJob):
        await asyncio.sleep(self.max_stall)
        if job.committed is None:
            # reschedule to new peers
            self.jobs.appendleft(job)

    async def _try_commit(self, job: TileJob):
        checksums = list(job.results.values())
        if len(checksums) >= self.min_votes:
            from collections import Counter
            cnt = Counter(checksums)
            top = cnt.most_common(1)
            if top[0][1] >= self.min_votes:
                job.committed = top[0][0]
                self._reward(job)
                self.jobs.popleft()

    def _reward(self, job: TileJob):
        for peer_id, _ in job.assigned:
            if peer_id in job.results:
                self.credits[peer_id] = self.credits.get(peer_id, 0) + self._credit(peer_id)

    def _find_job(self, tile_id: int) -> Optional[TileJob]:
        for j in self.jobs:
            if j.tile_id == tile_id:
                return j
        return None