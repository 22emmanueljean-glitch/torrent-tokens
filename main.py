"""
MIT – private entry point for scheduler + kernel + ledger
Emmanuel Dessallien 2024
"""
import asyncio, json
from scheduler.ring_scheduler import RingScheduler
from credit.ledger import ledger

async def demo():
    sched = RingScheduler()
    # mock peers
    for i in range(5):
        from scheduler.ring_scheduler import Peer
        p = Peer(f"peer-{i}", None, sram_mb=8, upload_mbps=100, watts=5)
        await sched.on_peer_join(p)
    # mock job
    await sched.add_job(tile_id=0, model_id=1, act_blob=b"\x00" * 512)
    await asyncio.sleep(1)
    print("Credits:", sched.credits)

if __name__ == "__main__":
    asyncio.run(demo())