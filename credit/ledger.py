"""
MIT – off-chain credit ledger (private)
Emmanuel Dessallien 2024
"""
import json, time, pathlib

LEDGER_FILE = pathlib.Path("ledger.json")

class Ledger:
    def __init__(self):
        self.data = json.loads(LEDGER_FILE.read_text()) if LEDGER_FILE.exists() else {}

    def add(self, peer_id: str, credits: float):
        self.data[peer_id] = self.data.get(peer_id, 0.0) + credits
        self._save()

    def spend(self, peer_id: str, amount: float) -> bool:
        if self.data.get(peer_id, 0.0) >= amount:
            self.data[peer_id] -= amount
            self._save()
            return True
        return False

    def balance(self, peer_id: str) -> float:
        return self.data.get(peer_id, 0.0)

    def _save(self):
        LEDGER_FILE.write_text(json.dumps(self.data, indent=2))

# singleton
ledger = Ledger()