

class Memory:
    """Fixed-size buffer to store (score, object) tuples"""

    def __init__(self,
                 memory_size: int = 1,
                 ):
        """Initialize a MemoryBuffer object"""
        self.memory_size = memory_size
        self.memory = []

    def add(self,
            score: float,
            obj,
            ) -> None:
        """Add a new agent to memory"""
        x = (score, obj)
        lo = 0
        hi = len(self.memory)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.memory[mid][0] < x[0]:
                lo = mid + 1
            else:
                hi = mid
        self.memory.insert(lo, x)
        self.memory = self.memory[:self.memory_size]

    def topk(self,
             k: int,
             ) -> list:
        """Return top K objects"""
        return [obj[1] for obj in self.memory[:k]]

    def bottomk(self,
                k: int,
                ) -> list:
        """Return bottom K objects"""
        return [obj[1] for obj in self.memory[-k:]]

    def set_size(self,
                 size: int,
                 ) -> None:
        self.memory_size = size
