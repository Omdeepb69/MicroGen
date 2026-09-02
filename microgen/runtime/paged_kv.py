"""Block-based physical memory allocator and block table mapping for Paged KV Cache."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PhysicalBlock:
    """Represents a fixed-size physical memory block in KV cache."""
    block_id: int
    block_size: int = 16
    ref_count: int = 0


@dataclass
class BlockTable:
    """Logical sequence to physical block mapping table for a single sequence/request."""
    sequence_id: str
    block_size: int = 16
    physical_blocks: List[PhysicalBlock] = field(default_factory=list)
    num_tokens: int = 0

    def num_blocks(self) -> int:
        """Return total number of physical blocks currently allocated to sequence."""
        return len(self.physical_blocks)

    def is_last_block_full(self) -> bool:
        """Check whether the most recently allocated physical block is fully filled."""
        if not self.physical_blocks:
            return True
        return (self.num_tokens % self.block_size) == 0


class PagedKVCacheAllocator:
    """Manages pool of fixed-size physical memory blocks and handles allocation/deallocation."""

    def __init__(self, num_blocks: int = 64, block_size: int = 16) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free_blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i, block_size=block_size) for i in range(num_blocks)
        ]
        self._allocated_blocks: Dict[int, PhysicalBlock] = {}

    def get_num_free_blocks(self) -> int:
        """Return count of available unallocated physical blocks."""
        return len(self._free_blocks)

    def get_num_allocated_blocks(self) -> int:
        """Return count of active allocated physical blocks."""
        return len(self._allocated_blocks)

    def allocate_block(self) -> PhysicalBlock:
        """Allocate a single physical block from free pool."""
        if not self._free_blocks:
            raise MemoryError(f"PagedKVCacheAllocator out of physical memory blocks (max={self.num_blocks}).")
        
        block = self._free_blocks.pop(0)
        block.ref_count = 1
        self._allocated_blocks[block.block_id] = block
        return block

    def free_block(self, block: PhysicalBlock) -> None:
        """Free a physical block or decrement its reference count."""
        if block.block_id not in self._allocated_blocks:
            return
        
        block.ref_count -= 1
        if block.ref_count <= 0:
            block.ref_count = 0
            del self._allocated_blocks[block.block_id]
            self._free_blocks.append(block)

    def allocate_sequence(self, sequence_id: str, prompt_token_count: int) -> BlockTable:
        """Allocate physical blocks for a new sequence given initial prompt token count."""
        num_blocks_needed = (prompt_token_count + self.block_size - 1) // self.block_size
        if prompt_token_count == 0:
            num_blocks_needed = 0

        if self.get_num_free_blocks() < num_blocks_needed:
            raise MemoryError(
                f"Insufficient free physical blocks: needed {num_blocks_needed}, available {self.get_num_free_blocks()}."
            )

        block_table = BlockTable(
            sequence_id=sequence_id,
            block_size=self.block_size,
            num_tokens=prompt_token_count,
        )

        for _ in range(num_blocks_needed):
            b = self.allocate_block()
            block_table.physical_blocks.append(b)

        return block_table

    def append_token(self, block_table: BlockTable) -> None:
        """Append 1 token to sequence, allocating a new physical block if the last block is full."""
        if block_table.is_last_block_full():
            new_block = self.allocate_block()
            block_table.physical_blocks.append(new_block)
        
        block_table.num_tokens += 1

    def free_sequence(self, block_table: BlockTable) -> None:
        """Free all physical blocks associated with a sequence."""
        for block in block_table.physical_blocks:
            self.free_block(block)
        block_table.physical_blocks.clear()
        block_table.num_tokens = 0
