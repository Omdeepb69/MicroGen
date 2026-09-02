"""Unit tests for Paged KV Cache block allocator and block table mapping."""

import pytest
from microgen.runtime.paged_kv import PhysicalBlock, BlockTable, PagedKVCacheAllocator


def test_physical_block_defaults():
    block = PhysicalBlock(block_id=0, block_size=16)
    assert block.block_id == 0
    assert block.block_size == 16
    assert block.ref_count == 0


def test_allocator_initialization():
    allocator = PagedKVCacheAllocator(num_blocks=10, block_size=16)
    assert allocator.get_num_free_blocks() == 10
    assert allocator.get_num_allocated_blocks() == 0


def test_single_block_allocation_and_free():
    allocator = PagedKVCacheAllocator(num_blocks=4, block_size=16)
    block = allocator.allocate_block()
    
    assert block.block_id == 0
    assert block.ref_count == 1
    assert allocator.get_num_free_blocks() == 3
    assert allocator.get_num_allocated_blocks() == 1

    allocator.free_block(block)
    assert allocator.get_num_free_blocks() == 4
    assert allocator.get_num_allocated_blocks() == 0


def test_sequence_block_allocation():
    allocator = PagedKVCacheAllocator(num_blocks=10, block_size=16)
    
    # 35 tokens require ceil(35/16) = 3 physical blocks
    table = allocator.allocate_sequence(sequence_id="seq-1", prompt_token_count=35)
    
    assert table.sequence_id == "seq-1"
    assert table.num_tokens == 35
    assert table.num_blocks() == 3
    assert allocator.get_num_free_blocks() == 7
    assert allocator.get_num_allocated_blocks() == 3

    allocator.free_sequence(table)
    assert allocator.get_num_free_blocks() == 10
    assert allocator.get_num_allocated_blocks() == 0
    assert table.num_blocks() == 0


def test_append_token_allocates_new_block_on_overflow():
    allocator = PagedKVCacheAllocator(num_blocks=5, block_size=16)
    
    # Exactly 16 tokens -> 1 full block
    table = allocator.allocate_sequence(sequence_id="seq-2", prompt_token_count=16)
    assert table.num_blocks() == 1
    assert table.is_last_block_full() is True

    # Append 1 token -> triggers allocation of 2nd block
    allocator.append_token(table)
    assert table.num_tokens == 17
    assert table.num_blocks() == 2
    assert table.is_last_block_full() is False

    # Append 15 more tokens -> fills 2nd block
    for _ in range(15):
        allocator.append_token(table)
    
    assert table.num_tokens == 32
    assert table.num_blocks() == 2
    assert table.is_last_block_full() is True


def test_out_of_memory_raises_memory_error():
    allocator = PagedKVCacheAllocator(num_blocks=2, block_size=16)
    
    # Allocate 32 tokens -> consumes all 2 blocks
    allocator.allocate_sequence(sequence_id="seq-3", prompt_token_count=32)
    assert allocator.get_num_free_blocks() == 0

    with pytest.raises(MemoryError):
        allocator.allocate_block()
