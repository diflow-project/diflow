from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diflow.operators.base import Operator


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def have_same_patches(op1: "Operator", op2: "Operator") -> bool:
    """
    Check if two operators have the same patches.

    Args:
        op1: First operator
        op2: Second operator

    Returns:
        bool: True if both operators have the same set of patches (by ID), False otherwise
    """
    patches1 = op1.get_patches()
    patches2 = op2.get_patches()

    # Get patch IDs from both lists
    patch_ids1 = {patch.id for patch in patches1}
    patch_ids2 = {patch.id for patch in patches2}

    return patch_ids1 == patch_ids2


class SchedulingPolicy(Enum):
    EXCLUSIVE = "exclusive"
    RANDOM = "random"
    DYNAMIC = "dynamic"
