

from dataclasses import dataclass
from typing import Iterable, List, Optional

Label = int

@dataclass
class BinaryTree:
    """
    Class which defines our model structure
    """
    id: int
    left_labels: Iterable[Label]
    right_labels: Iterable[Label]
    left_node: Optional["BinaryTree"]
    right_node: Optional["BinaryTree"]

def sorted_node_list(root: Optional[BinaryTree]) -> List[BinaryTree]:
    n_list = node_list(root)
    n_list.sort(key=lambda x: x.id)
    return n_list

def node_list(root: Optional[BinaryTree]) -> List[BinaryTree]:
    if root is None:
        return []
    return node_list(root.left_node) + [root] +  node_list(root.right_node)