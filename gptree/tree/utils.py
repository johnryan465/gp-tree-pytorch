

from dataclasses import dataclass
from typing import Generic, Iterable, List, Optional, TypeVar

Label = int

T = TypeVar("T")

@dataclass
class BinaryTree(Generic[T]):
    """
    Class which defines our model structure
    """
    left_labels: Iterable[Label]
    right_labels: Iterable[Label]
    left_node: Optional["BinaryTree[T]"]
    right_node: Optional["BinaryTree[T]"]
    data: T


    def node_list(self) -> List["BinaryTree[T]"]:
        left = self.left_node.node_list() if self.left_node is not None else []
        right = self.right_node.node_list() if self.right_node is not None else []

        return left + [self] + right