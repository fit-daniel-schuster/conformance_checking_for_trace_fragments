from pm4py.objects.process_tree.obj import ProcessTree


def is_leaf_node(process_tree: ProcessTree) -> bool:
    return process_tree is not None and \
           len(process_tree.children) == 0 and \
           process_tree.operator is None


def get_pt_node_height(node: ProcessTree) -> int:
    """
    returns the node's height
    the root node has height 0, children of root have height 1, etc.
    :param node:
    :return:
    """
    if node.parent:
        return 1 + get_pt_node_height(node.parent)
    else:
        return 0


def is_tau_leaf(pt: ProcessTree):
    return (pt.children == [] or pt.children is None) and pt.operator is None and pt.label is None
