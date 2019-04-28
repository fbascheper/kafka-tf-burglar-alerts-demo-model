

def fix_nodes(frozen_graph):
    """
    Fix nodes, copied from https://github.com/tensorflow/tensorflow/issues/3628
    :param frozen_graph:
    :return:
    """
    for node in frozen_graph.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index] and "Switch" not in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

# If that doesn't work, try suggestions from https://github.com/keras-team/keras/issues/12547
