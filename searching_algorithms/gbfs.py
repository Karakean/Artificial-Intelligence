import sys
from maze import Maze


def path_from(node):
    path = [node]
    while node.parent is not None:
        node = node.parent
        path.append(node)
    return path


def L1(A, B):
    return abs(A.x - B.x) + abs(A.y - B.y)


def gbfs(maze):
    start_node = maze.find_node('S')
    end_node = maze.find_node('E')
    q = [start_node]
    start_node.cost = 0
    while len(q) > 0:
        q.sort(key=lambda x: x.cost)
        node = q.pop(0)
        node.visited = True
        if node.type == 'E':
            return path_from(node)
        children = maze.get_possible_movements(node)
        for child in children:
            if not child.visited:
                child.cost = L1(child, end_node)
                child.parent = node
                q.append(child)
    return None


maze = Maze.from_file(sys.argv[1])
maze.draw()
maze.path = gbfs(maze)
print()
maze.draw()
print('path length: ', len(maze.path))
for node in maze.path:
    print(f'({node.x}, {node.y})', end=' ')
print()