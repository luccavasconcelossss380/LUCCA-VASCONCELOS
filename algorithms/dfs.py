from puzzle.base_search import BaseSearch
from puzzle.state import State
from puzzle.result import SearchResult

DEFAULT_DEPTH_LIMIT = 50


class DFS(BaseSearch):

    def __init__(self, depth_limit: int = DEFAULT_DEPTH_LIMIT):
        if depth_limit < 0:
            raise ValueError("O limite de profundidade deve ser não negativo.")
        self.depth_limit = depth_limit

    def search(self, initial: State) -> SearchResult:
        """Busca em profundidade com pilha e limite de profundidade."""
        result = SearchResult(solution=None, nodes_generated=1, max_frontier_size=1)
        if not initial.is_solvable:
            return result

        frontier = [(initial, 0)]
        best_depth = {initial: 0}

        while frontier:
            current, depth = frontier.pop()
            if depth != best_depth[current]:
                continue
            if current.is_goal:
                result.solution = current
                result.depth = depth
                return result
            if depth >= self.depth_limit:
                continue

            result.nodes_expanded += 1
            children = current.neighbors()
            result.nodes_generated += len(children)
            # A pilha inverte a ordem; assim UP continua sendo explorado primeiro.
            for child in reversed(children):
                next_depth = depth + 1
                # Reabrir por um caminho mais curto preserva o limite disponível.
                if next_depth < best_depth.get(child, float("inf")):
                    best_depth[child] = next_depth
                    frontier.append((child, next_depth))

            result.max_frontier_size = max(result.max_frontier_size, len(frontier))

        return result
