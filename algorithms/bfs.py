from collections import deque
from puzzle.base_search import BaseSearch
from puzzle.state import State
from puzzle.result import SearchResult


class BFS(BaseSearch):

    def search(self, initial: State) -> SearchResult:
        """Busca em largura: expande os estados por ordem de descoberta."""
        result = SearchResult(solution=None, nodes_generated=1, max_frontier_size=1)
        if not initial.is_solvable:
            return result

        frontier = deque([initial])
        discovered = {initial}

        while frontier:
            current = frontier.popleft()
            if current.is_goal:
                result.solution = current
                result.depth = len(current.path()) - 1
                return result

            result.nodes_expanded += 1
            for child in current.neighbors():
                result.nodes_generated += 1
                if child not in discovered:
                    discovered.add(child)
                    frontier.append(child)

            result.max_frontier_size = max(result.max_frontier_size, len(frontier))

        return result
