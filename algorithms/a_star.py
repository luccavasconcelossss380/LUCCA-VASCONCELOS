import heapq
from puzzle.base_search import BaseSearch
from puzzle.state import State
from puzzle.result import SearchResult


class AStar(BaseSearch):

    def __init__(self, heuristic=None):
        self._heuristic = heuristic

    def heuristic(self, state: State) -> int:
        """Distância Manhattan, sem contar o espaço vazio (heurística admissível)."""
        distance = 0
        for index, tile in enumerate(state.tiles):
            if tile != 0:
                row, column = divmod(index, 3)
                goal_row, goal_column = divmod(tile - 1, 3)
                distance += abs(row - goal_row) + abs(column - goal_column)
        return distance

    def search(self, initial: State) -> SearchResult:
        """Prioriza f(n) = g(n) + h(n), guardando o menor custo conhecido."""
        result = SearchResult(solution=None, nodes_generated=1, max_frontier_size=1)
        if not initial.is_solvable:
            return result

        heuristic = self._heuristic if self._heuristic is not None else self.heuristic
        frontier = [(initial.cost + heuristic(initial), initial)]
        best_cost = {initial: initial.cost}

        while frontier:
            _, current = heapq.heappop(frontier)
            # Uma entrada antiga pode continuar no heap após uma melhoria de custo.
            if current.cost != best_cost[current]:
                continue
            if current.is_goal:
                result.solution = current
                result.depth = len(current.path()) - 1
                return result

            result.nodes_expanded += 1
            for child in current.neighbors():
                result.nodes_generated += 1
                if child.cost < best_cost.get(child, float("inf")):
                    best_cost[child] = child.cost
                    priority = child.cost + heuristic(child)
                    heapq.heappush(frontier, (priority, child))

            result.max_frontier_size = max(result.max_frontier_size, len(frontier))

        return result
