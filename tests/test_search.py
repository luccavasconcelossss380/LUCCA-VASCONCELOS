import unittest
from unittest.mock import patch

from algorithms.a_star import AStar
from algorithms.bfs import BFS
from algorithms.dfs import DEFAULT_DEPTH_LIMIT, DFS
from puzzle.result import SearchResult
from puzzle.state import GOAL_STATE, State


class SearchTests(unittest.TestCase):
    def assert_valid_solution(self, initial, result, expected_depth=None):
        self.assertIsInstance(result, SearchResult)
        self.assertTrue(result.found)
        self.assertTrue(result.solution.is_goal)
        self.assertEqual(result.solution.tiles, GOAL_STATE)
        path = result.solution.path()
        self.assertIs(path[0], initial)
        self.assertEqual(result.depth, len(path) - 1)
        self.assertEqual(result.path_cost, len(result.actions))
        self.assertEqual(result.path_cost, result.depth)
        if expected_depth is not None:
            self.assertEqual(result.depth, expected_depth)

        # Confere cada troca diretamente, sem usar State.neighbors como oráculo.
        offsets = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
        for parent, child, action in zip(path, path[1:], result.actions):
            self.assertIs(child.parent, parent)
            row, column = divmod(parent.tiles.index(0), 3)
            row_delta, column_delta = offsets[action]
            target_row, target_column = row + row_delta, column + column_delta
            self.assertTrue(0 <= target_row < 3 and 0 <= target_column < 3)
            target = target_row * 3 + target_column
            expected = list(parent.tiles)
            blank = row * 3 + column
            expected[blank], expected[target] = expected[target], expected[blank]
            self.assertEqual(child.tiles, tuple(expected))
            self.assertEqual(child.cost, parent.cost + 1)

    def test_already_solved(self):
        for algorithm in (BFS(), DFS(depth_limit=0), AStar()):
            with self.subTest(algorithm=type(algorithm).__name__):
                initial = State(GOAL_STATE)
                result = algorithm.search(initial)
                self.assert_valid_solution(initial, result, expected_depth=0)
                self.assertIs(result.solution, initial)
                self.assertEqual(result.actions, [])
                self.assertEqual(result.nodes_expanded, 0)
                self.assertEqual(result.nodes_generated, 1)
                self.assertEqual(result.max_frontier_size, 1)

    def test_bfs_and_astar_find_optimal_solutions(self):
        cases = (
            ((1, 2, 3, 4, 5, 6, 7, 0, 8), 1),
            ((1, 2, 3, 4, 5, 6, 0, 7, 8), 2),
            ((1, 2, 3, 5, 0, 6, 4, 7, 8), 4),
            ((7, 2, 4, 5, 0, 6, 8, 3, 1), 20),
            ((8, 6, 7, 2, 5, 4, 3, 0, 1), 31),
        )
        for algorithm in (BFS(), AStar()):
            for tiles, distance in cases:
                with self.subTest(algorithm=type(algorithm).__name__, distance=distance):
                    initial = State(tiles)
                    result = algorithm.search(initial)
                    self.assert_valid_solution(initial, result, expected_depth=distance)
                    self.assertGreater(result.nodes_expanded, 0)
                    self.assertGreaterEqual(result.nodes_generated, result.nodes_expanded)
                    self.assertGreater(result.max_frontier_size, 0)

    def test_unsolvable_boards_return_no_solution(self):
        for algorithm in (BFS(), DFS(), AStar()):
            with self.subTest(algorithm=type(algorithm).__name__):
                result = algorithm.search(State((1, 2, 3, 4, 5, 6, 8, 7, 0)))
                self.assertFalse(result.found)
                self.assertIsNone(result.solution)
                self.assertEqual(result.actions, [])
                self.assertEqual(result.path_cost, -1)
                self.assertEqual(result.nodes_expanded, 0)

    def test_dfs_respects_the_depth_limit_and_accepts_the_boundary(self):
        initial = State((1, 2, 3, 4, 5, 6, 0, 7, 8))
        for limit in (0, 1):
            with self.subTest(limit=limit):
                self.assertFalse(DFS(depth_limit=limit).search(initial).found)
        self.assert_valid_solution(initial, DFS(depth_limit=2).search(initial), expected_depth=2)
        with self.assertRaises(ValueError):
            DFS(depth_limit=-1)

    def test_dfs_default_limit_produces_a_legal_solution(self):
        initial = State((7, 2, 4, 5, 0, 6, 8, 3, 1))
        result = DFS().search(initial)
        self.assert_valid_solution(initial, result)
        self.assertLessEqual(result.depth, DEFAULT_DEPTH_LIMIT)

    def test_dfs_finds_solutions_when_states_are_reached_again(self):
        # Caminhos profundos não podem bloquear uma visita posterior mais curta.
        for tiles, distance in (
            ((7, 2, 4, 5, 0, 6, 8, 3, 1), 20),
            ((1, 2, 3, 5, 0, 6, 4, 7, 8), 4),
        ):
            with self.subTest(distance=distance):
                initial = State(tiles)
                result = DFS(depth_limit=distance).search(initial)
                self.assert_valid_solution(initial, result, expected_depth=distance)

    def test_statistics_count_expansions_and_all_generated_children(self):
        original_neighbors = State.neighbors
        for algorithm in (BFS(), DFS(depth_limit=4), AStar()):
            expansions = []

            def counted_neighbors(state):
                children = original_neighbors(state)
                expansions.append(len(children))
                return children

            with self.subTest(algorithm=type(algorithm).__name__):
                with patch.object(State, "neighbors", counted_neighbors):
                    result = algorithm.search(State((1, 2, 3, 5, 0, 6, 4, 7, 8)))
                self.assertTrue(result.found)
                self.assertEqual(result.nodes_expanded, len(expansions))
                self.assertEqual(result.nodes_generated, 1 + sum(expansions))

    def test_manhattan_ignores_blank_and_is_admissible(self):
        algorithm = AStar()
        for tiles, expected, optimum in (
            (GOAL_STATE, 0, 0),
            ((1, 2, 3, 4, 5, 6, 7, 0, 8), 1, 1),
            ((7, 2, 4, 5, 0, 6, 8, 3, 1), 14, 20),
            ((8, 6, 7, 2, 5, 4, 3, 0, 1), 21, 31),
        ):
            with self.subTest(tiles=tiles):
                state = State(tiles)
                self.assertEqual(algorithm.heuristic(state), expected)
                self.assertLessEqual(algorithm.heuristic(state), optimum)
                for child in state.neighbors():
                    self.assertLessEqual(algorithm.heuristic(state), 1 + algorithm.heuristic(child))

    def test_astar_accepts_a_custom_heuristic(self):
        initial = State((1, 2, 3, 5, 0, 6, 4, 7, 8))
        self.assert_valid_solution(initial, AStar(heuristic=lambda state: 0).search(initial), 4)

    def test_astar_with_an_admissible_inconsistent_heuristic(self):
        manhattan = AStar().heuristic

        def heuristic(state):
            return manhattan(state) if state.blank_index % 2 else 0

        initial = State((7, 2, 4, 5, 0, 6, 8, 3, 1))
        result = AStar(heuristic=heuristic).search(initial)
        self.assert_valid_solution(initial, result, expected_depth=20)

    def test_search_instances_can_be_reused(self):
        for algorithm in (BFS(), DFS(depth_limit=4), AStar()):
            for tiles in (GOAL_STATE, (1, 2, 3, 4, 5, 6, 0, 7, 8), GOAL_STATE):
                with self.subTest(algorithm=type(algorithm).__name__, tiles=tiles):
                    initial = State(tiles)
                    self.assert_valid_solution(initial, algorithm.search(initial))


if __name__ == "__main__":
    unittest.main()
