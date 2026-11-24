import unittest

from puzzle.state import GOAL_STATE, State


class StateTests(unittest.TestCase):
    def test_invalid_boards_are_rejected(self):
        for tiles in ((1, 2, 3), (1,) * 9, (0, 1, 2, 3, 4, 5, 6, 7, 9)):
            with self.subTest(tiles=tiles), self.assertRaises(ValueError):
                State(tiles)

    def test_neighbors_for_every_blank_position(self):
        expected_actions = (
            ["DOWN", "RIGHT"],
            ["DOWN", "LEFT", "RIGHT"],
            ["DOWN", "LEFT"],
            ["UP", "DOWN", "RIGHT"],
            ["UP", "DOWN", "LEFT", "RIGHT"],
            ["UP", "DOWN", "LEFT"],
            ["UP", "RIGHT"],
            ["UP", "LEFT", "RIGHT"],
            ["UP", "LEFT"],
        )
        offsets = {"UP": -3, "DOWN": 3, "LEFT": -1, "RIGHT": 1}
        for blank, actions in enumerate(expected_actions):
            with self.subTest(blank=blank):
                tiles = list(GOAL_STATE)
                tiles[blank], tiles[8] = tiles[8], tiles[blank]
                state = State(tuple(tiles), cost=7)
                before = state.tiles
                children = state.neighbors()
                self.assertEqual([child.action for child in children], actions)
                self.assertEqual(state.tiles, before)
                for child in children:
                    self.assertIs(child.parent, state)
                    self.assertEqual(child.cost, 8)
                    target = blank + offsets[child.action]
                    expected = list(before)
                    expected[blank], expected[target] = expected[target], expected[blank]
                    self.assertEqual(child.tiles, tuple(expected))

    def test_path_and_actions_include_the_complete_solution(self):
        initial = State((1, 2, 3, 4, 5, 6, 0, 7, 8))
        middle = next(child for child in initial.neighbors() if child.action == "RIGHT")
        goal = next(child for child in middle.neighbors() if child.action == "RIGHT")
        self.assertEqual(initial.path(), [initial])
        self.assertEqual(initial.actions(), [])
        self.assertEqual(goal.path(), [initial, middle, goal])
        self.assertEqual(goal.actions(), ["RIGHT", "RIGHT"])
        self.assertTrue(goal.is_goal)
        self.assertEqual(goal.cost, 2)

    def test_long_paths_do_not_depend_on_recursion(self):
        state = State(GOAL_STATE)
        for step in range(1500):
            action = "LEFT" if step % 2 == 0 else "RIGHT"
            state = next(child for child in state.neighbors() if child.action == action)
        self.assertEqual(len(state.path()), 1501)
        self.assertEqual(len(state.actions()), 1500)

    def test_equality_and_hash_depend_only_on_tiles(self):
        first = State(GOAL_STATE)
        second = State(GOAL_STATE, parent=first, action="UP", cost=10)
        self.assertEqual(first, second)
        self.assertEqual(len({first, second}), 1)
        self.assertNotEqual(first, object())

    def test_solvability(self):
        self.assertTrue(State(GOAL_STATE).is_solvable)
        self.assertTrue(State((7, 2, 4, 5, 0, 6, 8, 3, 1)).is_solvable)
        self.assertFalse(State((1, 2, 3, 4, 5, 6, 8, 7, 0)).is_solvable)
        self.assertFalse(State((2, 8, 3, 1, 6, 4, 7, 0, 5)).is_solvable)


if __name__ == "__main__":
    unittest.main()
