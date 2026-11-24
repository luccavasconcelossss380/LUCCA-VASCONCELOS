from __future__ import annotations
from typing import List, Optional, Tuple


GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)


class State:
    """Representa um estado do 8-puzzle como tupla imutável de 9 inteiros (0 = espaço vazio)."""

    def __init__(self, tiles: Tuple[int, ...], parent: Optional["State"] = None, action: Optional[str] = None, cost: int = 0):
        if len(tiles) != 9 or set(tiles) != set(range(9)):
            raise ValueError("Estado inválido: deve conter exatamente os valores 0-8.")
        self.tiles = tiles
        self.parent = parent
        self.action = action
        self.cost = cost

    @property
    def is_goal(self) -> bool:
        return self.tiles == GOAL_STATE

    @property
    def blank_index(self) -> int:
        return self.tiles.index(0)

    @property
    def is_solvable(self) -> bool:
        """Em um tabuleiro 3x3, o número de inversões deve ser par."""
        numbered_tiles = [tile for tile in self.tiles if tile != 0]
        inversions = 0
        for i in range(len(numbered_tiles)):
            for j in range(i + 1, len(numbered_tiles)):
                if numbered_tiles[i] > numbered_tiles[j]:
                    inversions += 1
        return inversions % 2 == 0

    def neighbors(self) -> List["State"]:
        """Gera movimentos do vazio na ordem UP, DOWN, LEFT, RIGHT."""
        blank = self.blank_index
        row, column = divmod(blank, 3)
        children = []
        moves = (("UP", -1, 0), ("DOWN", 1, 0), ("LEFT", 0, -1), ("RIGHT", 0, 1))

        for action, row_delta, column_delta in moves:
            next_row, next_column = row + row_delta, column + column_delta
            if 0 <= next_row < 3 and 0 <= next_column < 3:
                target = next_row * 3 + next_column
                tiles = list(self.tiles)
                tiles[blank], tiles[target] = tiles[target], tiles[blank]
                children.append(State(tuple(tiles), parent=self, action=action, cost=self.cost + 1))

        return children

    def path(self) -> List["State"]:
        """Retorna a sequência de estados do estado inicial até este."""
        states = []
        current = self
        while current is not None:
            states.append(current)
            current = current.parent
        states.reverse()
        return states

    def actions(self) -> List[str]:
        """Retorna a sequência de ações do estado inicial até este."""
        return [state.action for state in self.path()[1:] if state.action is not None]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, State) and self.tiles == other.tiles

    def __hash__(self) -> int:
        return hash(self.tiles)

    def __lt__(self, other: "State") -> bool:
        return self.cost < other.cost

    def __repr__(self) -> str:
        t = self.tiles
        return (
            f"+-------+\n"
            f"| {t[0]} {t[1]} {t[2]} |\n"
            f"| {t[3]} {t[4]} {t[5]} |\n"
            f"| {t[6]} {t[7]} {t[8]} |\n"
            f"+-------+"
        ).replace("0", " ")
