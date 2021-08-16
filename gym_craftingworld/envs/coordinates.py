"""
Coordinates
"""


class Coord():
    """
    Coordinate class
    """
    def __init__(self,
                 row: int,
                 col: int,
                 max_row: int = 100,
                 max_col: int = 100,
                 name: str = None):
        self.row = row
        self.col = col
        self.max_row = max_row
        self.max_col = max_col
        self.name = name

    def __add__(self, coord):
        new_row = max(0, min(self.row + coord.row, self.max_row))
        new_col = max(0, min(self.col + coord.col, self.max_col))
        return Coord(new_row, new_col, self.max_row, self.max_col)

    def __sub__(self, coord):
        new_row = max(0, min(self.row - coord.row, self.max_row))
        new_col = max(0, min(self.col - coord.col, self.max_col))
        return Coord(new_row, new_col, self.max_row, self.max_col)

    def __eq__(self, coord):
        if not isinstance(coord, Coord):
            return False
        return self.row == coord.row and self.col == coord.col

    def __str__(self):
        return str(self.tuple())

    def tuple(self):
        """Return a tuple representation."""
        return (self.row, self.col)
