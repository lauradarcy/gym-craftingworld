class coord(object):
    def __init__(self, row, col, max_row = 100, max_col = 100, name = None):
        self.row = row
        self.col = col
        self.max_row = max_row
        self.max_col = max_col
        self.name = name

    def __add__(self,c):
        new_row = max(0, min(self.row + c.row, self.max_row))
        new_col = max(0, min(self.col + c.col, self.max_col))
        return coord(new_row, new_col, self.max_row, self.max_col)

    def __sub__(self,c):
        new_row = max(0, min(self.row - c.row, self.max_row))
        new_col = max(0, min(self.col - c.col, self.max_col))
        return coord(new_row, new_col, self.max_row, self.max_col)

    def __eq__(self,c): #compares two coordsject type
        if type(c) != coord:
            return False
        return self.row == c.row and self.col == c.col

    def t(self): #return a tuple representation.
        return (self.row, self.col)

    def __str__(self):
        return str(self.t())

