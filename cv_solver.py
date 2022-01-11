import copy

OPPOSITES = {
    "a1": "a2",
    "a2": "a1",
    "b1": "b2",
    "b2": "b1",
    "c1": "c2",
    "c2": "c1",
    "d1": "d2",
    "d2": "d1"
}


class Piece:

    def __init__(self, id_, north, east, south, west):
        if not north or not east or not south or not west:
            raise ValueError("Piece must have all sides givin")
        self.north_value = north
        self.east_value = east
        self.south_value = south
        self.west_value = west

        self.id_ = id_


class Slot:

    def __init__(self,
                 piece=None,
                 north=None,
                 east=None,
                 south=None,
                 west=None) -> None:
        self.north = north
        self.east = east
        self.south = south
        self.west = west
        self.piece = piece

    def set_piece(self, piece):
        self.piece = piece

    def sides_are_good(self):
        if not self.piece:
            return
        if self.north:
            if self.north.piece:
                if OPPOSITES[self.north.piece.south_value] \
                        != self.piece.north_value:
                    return

        if self.east:
            if self.east.piece:
                if OPPOSITES[self.east.piece.west_value] \
                        != self.piece.east_value:
                    return

        if self.south:
            if self.south.piece:
                if OPPOSITES[self.south.piece.north_value] \
                        != self.piece.south_value:
                    return

        if self.west:
            if self.west.piece:
                if OPPOSITES[self.west.piece.east_value] \
                        != self.piece.west_value:
                    return

        return True


class Board:

    def __init__(self):

        self.nw = Slot()
        self.n = Slot()
        self.ne = Slot()
        self.w = Slot()
        self.c = Slot()
        self.e = Slot()
        self.sw = Slot()
        self.s = Slot()
        self.se = Slot()

        slots = [
            self.nw,
            self.n,
            self.ne,
            self.w,
            self.c,
            self.e,
            self.sw,
            self.s,
            self.se,
        ]
        self.slots = slots

        self.nw = slots[0]
        self.n = slots[1]
        self.e = slots[2]
        self.w = slots[3]
        self.c = slots[4]
        self.e = slots[5]
        self.sw = slots[6]
        self.s = slots[7]
        self.se = slots[8]

    def set_slots(self):
        nw = self.nw
        n = self.n
        ne = self.ne
        w = self.w
        c = self.c
        e = self.e
        sw = self.sw
        s = self.s
        se = self.se

        self.nw.east = n
        self.nw.south = w

        self.n.east = ne
        self.n.south = c
        self.n.west = nw

        self.ne.south = e
        self.ne.west = n

        self.w.north = nw
        self.w.east = c
        self.w.south = sw

        self.c.north = n
        self.c.east = e
        self.c.south = s
        self.c.west = w

        self.e.north = ne
        self.e.south = se
        self.e.west = c

        self.sw.north = w
        self.sw.east = s

        self.s.north = c
        self.s.east = se
        self.s.west = sw

        self.se.north = e
        self.se.west = s

        slots = [nw, n, ne, w, c, e, sw, s, se]
        self.slots = slots

        self.nw = slots[0]
        self.n = slots[1]
        self.e = slots[2]
        self.w = slots[3]
        self.c = slots[4]
        self.e = slots[5]
        self.sw = slots[6]
        self.s = slots[7]
        self.se = slots[8]

    def check(self, slot):
        if self.slots[slot].sides_are_good():
            return True

    def get_rotations(self, piece):
        rots = []
        rots.append([
            piece.north_value, piece.east_value, piece.south_value,
            piece.west_value
        ])
        rots.append(rots[0][1:])
        rots[1].append(rots[0][0])
        rots.append(rots[1][1:])
        rots[2].append(rots[1][0])
        rots.append(rots[2][1:])
        rots[3].append(rots[2][0])

        return rots


def solve(board, available, slot=0):

    if slot == 9:
        print("Solution Found!")
        answer = [(
            slot.piece.id_,
            slot.piece.north_value,
            slot.piece.east_value,
            slot.piece.south_value,
            slot.piece.west_value,
        ) for slot in board.slots]
        for x in range(0, 9, 3):
            print(answer[x:x + 3])
        return True
    tried = []
    for piece in available[:]:
        current = piece
        available.remove(current)
        for rotation in board.get_rotations(piece):
            global iteration
            iteration += 1
            print("Piece set. Iteration:", iteration)
            rotation = Piece(current.id_, *rotation)
            temp = copy.deepcopy(board)
            temp.slots[slot].set_piece(rotation)
            temp.set_slots()
            if temp.check(slot):
                if solve(temp, available + tried, slot=slot + 1):
                    return True
        tried.append(current)
    return


pieces = [
    Piece(1, "", "", "", ""),
    Piece(2, "", "", "", ""),
    Piece(3, "", "", "", ""),
    Piece(4, "", "", "", ""),
    Piece(5, "", "", "", ""),
    Piece(6, "", "", "", ""),
    Piece(7, "", "", "", ""),
    Piece(8, "", "", "", ""),
    Piece(9, "", "", "", ""),
]

iteration = 0

board = Board()
if not solve(board, pieces):
    print("Solution not found")
