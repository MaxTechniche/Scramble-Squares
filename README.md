# Scramble Squares Puzzle Solver

## Instructions

1. Lay out the 9 squares in a 3x3 grid.
  - Doesn't matter the order.
  - From left to right, top to bottom, the squares will be given the numbers 1 - 9.
  - Keep them in this position until the program gives you the solution.
2. Input the information on each square starting with square 1.
  - Starting at the top and going clockwise around each square, enter the given piece of information on that edge.
    - The program uses the letters a, b, c, and d to differentiate the 4 pictures on the grid and the numbers 1 and 2 for the 'top' and 'bottom' of the picture.
    - "a1" = the top of picture 1, while "c2" = the bottom of picture 3
  - There are 4 pieces of information on each square
    - square 1 = Square("a2", "b1", "d2", "b2").....
3. Run the Python file.
  - The program should output a list of the squares.
4. Assemble the solved grid.
  - Following the list, grab the corresponding square from the previous grid and place it into the new grid.
  - starting at the top left, going left to right, top to bottom, place all nine squares into the grid.
    - The order in which the letter number pairs are in, tells you how the square should be rotated.
    - If the output for square 1 above is ["d2", "b2", "a2", "b1"], the bottom of the picture that corresponds to "d2" will be placed on top. ("b2" on right. "a2" on bottom, and "b1" on left.)
5. Double check the grid.
  - Make sure all the pictures match.
  - If they don't, make sure the information for each square was input correctly, and run the Python file again.
