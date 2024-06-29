import unittest
from translateMove import output_space, move_to_int, int_to_move

class TestTranslateMove(unittest.TestCase):

    def test_output_space(self):
        for board_size in range(5, 22):
            expected_output = board_size**2 + 2 * (board_size - 1)**2
            self.assertEqual(output_space(board_size), expected_output,
                             f"Failed for board size {board_size}")

    def test_move_conversions(self):
        for board_size in range(5, 22):
            # Test player moves
            for x in range(board_size):
                for y in range(board_size):
                    move = [x, y]
                    move_int = move_to_int(board_size, move)
                    self.assertEqual(int_to_move(board_size, move_int), move,
                                     f"Failed for player move {move} on board size {board_size}")

            # Test placement moves
            for i in range(board_size - 1):
                for j in range(board_size - 1):
                    for h in range(2):
                        move = [i, j, h]
                        move_int = move_to_int(board_size, move)
                        self.assertEqual(int_to_move(board_size, move_int), move,
                                         f"Failed for placement move {move} on board size {board_size}")

            # Test invalid moveInt
            invalid_move_int = output_space(board_size) + 1
            with self.assertRaises(ValueError):
                int_to_move(board_size, invalid_move_int)

if __name__ == '__main__':
    unittest.main()
