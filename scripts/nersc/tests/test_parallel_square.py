import unittest
from io import StringIO
from contextlib import redirect_stdout
from parallel_square import parallel_square


class TestParallelSquare(unittest.TestCase):
    def test_parallel_square(self):
        numbers = [1, 2, 3, 4, 5]
        expected_result = [1, 4, 9, 16, 25]

        with StringIO() as buffer, redirect_stdout(buffer):
            result = parallel_square(numbers)

            # Retrieve the printed output
            printed_output = buffer.getvalue()

        self.assertEqual(result, expected_result)

        for i, number in enumerate(numbers):
            self.assertIn(f"Thread ThreadPoolExecutor-", printed_output)
            self.assertIn(f"Squaring {number}", printed_output)


if __name__ == "__main__":
    unittest.main()
