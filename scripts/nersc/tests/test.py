import numpy as np
from parallel_square import parallel_square


def main():
    numbers = np.arange(10)
    parallel_square(numbers)


if __name__ == "__main__":
    main()
