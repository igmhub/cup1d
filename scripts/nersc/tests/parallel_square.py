from multiprocessing.pool import ThreadPool
import threading


def square_number(number):
    thread_id = threading.current_thread().name
    print(f"Thread {thread_id}: Squaring {number}")
    return number**2


def parallel_square(numbers):
    with ThreadPool() as pool:
        squared_numbers = pool.map(square_number, numbers)
    return squared_numbers
