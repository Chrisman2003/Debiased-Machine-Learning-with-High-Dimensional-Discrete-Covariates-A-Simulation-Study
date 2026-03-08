import math
import time
from joblib import Parallel, delayed

def factorial_chunk(start, end):
    return [math.factorial(x) for x in range(start, end)]

t1 = time.perf_counter()

results = Parallel(n_jobs=2)(
    delayed(factorial_chunk)(i, i + 5000)
    for i in range(0, 10000, 5000)
)

results = [item for sublist in results for item in sublist]

t2 = time.perf_counter()

print(f"Time taken: {t2 - t1:.4f} seconds")