import math 
import time
from joblib import Parallel, delayed

if __name__ == "__main__":
    t1 = time.time()
    #results = [math.factorial(x) for x in range(10000)]
    results = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(math.factorial)(x) for x in range(10000)
    )
    t2 = time.time()

    print(f"Time taken: {t2 - t1:.4f} seconds")