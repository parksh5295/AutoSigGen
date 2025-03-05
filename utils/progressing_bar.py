from tqdm import tqdm
from contextlib import contextmanager   # for Automatically handle progress bars


@contextmanager     # Context manager to automatically manage TQDM progress bars
def progress_bar(total_len, desc="Processing", unit="items"):
    print("Clustering in Progress...")

    with tqdm(total=total_len, desc=desc, unit=unit) as pbar:
        yield lambda x: pbar.update(x)  # Provide an update function