"""对 HOI 批次执行有界并发并按提交顺序返回。"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor


def ordered_parallel_map(function, items, workers):
    """最多保留 workers 个在途任务，并按原顺序返回或传播异常。"""
    if workers < 1:
        raise ValueError("并发批次数必须为正整数")
    iterator = iter(items)
    pending = deque()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hoi-batch") as executor:
        for _ in range(workers):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass
