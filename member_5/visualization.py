import time


def run_benchmark(spec, iterations=5):
    from member_3.time_warp import time_warp
    from member_3.masking import freq_mask, time_mask

    print(f"  Running {iterations} iterations of each transform...\n")

    for name, fn in [("time_warp", lambda: time_warp(spec)),
                     ("freq_mask", lambda: freq_mask(spec, num_masks=2)),
                     ("time_mask", lambda: time_mask(spec, num_masks=2)),
                     ("combined (all three)", lambda: time_mask(
                         freq_mask(time_warp(spec), num_masks=2), num_masks=2))]:
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times)
        print(f"  {name:25s}  {avg*1000:8.2f} ms  (over {iterations} runs)")
