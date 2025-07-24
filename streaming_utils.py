import time

def stream_simulator_gen(data_loader, arrival_interval_ms):
    """
    A generator that simulates a data stream, yielding batches with their global index and stats.
    It can operate in a normal mode or a high-velocity mode where batches can be dropped.
    
    Yields:
        tuple: (batch_idx, batch_data, stats)
            - batch_idx (int): The global, 0-based index of the batch in the original stream.
            - batch_data (tuple): The data from the data_loader (e.g., (features, labels)).
            - stats (dict): A dictionary with processing statistics ('processed', 'dropped', 'total').
    """
    if arrival_interval_ms is None:
        total_batches = len(data_loader)
        for i, batch_data in enumerate(data_loader):
            stats = {"processed": i + 1, "dropped": 0, "total": total_batches}
            yield i, batch_data, stats
        return


    data_iterator = iter(data_loader)
    batches_processed = 0
    batches_dropped = 0
    total_batches = len(data_loader)
    arrival_interval_s = arrival_interval_ms / 1000.0
    
    global_batch_index = 0
    
    next_arrival_time = time.monotonic() + arrival_interval_s

    while global_batch_index < total_batches:
        current_time = time.monotonic()

        try:
            if current_time > next_arrival_time:
                # BATCH DROPPED
                _ = next(data_iterator)  # Consume the batch from the iterator
                batches_dropped += 1
            else:
                # BATCH PROCESSED
                wait_time = next_arrival_time - current_time
                time.sleep(wait_time)
                
                batch_data = next(data_iterator)
                batches_processed += 1
                stats = {"processed": batches_processed, "dropped": batches_dropped, "total": total_batches}
                
                # ✅ Yield the correct global index along with the data
                yield global_batch_index, batch_data, stats

            # ✅ Increment the global index and schedule the next arrival
            global_batch_index += 1
            next_arrival_time += arrival_interval_s

        except StopIteration:
            break
    print() # Newline after the simulation progress bar