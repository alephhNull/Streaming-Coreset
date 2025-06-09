import numpy as np

# Assuming ReservoirSamplerBatchStreamer is defined as in the previous response
class ReservoirSamplerBatchStreamer:
    def __init__(self, coreset_size, random_seed=42):
        self.m = coreset_size; self.random_seed=random_seed; np.random.seed(random_seed)
        self.reservoir_global_indices = []; self._items_processed_count = 0
    def process_batch(self, batch_global_start_idx, batch_size):
        if self.m == 0: return
        for i in range(batch_size):
            current_item_global_idx = batch_global_start_idx + i
            self._items_processed_count += 1
            if len(self.reservoir_global_indices) < self.m: self.reservoir_global_indices.append(current_item_global_idx)
            else:
                j = np.random.randint(0, self._items_processed_count)
                if j < self.m: self.reservoir_global_indices[j] = current_item_global_idx
    def get_coreset_indices(self): return np.array(self.reservoir_global_indices, dtype=int)
    def get_final_coreset_details(self, all_stream_data_accumulator_np, all_stream_labels_accumulator_np=None):
        indices = self.get_coreset_indices()
        valid_indices = indices[indices < len(all_stream_data_accumulator_np)]
        coreset_X = all_stream_data_accumulator_np[valid_indices]
        weights = np.ones(len(valid_indices)) / (len(valid_indices) if len(valid_indices)>0 else 1)
        if all_stream_labels_accumulator_np is not None:
            coreset_y = all_stream_labels_accumulator_np[valid_indices]
            return coreset_X, coreset_y, weights
        return coreset_X, weights