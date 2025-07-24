import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification # For generating synthetic data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.special import expit

# --- Gradient Calculation Function ---
def calculate_gradients(model, X, y):
    """
    Calculates the gradient of the log-loss for each sample using a
    numerically stable method for a binary logistic regression model.
    The gradient for logistic loss with respect to weights 'w' is (sigmoid(z) - y) * x,
    where z = w^T * x, sigmoid is the logistic function, and y is the true label (0 or 1).
    This function handles the intercept term by appending a column of ones to X.
    """
    if not hasattr(model, 'coef_') or model.coef_.shape[0] == 0:
        # If the model is not fitted yet, or has no coefficients, return zero gradients.
        # This handles the very first call before any partial_fit has occurred.
        return np.zeros((X.shape[0], X.shape[1]))

    # Get the coefficients (weights) from the model. For SGDClassifier with binary
    # classification, model.coef_ will be a 2D array, so we take the first row.
    w = model.coef_[0]

    # Handle the intercept term. If the model includes an intercept, we append
    # a column of ones to the features matrix X, and append the intercept to w.
    if model.fit_intercept:
        w_ext = np.append(w, model.intercept_[0]) # Append intercept to weights
        X_b = np.c_[X, np.ones(X.shape[0])]      # Add bias (intercept) term to X
    else:
        w_ext = w
        X_b = X

    # Calculate the linear prediction: z = X_b * w_ext
    # This is the input to the sigmoid function.
    z = X_b @ w_ext

    # Use the numerically stable sigmoid function (expit = 1 / (1 + exp(-z))).
    # This avoids potential overflow issues when z is very large negative or positive,
    # which can occur with a direct np.exp() calculation.
    s = expit(z)

    # The gradient scale for logistic loss is (sigmoid(z) - y).
    # This is a key simplification for logistic regression gradients.
    scale = s - y

    # The full gradient for each sample is `scale` multiplied by the corresponding `X_b` row.
    # `np.newaxis` reshapes `scale` to allow broadcasting for element-wise multiplication.
    grads = scale[:, np.newaxis] * X_b

    # Return gradients, removing the intercept part if it was added.
    # The gradients for the features are the first X.shape[1] columns.
    return grads[:, :-1] if model.fit_intercept else grads

# --- OCSStreamer Class Implementation ---
class OCSStreamer:
    def __init__(self, m_coreset_size, batch_size, tau=1.0, random_seed=None):
        """
        Initializes the Online Coreset Selection (OCS) streamer.

        Args:
            m_coreset_size (int): The maximum size of the final coreset.
            batch_size (int): The size of data batches processed at each step.
            tau (float): Hyperparameter weighting the Coreset Affinity (sim_A).
                         A higher tau means more emphasis on affinity to past tasks.
            random_seed (int, optional): Seed for reproducibility.
        """
        self.m_coreset_size = m_coreset_size
        self.batch_size = batch_size
        self.tau = tau
        self.rng = np.random.default_rng(random_seed) # Random number generator for sampling

        # Initialize an SGDClassifier for logistic regression.
        # 'log_loss' is equivalent to logistic regression.
        # 'warm_start=True' allows the model to be trained incrementally using partial_fit.
        self.model = SGDClassifier(loss='log_loss', warm_start=True, random_state=random_seed)

        # Define all possible classes. This is crucial for `partial_fit` to correctly
        # initialize its internal structures, especially if not all classes are present
        # in the very first batch. Assuming binary classification for simplicity (0, 1).
        self.classes_ = np.array([0, 1])

        # Stores indices and their calculated OCS scores for all processed samples.
        # This list will be sorted later to select the final coreset.
        self.coreset_candidates_indices = []

        self.stream_processed_count = 0 # Counter for the total number of samples processed


    def process_batch(self, X_batch, y_batch, batch_indices, X_train_full, y_train_full):
        """
        Processes a single batch of streaming data to update coreset candidates.

        Args:
            X_batch (np.ndarray): Features of the current data batch.
            y_batch (np.ndarray): Labels of the current data batch.
            batch_indices (np.ndarray): Original indices of samples in X_batch within the
                                        full training dataset. Used to track samples.
            X_train_full (np.ndarray): The complete training feature set (needed to fetch
                                       coreset samples for affinity calculation).
            y_train_full (np.ndarray): The complete training label set.
        """
        if X_batch.shape[0] == 0:
            # If the batch is empty, do nothing.
            return

        # 1. Calculate Gradients for the current batch
        # These gradients represent the "informativeness" of each sample.
        grads_batch = calculate_gradients(self.model, X_batch, y_batch)

        # Handle cases where gradients might be all zeros (e.g., model not yet fitted or constant input)
        # Adding a small epsilon to norms to prevent division by zero if a norm is zero.
        norm_grads_batch = np.linalg.norm(grads_batch, axis=1, keepdims=True) + 1e-8
        avg_grad_batch = np.mean(grads_batch, axis=0, keepdims=True)
        norm_avg_grad_batch = np.linalg.norm(avg_grad_batch, axis=1, keepdims=True) + 1e-8

        # 2. Calculate Minibatch Similarity (sim_S) - Equation 3 in the paper
        # Measures how representative each sample's gradient is to the average gradient of the current batch.
        # (gradient_of_sample @ average_gradient_of_batch) / (norm_of_sample_gradient * norm_of_average_gradient)
        # It's essentially cosine similarity.
        sim_S = (grads_batch @ avg_grad_batch.T) / (norm_grads_batch * norm_avg_grad_batch)

        # 3. Calculate Sample Diversity (sim_V) - Equation 4 in the paper
        # Measures how diverse a sample is from other samples in the current batch.
        # It's an average of negative cosine similarities.
        if X_batch.shape[0] > 1:
            # Calculate pairwise cosine similarity matrix within the batch
            # Using (grads_batch @ grads_batch.T) / (norm_grads_batch @ norm_grads_batch.T) is numerically stable
            # and equivalent to cosine_similarity(grads_batch) if rows are L2-normalized.
            cosine_sim_matrix = (grads_batch @ grads_batch.T) / (norm_grads_batch @ norm_grads_batch.T)
            # Sum of similarities for each sample, subtract 1 to remove self-similarity, then average
            # and take negative as per the definition of diversity in the paper (negative similarity).
            sim_V = - (np.sum(cosine_sim_matrix, axis=1) - 1) / (X_batch.shape[0] - 1)
        else:
            # If batch size is 1, diversity is not applicable, so diversity score is 0.
            sim_V = np.zeros(X_batch.shape[0])

        # 4. Calculate Coreset Affinity (sim_A) - Equation 7 in the paper
        # Measures how well a sample's gradient aligns with the average gradient of the current coreset candidates.
        # This helps maintain knowledge of previous tasks by prioritizing samples similar to the coreset.
        sim_A = np.zeros(X_batch.shape[0])
        # Only calculate affinity if there are enough coreset candidates to sample from
        if len(self.coreset_candidates_indices) >= self.m_coreset_size:
            # Select the top 'm_coreset_size' samples from current candidates based on their scores.
            # This forms the 'current_coreset_indices' that represent past tasks.
            # The paper states "randomly sampled from the coreset C", so we sample from the *current*
            # set of highly-scored candidates.
            current_coreset_indices_sorted = sorted(self.coreset_candidates_indices, key=lambda x: x[1], reverse=True)
            current_coreset_top_indices = [idx for idx, score in current_coreset_indices_sorted[:self.m_coreset_size]]

            # Sample a subset from these top coreset candidates for efficiency.
            # The sample size is limited by the current batch_size or the number of available candidates.
            sample_size = min(len(current_coreset_top_indices), self.batch_size)
            coreset_sample_indices = self.rng.choice(current_coreset_top_indices, size=sample_size, replace=False)

            # Fetch features and labels for the sampled coreset elements from the full dataset.
            X_coreset_sample = X_train_full[coreset_sample_indices]
            y_coreset_sample = y_train_full[coreset_sample_indices]

            if X_coreset_sample.shape[0] > 0:
                # Calculate gradients for the sampled coreset elements.
                grads_coreset = calculate_gradients(self.model, X_coreset_sample, y_coreset_sample)
                avg_grad_coreset = np.mean(grads_coreset, axis=0, keepdims=True)
                norm_avg_grad_coreset = np.linalg.norm(avg_grad_coreset, axis=1, keepdims=True) + 1e-8

                # Calculate cosine similarity between batch gradients and average coreset gradient.
                sim_A = ((grads_batch @ avg_grad_coreset.T) / (norm_grads_batch * norm_avg_grad_coreset)).flatten()

        # 5. Combine scores - Equation 8 in the paper
        # The final score for each sample is a weighted sum of its similarity, diversity, and affinity.
        # The tau parameter weights the affinity component.
        final_scores = sim_S.flatten() + sim_V.flatten() + self.tau * sim_A.flatten()

        # 6. Add samples to coreset candidates
        # Store the original index of each sample along with its calculated score.
        for i in range(X_batch.shape[0]):
            self.coreset_candidates_indices.append((batch_indices[i], final_scores[i]))

        # 7. Update the model with the current batch (online learning step)
        # partial_fit allows the model to learn incrementally.
        # The `classes_` argument is essential here for the first call to `partial_fit`
        # if not all classes are present in the batch, or if the model hasn't seen all classes yet.
        self.model.partial_fit(X_batch, y_batch, classes=self.classes_)

        self.stream_processed_count += X_batch.shape[0]

    def get_final_coreset_details(self, X_train_full, y_train_full):
        """
        Selects the final coreset from all processed candidates based on their scores.

        Args:
            X_train_full (np.ndarray): The complete training feature set.
            y_train_full (np.ndarray): The complete training label set.

        Returns:
            tuple: (X_core, y_core, w_core)
                X_core (np.ndarray): Features of the selected coreset.
                y_core (np.ndarray): Labels of the selected coreset.
                w_core (np.ndarray): Weights for each coreset sample (uniform weights here).
        """
        # Sort all coreset candidates by their score in descending order.
        self.coreset_candidates_indices.sort(key=lambda x: x[1], reverse=True)

        # Select the top 'm_coreset_size' indices.
        final_indices = [idx for idx, score in self.coreset_candidates_indices[:self.m_coreset_size]]

        # Store the selected coreset indices.
        self.coreset_indices = np.array(final_indices, dtype=int)

        # Retrieve the actual data for the selected coreset.
        X_core = X_train_full[self.coreset_indices]
        y_core = y_train_full[self.coreset_indices]

        # Assign uniform weights to coreset samples, as is common in many coreset methods.
        # The paper mentions 'replayed later to alleviate catastrophic forgetting'
        # and 'small fraction of data points for previous tasks'.
        w_core = np.ones(len(X_core)) / len(X_core) # Uniform weights for simplicity

        return X_core, y_core, w_core