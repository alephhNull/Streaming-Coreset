import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Assuming SimpleNN is defined elsewhere or above this class
class SimpleNN(nn.Module):
    """
    A simple neural network for classification, suitable for the adult dataset.
    This will serve as M_tr and M_cs in the BCSR algorithm.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid() # For binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

class BilevelCoresetSelector:
    """
    Implementation of Bilevel Coreset Selection via Regularization (BCSR).
    Follows the principles outlined in the provided research paper,
    making reasonable interpretations where specific mathematical forms
    or detailed algorithmic steps are not explicitly given.
    """
    def __init__(self, input_dim, m_coreset_size, outer_loops=10, inner_loops=5,
                 lr_outer=0.01, lr_inner=0.001, lambda_reg=0.01, random_seed=42):
        
        self.m_coreset_size = m_coreset_size
        self.outer_loops = outer_loops # J in Algorithm 2 (Outer optimization steps for w)
        self.inner_loops = inner_loops # N in Algorithm 2 (Inner optimization steps for theta)
        self.lr_outer = lr_outer     # Learning rate for updating coreset weights (w)
        self.lr_inner = lr_inner     # Learning rate for updating inner model parameters (theta)
        self.lambda_reg = lambda_reg # Regularization parameter for the smoothed top-K loss
        self.random_state = np.random.RandomState(random_seed)
        
        # Initialize two neural network models as per the paper (M_tr and M_cs)
        self.model_tr = SimpleNN(input_dim) # M_tr: model for overall training / outer loop context
        self.model_cs = SimpleNN(input_dim) # M_cs: proxy model for coreset selection (inner loop)
        
        # Initialize M_cs with weights from M_tr (Algorithm 2, line 8, in principle)
        self.model_cs.load_state_dict(self.model_tr.state_dict()) 

        self.loss_fn = nn.BCELoss(reduction='none') # Binary Cross Entropy Loss, `none` for per-sample loss

        # Buffers to store all seen data and their original flat indices
        self.data_buffer = []      # Stores X values
        self.label_buffer = []     # Stores y values (labels)
        self.index_buffer = []     # Stores original global indices
        self.total_points_seen = 0 # Total number of points streamed so far

        # Coreset weights (probability distribution 'w' over the samples in the buffer)
        self.coreset_weights = None 

        # Full training data and labels (needed for upper-level objective evaluation)
        self.X_train_full = None
        self.y_train_full = None
    
    def set_full_data(self, X_train_full, y_train_full):
        """
        Sets the full training dataset. This is used to evaluate the upper-level
        objective, which aims to minimize the training error over the *whole data*.
        """
        self.X_train_full = X_train_full
        self.y_train_full = y_train_full

    def _smoothed_top_k_regularizer(self, w_raw, K):
        """
        Approximation of the smoothed top-K regularizer R(w, delta).
        This function is designed to penalize the sum of probabilities of elements
        that are *not* among the top K largest weighted samples.
        The goal is to encourage sparsity in the `w` distribution, pushing mass
        towards a few (approximately K) dominant samples.

        Parameters:
            w_raw (torch.Tensor): The raw (pre-softmax/pre-projection) coreset weights.
            K (int): The target coreset size.
        Returns:
            torch.Tensor: The scalar regularization loss to be minimized.
        """
        # Ensure w is a valid probability distribution before sorting and penalizing
        # Using softmax to create a differentiable probability distribution from raw weights.
        w_prob = torch.softmax(w_raw, dim=-1)
        
        # Sort the probabilities in descending order
        sorted_w_prob, _ = torch.sort(w_prob, descending=True)
        
        # The penalty is the sum of probabilities of elements beyond the top K.
        # If the number of elements is less than or equal to K, there's no penalty.
        if sorted_w_prob.size(0) > K:
            # Sum the probabilities of the (N-K) smallest elements.
            # Minimizing this sum will encourage these probabilities to become very small.
            penalty = torch.sum(sorted_w_prob[K:])
        else:
            penalty = torch.tensor(0.0, dtype=torch.float32) # No penalty if K is >= current size
        
        return penalty

    def _project_onto_simplex(self, w_tensor_data): 
        """ 
        Projects a tensor w onto the probability simplex.
        This ensures that w_i >= 0 and sum(w_i) = 1.
        Uses a standard algorithm for projection onto the simplex.
        """
        # Sort values in descending order
        sorted_w, _ = torch.sort(w_tensor_data, descending=True) 
        
        # Calculate cumulative sum
        cumsum_sorted_w = torch.cumsum(sorted_w, dim=0)
        
        # Find rho, the largest index such that sorted_w[rho] - (cumsum_sorted_w[rho] - 1) / (rho + 1) > 0
        rho = 0
        for i in range(len(sorted_w)):
            threshold = (cumsum_sorted_w[i] - 1) / (i + 1)
            if sorted_w[i] - threshold > 0:
                rho = i
            else:
                break
        
        theta = (cumsum_sorted_w[rho] - 1) / (rho + 1)
        w_proj = torch.relu(w_tensor_data - theta) 
        
        if torch.sum(w_proj).item() > 1e-6:
             w_proj = w_proj / torch.sum(w_proj)
        else:
             w_proj = torch.ones_like(w_tensor_data) / w_tensor_data.size(0) 
        return w_proj 

    def process_batch(self, X_batch, y_batch):
        """
        Processes a new batch of data from the stream, incorporating it into the buffer
        and performing bilevel optimization to update coreset weights.

        Parameters:
            X_batch (np.ndarray): Features of the current data batch.
            y_batch (np.ndarray): Labels of the current data batch.
        """
        # Store current batch data and their original indices
        current_batch_indices = np.arange(self.total_points_seen, self.total_points_seen + len(X_batch))
        self.data_buffer.extend(X_batch)
        self.label_buffer.extend(y_batch)
        self.index_buffer.extend(current_batch_indices)
        self.total_points_seen += len(X_batch)

        # Initialize or resize coreset weights (w) for the entire current buffer.
        current_buffer_size = len(self.data_buffer)
        if self.coreset_weights is None or self.coreset_weights.size(0) != current_buffer_size:
            if self.coreset_weights is None:
                self.coreset_weights = torch.ones(current_buffer_size, dtype=torch.float32)
            else:
                old_len = self.coreset_weights.size(0)
                new_weights_part = torch.ones(current_buffer_size - old_len, dtype=torch.float32) 
                self.coreset_weights = torch.cat([self.coreset_weights * (old_len / current_buffer_size), new_weights_part])
            
            self.coreset_weights = self.coreset_weights / torch.sum(self.coreset_weights)
            self.coreset_weights.requires_grad_(True)
            self.coreset_weights.retain_grad() # Crucial: Ensure gradients are retained for this non-leaf tensor
        
        # Convert the full buffer data to PyTorch tensors for processing
        x_buffer_tensor = torch.tensor(np.array(self.data_buffer)).float()
        y_buffer_tensor = torch.tensor(np.array(self.label_buffer)).float().unsqueeze(1)

        # --- Algorithm 3: Find-coreset (Outer Loop for w optimization) ---
        for j in range(self.outer_loops):
            self.model_cs.load_state_dict(self.model_tr.state_dict())
            
            # Optimizer for the inner-level problem (optimizing M_cs parameters)
            # Re-initialize optimizer for each outer loop iteration to reset its state (e.g., momentum)
            optimizer_cs = optim.Adam(self.model_cs.parameters(), lr=self.lr_inner) 
            
            # --- Inner Loop (for theta optimization) ---
            for i in range(self.inner_loops):
                self.model_cs.zero_grad() # Zero gradients for model_cs parameters at the start of each inner step

                current_coreset_weights = self.coreset_weights.detach()
                current_coreset_weights = torch.relu(current_coreset_weights) 
                if torch.sum(current_coreset_weights).item() == 0:
                    current_coreset_weights = torch.ones_like(current_coreset_weights) 
                current_coreset_weights = current_coreset_weights / torch.sum(current_coreset_weights) 

                if current_buffer_size == 0:
                    print("Warning: Data buffer is empty during inner loop. Skipping inner optimization.")
                    break 
                
                # BATCH_SIZE is now dynamically set to len(X_batch)
                num_samples_for_multinomial = min(len(X_batch), current_buffer_size)

                sample_indices_in_buffer = torch.multinomial(
                    current_coreset_weights, 
                    num_samples=num_samples_for_multinomial, 
                    replacement=True 
                )
                
                x_inner_batch = x_buffer_tensor[sample_indices_in_buffer]
                y_inner_batch = y_buffer_tensor[sample_indices_in_buffer]
                
                weights_for_sampled_batch = self.coreset_weights[sample_indices_in_buffer]
                
                inner_output = self.model_cs(x_inner_batch)
                lower_loss_per_sample = self.loss_fn(inner_output, y_inner_batch)
                
                lower_loss = torch.mean(lower_loss_per_sample * weights_for_sampled_batch)
                
                # Explicitly compute gradients for model_cs parameters and coreset_weights
                # This helps avoid `RuntimeError: Trying to backward through the graph a second time`
                # by controlling when and how gradients are computed and accumulated.
                model_cs_params_and_coreset_weights = list(self.model_cs.parameters()) + [self.coreset_weights]
                
                grads = torch.autograd.grad(lower_loss, model_cs_params_and_coreset_weights, retain_graph=True, allow_unused=True)
                
                # Assign gradients to model_cs parameters
                for param, grad in zip(self.model_cs.parameters(), grads[:-1]): # Last grad is for coreset_weights
                    if grad is not None:
                        if param.grad is None:
                            param.grad = grad
                        else:
                            param.grad.add_(grad) # Accumulate gradients

                # Accumulate gradients for coreset_weights
                coreset_w_grad = grads[-1]
                if coreset_w_grad is not None:
                    if self.coreset_weights.grad is None:
                        self.coreset_weights.grad = coreset_w_grad
                    else:
                        self.coreset_weights.grad.add_(coreset_w_grad)
                
                optimizer_cs.step() 

            # Update `coreset_weights` (w) using its accumulated gradients
            if self.coreset_weights.grad is not None:
                # Calculate the gradient from the regularization term separately.
                reg_loss = self.lambda_reg * self._smoothed_top_k_regularizer(self.coreset_weights, self.m_coreset_size)
                
                # Get gradients of reg_loss w.r.t. coreset_weights using torch.autograd.grad
                # This prevents the "backward through graph a second time" error
                reg_grad_w = torch.autograd.grad(reg_loss, self.coreset_weights, allow_unused=True)[0]
                
                if reg_grad_w is not None:
                    self.coreset_weights.grad.add_(reg_grad_w) # Add regularization gradient
                
                with torch.no_grad():
                    # Update weights data (in-place modification of the underlying tensor data)
                    self.coreset_weights.data -= self.lr_outer * self.coreset_weights.grad.data
                    
                    # Project the updated weights data onto the simplex.
                    projected_weights_data = self._project_onto_simplex(self.coreset_weights.data) 
                    
                    # Copy the projected data back to self.coreset_weights.data (in-place)
                    self.coreset_weights.data.copy_(projected_weights_data) 
                    
                    # Clear gradients for the next outer loop iteration
                    self.coreset_weights.grad.zero_() 
            else:
                print("Warning: Coreset weights gradients are None after inner loops. Re-initializing weights uniformly.")
                with torch.no_grad():
                    self.coreset_weights = torch.ones_like(self.coreset_weights) / self.coreset_weights.size(0)
                self.coreset_weights.requires_grad_(True) 
                self.coreset_weights.retain_grad() # Re-enable and retain grad for the new tensor

            self.model_tr.load_state_dict(self.model_cs.state_dict())


    def get_final_coreset(self):
        """
        After streaming and bilevel optimization, selects the final coreset
        based on the learned coreset weights (w).
        """
        if self.coreset_weights is None or len(self.data_buffer) == 0:
            return np.array([]), np.array([])
        
        positive_weights = torch.relu(self.coreset_weights)
        if torch.sum(positive_weights).item() == 0:
            print("Warning: All coreset weights are zero during final selection. Falling back to uniform selection.")
            positive_weights = torch.ones_like(positive_weights)
        
        num_to_select = min(self.m_coreset_size, len(self.data_buffer))
        if num_to_select == 0:
            return np.array([]), np.array([])

        topk_values, topk_indices_in_buffer = torch.topk(positive_weights, num_to_select)
        
        coreset_flat_indices = np.array(self.index_buffer)[topk_indices_in_buffer.numpy()]
        
        # Fix: Use .detach().numpy() to avoid RuntimeError on tensors that require grad
        coreset_weights_final = topk_values.detach().numpy() / np.sum(topk_values.detach().numpy())
        
        return coreset_flat_indices, coreset_weights_final