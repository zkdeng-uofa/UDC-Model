import torch

class LossFunctions():
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

    def calculate_dynamic_alpha(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Dynamically calculates alpha based on the statistical properties of the logits.

        Args:
            logits (torch.Tensor): Logits predicted by the model (batch_size, num_classes).

        Returns:
            torch.Tensor: The dynamically calculated alpha value.
        """
        # Calculate the absolute mean and standard deviation of the logits
        mean_logit = torch.mean(torch.abs(logits))
        std_logit = torch.std(logits)

        # Calculate alpha using the mathematically defined formula
        alpha = mean_logit / (std_logit + self.epsilon)
        return alpha

    def cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-entropy loss from scratch.

        Args:
            logits (torch.Tensor): Logits predicted by the model (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels (batch_size).

        Returns:
            torch.Tensor: Computed scalar loss value.
        """
        # Convert logits to probabilities using softmax
        probs = torch.softmax(logits, dim=-1)

        # Select the predicted probabilities corresponding to the target class
        batch_size = logits.shape[0]
        target_probs = probs[range(batch_size), targets]

        # Take the log of the probabilities
        log_probs = -torch.log(target_probs + 1e-9)  # Add small value to avoid log(0)

        # Compute the mean loss
        loss = log_probs.mean()
        return loss
    
    def CELossLT(self, output_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-entropy loss from scratch.

        Args:
            logits (torch.Tensor): Logits predicted by the model (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels (batch_size).

        Returns:
            torch.Tensor: Computed scalar loss value.
        """
        alpha = self.calculate_dynamic_alpha(output_logits)
        #print(f"Dynamically calculated alpha: {alpha}")
        batch_size = output_logits.shape[0]
        # cost_matrix = torch.tensor([
        #     [100.0, 1.0, 1.0, 1.0, 1.0],  # Class 0 misclassification costs
        #     [1.0, 100.0, 1.0, 1.0, 1.0],  # Class 1
        #     [1.0, 1.0, 100.0, 1.0, 1.0],  # Class 2
        #     [1.0, 1.0, 1.0, 100.0, 1.0],  # Class 3
        #     [1.0, 1.0, 1.0, 1.0, 100.0],  # Class 4
        # ], dtype=torch.float32, device=self.device)

        # cost_matrix = torch.tensor([
        #     [1.0, 5.0, 5.0, 5.0, 5.0],  # Class 0 misclassification costs
        #     [5.0, 1.0, 5.0, 5.0, 5.0],  # Class 1
        #     [5.0, 5.0, 1.0, 5.0, 5.0],  # Class 2
        #     [5.0, 5.0, 5.0, 1.0, 5.0],  # Class 3
        #     [5.0, 5.0, 5.0, 5.0, 1.0],  # Class 4
        # ], dtype=torch.float32, device=self.device)

        # cost_matrix = torch.tensor([
        #     [10.0, 10.0, 10.0, 10.0, 10.0],  # Class 0 misclassification costs
        #     [10.0, 10.0, 10.0, 10.0, 10.0],  # Class 1
        #     [10.0, 10.0, 10.0, 10.0, 10.0],  # Class 2
        #     [10.0, 10.0, 10.0, 1.0, 10.0],  # Class 3
        #     [10.0, 10.0, 10.0, 10.0, 10.0],  # Class 4
        # ], dtype=torch.float32, device=self.device)

        cost_matrix = torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 1.0],  # Class 0 misclassification costs
            [1.0, 1.0, 1.0, 1.0, 1.0],  # Class 1
            [1.0, 1.0, 1.0, 1.0, 1.0],  # Class 2
            [1.0, 1.0, 1.0, 1.0, 1.0],  # Class 3
            [1.0, 1.0, 1.0, 1.0, 1.0],  # Class 4
        ], dtype=torch.float32, device=self.device)

        # cost_matrix = torch.tensor([
        #     [1.0, 5.0, 2.0, 3.0, 4.0],  # Class 0 misclassification costs
        #     [4.0, 1.0, 3.0, 2.0, 5.0],  # Class 1
        #     [2.0, 3.0, 1.0, 5.0, 4.0],  # Class 2
        #     [3.0, 2.0, 5.0, 1.0, 4.0],  # Class 3
        #     [4.0, 5.0, 2.0, 3.0, 1.0],  # Class 4
        # ], dtype=torch.float32, device=self.device)

        # cost_matrix = torch.tensor([
        #     [1.0, 1.0, 1.0, 1.0, 1.0],  # Class 0 misclassification costs
        #     [2.0, 2.0, 2.0, 2.0, 2.0],  # Class 1
        #     [3.0, 3.0, 3.0, 3.0, 3.0],  # Class 2
        #     [4.0, 4.0, 4.0, 4.0, 4.0],  # Class 3
        #     [5.0, 5.0, 5.0, 5.0, 5.0],  # Class 4
        # ], dtype=torch.float32, device=self.device)


        _, predicted_classes = torch.max(output_logits, dim=1)

        cost_values = cost_matrix[predicted_classes, targets]
        cost_values = cost_values.view(-1, 1)
        # Softmax conversion

        #output_logits_values = output_logits[-1]
        #print(f"output_logits_shape: {output_logits.shape}")
        # probs = torch.softmax(output_logits, dim=-1)
        # weighted_probs = probs * cost_values
        # weighted_probs = weighted_probs / weighted_probs.sum(dim=-1, keepdim=True)
        # target_probs = weighted_probs[range(batch_size), targets]
        #print(f"output_logits: {output_logits}, cost_values: {cost_values}, prediction/target: {predicted_classes} / {target}")


        # weighted_logits = output_logits * cost_values
        # weighted_logits = output_logits + cost_values + 1e-9
        weighted_logits = output_logits * (alpha * cost_values)


        weighted_probs = torch.softmax(weighted_logits, dim=-1)
        target_probs = weighted_probs[range(batch_size), targets]

        #print(f"output_logits: {output_logits}, weighted_logits: {weighted_logits}, cost_values: {cost_values}")
        # One hot encoding of target

        #target_probs = probs[range(batch_size), target]

        # Cross Entropy Loss
        log_probs = -torch.log(target_probs + 1e-9)  # Add small value to avoid log(0)

        loss = log_probs.mean()

        loss = loss * cost_values.mean()
        return loss
    
    def cost_matrix_cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-entropy loss from scratch.

        Args:
            logits (torch.Tensor): Logits predicted by the model (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels (batch_size).

        Returns:
            torch.Tensor: Computed scalar loss value.
        """
        device = logits.device
        # Convert logits to probabilities using softmax
        probs = torch.softmax(logits, dim=-1)

        # Select the predicted probabilities corresponding to the target class
        batch_size, num_classes = logits.shape

        one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()

        log_probs = torch.log(probs + 1e-9)
        cost_matrix = torch.tensor([
            [1.0, 5.0, 2.0, 3.0, 4.0],  # Class 0 misclassification costs
            [4.0, 1.0, 3.0, 2.0, 5.0],  # Class 1
            [2.0, 3.0, 1.0, 5.0, 4.0],  # Class 2
            [3.0, 2.0, 5.0, 1.0, 4.0],  # Class 3
            [4.0, 5.0, 2.0, 3.0, 1.0],  # Class 4
        ], dtype=torch.float32, device=device)

        print(type(cost_matrix))
        print(type(targets))
        targets = targets.to(device)
        #cost_matrix = cost_matrix.to(logits.device)
        cost_weights = cost_matrix[targets]
        # cost_matrix.to(torch.device("mps"))
        # cost_weights.to(torch.device("mps"))

        weighted_log_probs = cost_weights * log_probs 

        # Take the log of the probabilities
        loss = -torch.sum(one_hot_targets * weighted_log_probs, dim=-1).mean()
    
        # Compute the mean loss
        return loss

    def seesaw_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 2.0,
        beta: float = 0.8,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Implements the Seesaw Loss function.

        Args:
            logits (torch.Tensor): Logits predicted by the model (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels (batch_size).
            alpha (float): Scaling factor for the positive sample term.
            beta (float): Scaling factor for the negative sample term.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

        Returns:
            torch.Tensor: Computed scalar loss value.
        """
        num_classes = logits.size(-1)
        batch_size = logits.size(0)

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Create one-hot encoded target tensor
        target_one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)

        # Positive and negative logits
        pos_logits = logits * target_one_hot
        neg_logits = logits * (1 - target_one_hot)

        # Positive term
        pos_probs = probs * target_one_hot
        pos_loss = -alpha * torch.log(pos_probs + 1e-9) * target_one_hot

        # Negative term
        neg_probs = probs * (1 - target_one_hot)
        neg_factor = torch.pow(1 - neg_probs, beta)
        neg_loss = -neg_factor * torch.log(1 - probs + 1e-9) * (1 - target_one_hot)

        # Total loss
        loss = pos_loss + neg_loss
        if reduction == "mean":
            return loss.sum() / batch_size
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss
        
    def loss_function(self, loss_name):
        if loss_name == "cross_entropy":
            return self.cross_entropy
        elif loss_name == "seesaw":
            return self.seesaw_loss
        elif loss_name == "cost_matrix_cross_entropy":
            return self.CELossLT
        else:
            raise ValueError(f"Invalid loss function: {loss_name}")


    
# def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#     """
#     Computes the cross-entropy loss from scratch.

#     Args:
#         logits (torch.Tensor): Logits predicted by the model (batch_size, num_classes).
#         targets (torch.Tensor): Ground truth labels (batch_size).

#     Returns:
#         torch.Tensor: Computed scalar loss value.
#     """
#     # Convert logits to probabilities using softmax
#     probs = torch.softmax(logits, dim=-1)

#     # Select the predicted probabilities corresponding to the target class
#     batch_size = logits.shape[0]
#     target_probs = probs[range(batch_size), targets]

#     # Take the log of the probabilities
#     log_probs = -torch.log(target_probs + 1e-9)  # Add small value to avoid log(0)

#     # Compute the mean loss
#     loss = log_probs.mean()
#     return loss

# def seesaw_loss(
#     logits: torch.Tensor,
#     targets: torch.Tensor,
#     alpha: float = 2.0,
#     beta: float = 0.8,
#     reduction: str = "mean"
# ) -> torch.Tensor:
#     """
#     Implements the Seesaw Loss function.

#     Args:
#         logits (torch.Tensor): Logits predicted by the model (batch_size, num_classes).
#         targets (torch.Tensor): Ground truth labels (batch_size).
#         alpha (float): Scaling factor for the positive sample term.
#         beta (float): Scaling factor for the negative sample term.
#         reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

#     Returns:
#         torch.Tensor: Computed scalar loss value.
#     """
#     num_classes = logits.size(-1)
#     batch_size = logits.size(0)

#     # Convert logits to probabilities
#     probs = torch.softmax(logits, dim=-1)

#     # Create one-hot encoded target tensor
#     target_one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)

#     # Positive and negative logits
#     pos_logits = logits * target_one_hot
#     neg_logits = logits * (1 - target_one_hot)

#     # Positive term
#     pos_probs = probs * target_one_hot
#     pos_loss = -alpha * torch.log(pos_probs + 1e-9) * target_one_hot

#     # Negative term
#     neg_probs = probs * (1 - target_one_hot)
#     neg_factor = torch.pow(1 - neg_probs, beta)
#     neg_loss = -neg_factor * torch.log(1 - probs + 1e-9) * (1 - target_one_hot)

#     # Total loss
#     loss = pos_loss + neg_loss
#     if reduction == "mean":
#         return loss.sum() / batch_size
#     elif reduction == "sum":
#         return loss.sum()
#     else:
#         return loss
    
# def cost_matrix_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         """
#         Computes the cross-entropy loss from scratch.

#         Args:
#             logits (torch.Tensor): Logits predicted by the model (batch_size, num_classes).
#             targets (torch.Tensor): Ground truth labels (batch_size).

#         Returns:
#             torch.Tensor: Computed scalar loss value.
#         """
#         # Convert logits to probabilities using softmax
#         device = logits.device
#         probs = torch.softmax(logits, dim=-1)

#         # Select the predicted probabilities corresponding to the target class
#         batch_size, num_classes = logits.shape

#         one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()

#         log_probs = torch.log(probs + 1e-9)
#         cost_matrix = torch.tensor([
#             [1.0, 5.0, 2.0, 3.0, 4.0],  # Class 0 misclassification costs
#             [4.0, 1.0, 3.0, 2.0, 5.0],  # Class 1
#             [2.0, 3.0, 1.0, 5.0, 4.0],  # Class 2
#             [3.0, 2.0, 5.0, 1.0, 4.0],  # Class 3
#             [4.0, 5.0, 2.0, 3.0, 1.0],  # Class 4
#         ], dtype=torch.float32, device=device)

#         cost_matrix = torch.tensor([
#             [1.0, 5.0, 5.0, 5.0, 5.0],  # Class 0 misclassification costs
#             [5.0, 1.0, 5.0, 5.0, 5.0],  # Class 1
#             [5.0, 5.0, 1.0, 5.0, 5.0],  # Class 2
#             [5.0, 5.0, 5.0, 1.0, 5.0],  # Class 3
#             [5.0, 5.0, 5.0, 5.0, 1.0],  # Class 4
#         ], dtype=torch.float32, device=device)
#         targets = targets.to(device)

#         cost_weights = cost_matrix[targets]

#         weighted_log_probs = cost_weights * log_probs 

#         # Take the log of the probabilities
#         loss = -torch.sum(one_hot_targets * weighted_log_probs, dim=-1).mean()
    
#         # Compute the mean loss
#         return loss
