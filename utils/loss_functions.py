import torch
from typing import Literal

class LossFunctions():
    def __init__(self, epsilon=1e-9, cost_matrix=None):
        self.epsilon = epsilon
        self.cost_matrix = cost_matrix
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # Convert cost_matrix to tensor if provided
        if self.cost_matrix is not None:
            self.cost_matrix = torch.tensor(self.cost_matrix, dtype=torch.float32, device=self.device)
        else:
            # Default 5x5 identity-like matrix if no cost_matrix is provided
            self.cost_matrix = torch.tensor([
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ], dtype=torch.float32, device=self.device)

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
    
    def CELossLT_LossMult(self, output_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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

        # Use the configurable cost_matrix
        cost_matrix = self.cost_matrix

        _, predicted_classes = torch.max(output_logits, dim=1)

        cost_values = cost_matrix[predicted_classes, targets]
        cost_values = cost_values.view(-1, 1)
        # Softmax conversion

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

    def CELogitAdjustment(self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mode: Literal["standard", "positive", "negative"] = "negative"
    ) -> torch.Tensor:
        """
        Computes the cross-entropy loss from scratch, with optional logit modifications
        based on misclassification using cost matrix values.

        Args:
            logits (torch.Tensor): Logits predicted by the model (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels (batch_size).
            mode (str): One of ['standard', 'positive', 'negative'].
                        'positive': boost correct class logit on misclassification.
                        'negative': penalize correct class logit on misclassification.
                        'standard': regular CE.

        Returns:
            torch.Tensor: Computed scalar loss value.
        """
        # Clone logits to avoid in-place ops
        modified_logits = logits.clone()

        batch_size, num_classes = logits.shape

        # Get predicted class (argmax)
        pred_classes = torch.argmax(modified_logits, dim=1)

        # Gather predicted logits and target logits
        max_logits = modified_logits[range(batch_size), pred_classes]
        target_logits = modified_logits[range(batch_size), targets]

        # Identify misclassified examples
        misclassified = pred_classes != targets

        # Compute difference where misclassified
        diff = torch.abs(max_logits - target_logits)

        # Get cost values from cost matrix based on true label (row) and predicted label (column)
        # cost_matrix[true_label][predicted_label]
        if self.cost_matrix is not None:
            cost_values = self.cost_matrix[targets, pred_classes]  # Shape: (batch_size,)
        else:
            # Fallback to fixed values if cost matrix is not available
            cost_values = torch.ones_like(targets, dtype=torch.float32)

        print(f"cost_values: {cost_values}")
        if mode == "positive":
            # Apply correction: target_logit += cost_value * difference
            corrected = target_logits + cost_values * diff
        elif mode == "negative":
            # Apply correction: target_logit -= cost_value * difference
            corrected = target_logits - cost_values * diff
        else:
            corrected = target_logits  # no modification

        # Replace only misclassified target logits
        modified_logits[range(batch_size), targets] = torch.where(
            misclassified, corrected, target_logits
        )

        # Compute softmax and cross-entropy
        probs = torch.softmax(modified_logits, dim=-1)
        target_probs = probs[range(batch_size), targets]
        log_probs = -torch.log(target_probs + 1e-9)
        return log_probs.mean()

    def CELogitAdjustmentV2(self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the cross-entropy loss with logit adjustment by modifying the predicted class logits
        based on misclassification using cost matrix values.

        Args:
            logits (torch.Tensor): Logits predicted by the model (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels (batch_size).

        Returns:
            torch.Tensor: Computed scalar loss value.
        """
        # Clone logits to avoid in-place ops
        modified_logits = logits.clone()

        batch_size, num_classes = logits.shape

        # Get predicted class (argmax)
        pred_classes = torch.argmax(modified_logits, dim=1)

        # Gather predicted logits and target logits
        max_logits = modified_logits[range(batch_size), pred_classes]
        target_logits = modified_logits[range(batch_size), targets]

        # Identify misclassified examples
        misclassified = pred_classes != targets

        # Compute difference where misclassified
        diff = torch.abs(max_logits - target_logits)

        # Get cost values from cost matrix based on true label (row) and predicted label (column)
        # cost_matrix[true_label][predicted_label]
        if self.cost_matrix is not None:
            cost_values = self.cost_matrix[targets, pred_classes]  # Shape: (batch_size,)
        else:
            # Fallback to fixed values if cost matrix is not available
            cost_values = torch.ones_like(targets, dtype=torch.float32)

        # Apply correction: max_logit += cost_value * difference
        # This penalizes the incorrectly predicted class by increasing its logit further,
        # making it even more confident in the wrong prediction, which should increase the loss
        corrected_max_logits = max_logits + cost_values * diff

        # Replace only misclassified predicted class logits (not target logits)
        modified_logits[range(batch_size), pred_classes] = torch.where(
            misclassified, corrected_max_logits, max_logits
        )

        # Compute softmax and cross-entropy
        probs = torch.softmax(modified_logits, dim=-1)
        target_probs = probs[range(batch_size), targets]
        log_probs = -torch.log(target_probs + 1e-9)
        return log_probs.mean()
        
    def CELossLTV1(self, output_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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

        # Use the configurable cost_matrix
        cost_matrix = self.cost_matrix

        _, predicted_classes = torch.max(output_logits, dim=1)

        cost_values = cost_matrix[predicted_classes, targets]
        cost_values = cost_values.view(-1, 1)

        ### Softmax conversion
        weighted_logits = output_logits * (alpha * cost_values)

        weighted_probs = torch.softmax(weighted_logits, dim=-1)
        target_probs = weighted_probs[range(batch_size), targets]

        ### Cross Entropy Loss
        log_probs = -torch.log(target_probs + 1e-9)  # Add small value to avoid log(0)

        loss = log_probs.mean()
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
            return self.CELossLTV1
        elif loss_name == "test":
            return self.CELogitAdjustmentV2
        else:
            raise ValueError(f"Invalid loss function: {loss_name}")
