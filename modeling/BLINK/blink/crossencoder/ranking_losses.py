"""
Ranking Loss Functions for Entity Linking
Integrates with existing reranker code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalRankingLoss(nn.Module):
    """
    Focal loss adapted for ranking - focuses on hard-to-rank examples
    where correct entity is not ranked first
    """
    def __init__(self, margin=1.0, gamma=2.0):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
    
    def forward(self, scores, labels):
        """
        Args:
            scores: [batch_size, num_candidates] - scores from reranker
            labels: [batch_size] - index of correct entity
        
        Returns:
            loss: scalar tensor
        """
        batch_size = scores.size(0)
        
        # Get correct entity scores
        correct_scores = scores.gather(1, labels.unsqueeze(1))
        
        # Find max score among negatives (hardest negative)
        negative_scores = scores.clone()
        negative_scores.scatter_(1, labels.unsqueeze(1), float('-inf'))
        max_negative_scores, _ = negative_scores.max(dim=1, keepdim=True)
        
        # Compute ranking margin violation
        ranking_violation = F.relu(self.margin + max_negative_scores - correct_scores)
        
        # Focal weighting
        normalized_violation = torch.clamp(ranking_violation / (self.margin + 1e-8), 0, 1)
        focal_weight = (normalized_violation) ** self.gamma
        
        # Weighted ranking loss
        loss = focal_weight * ranking_violation
        
        return loss.mean()


class TripletLossWithHardNegatives(nn.Module):
    """
    Triplet loss with hard negative mining
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, scores, labels):
        """
        Args:
            scores: [batch_size, num_candidates]
            labels: [batch_size]
        """
        # Positive scores (correct entities)
        positive_scores = scores.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # Find hardest negatives
        negative_scores = scores.clone()
        negative_scores.scatter_(1, labels.unsqueeze(1), float('-inf'))
        hard_negative_scores, _ = negative_scores.max(dim=1)
        
        # Triplet loss
        loss = F.relu(self.margin + hard_negative_scores - positive_scores)
        
        return loss.mean()


class CombinedRankingLoss(nn.Module):
    """
    Combines focal and triplet losses for robust training
    """
    def __init__(self, margin=1.0, gamma=2.0, lambda_focal=0.7, lambda_triplet=0.3):
        super().__init__()
        self.focal_loss = FocalRankingLoss(gamma=gamma, margin=margin)
        self.triplet_loss = TripletLossWithHardNegatives(margin=margin)
        self.lambda_focal = lambda_focal
        self.lambda_triplet = lambda_triplet
    
    def forward(self, scores, labels):
        focal = self.focal_loss(scores, labels)
        triplet = self.triplet_loss(scores, labels)
        
        total_loss = self.lambda_focal * focal + self.lambda_triplet * triplet
        
        return total_loss


class LabelSmoothingRankingLoss(nn.Module):
    """
    Cross-entropy with label smoothing for noisy labels
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, scores, labels):
        """
        Args:
            scores: [batch_size, num_candidates]
            labels: [batch_size]
        """
        num_classes = scores.size(-1)
        
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / num_classes
        
        # Create smoothed labels
        one_hot = torch.zeros_like(scores).scatter_(1, labels.unsqueeze(1), 1)
        smooth_labels = one_hot * confidence + smooth_value
        
        log_probs = F.log_softmax(scores, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        
        return loss


def get_loss_after_removing_gt_from_neg(scores, label_input, label_input_gt):
    """
    Helper function to handle the exclude_gt case
    Removes ground truth from negatives before computing loss
    
    Args:
        scores: [batch_size, num_candidates] - original scores
        label_input: candidate labels
        label_input_gt: ground truth labels to exclude
    
    Returns:
        new_scores: scores after removing gt
        new_labels: adjusted label indices
    """
    batch_size = scores.size(0)
    num_candidates = scores.size(1)
    
    # This should match your existing implementation
    # Placeholder - adjust based on your actual implementation
    new_scores_list = []
    new_labels_list = []
    
    for i in range(batch_size):
        # Find and remove gt from candidates
        gt_mask = label_input[i] != label_input_gt[i]
        batch_scores = scores[i][gt_mask]
        
        # Find new label position (should be where original correct was)
        # Adjust this based on your logic
        new_label = 0  # Placeholder
        
        new_scores_list.append(batch_scores)
        new_labels_list.append(new_label)
    
    # Stack back
    new_scores = torch.stack(new_scores_list)
    new_labels = torch.tensor(new_labels_list, device=scores.device)
    
    return new_scores, new_labels