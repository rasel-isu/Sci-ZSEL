import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

logger = logging.getLogger(__name__)


class PseudoLabelRefiner:
    """
    Refines pseudo-labels using the model's own predictions
    Integrates with your existing dataloader
    """
    def __init__(self, reranker, device, params, context_length):
        self.reranker = reranker
        self.device = device
        self.params = params
        self.context_length = context_length
    
    def refine_dataloader_labels(
        self, 
        train_dataloader, 
        confidence_threshold=0.7,
        strategy='adaptive',
        ranking_loss_fn=None
    ):
        """
        Refine labels in the training dataloader
        
        Args:
            train_dataloader: Your existing training dataloader
            confidence_threshold: Minimum confidence to relabel
            strategy: 'aggressive', 'conservative', or 'adaptive'
        
        Returns:
            Number of labels refined
        """
        self.reranker.eval()
        
        refined_count = 0
        total_count = 0
        
        logger.info(f"Refining pseudo-labels with strategy: {strategy}, threshold: {confidence_threshold}")
        
        context_input_list = []
        new_label_input_list = []
        sample_id_list = []

        with torch.no_grad():
            # Iterate through dataset
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                context_input = batch[0] 
                label_input = batch[1]
                sample_id = batch[2]
                
                loss, batch_scores = self.reranker(context_input, label_input, self.context_length)
                B, C = batch_scores.shape
                for i in range(B):
                    sample_id_list.append(sample_id[i].item())
                    current_label = label_input[i].item()
                    # if current_label !=0:
                    #     print(0)
                    scores = batch_scores[i]
                    probs = F.softmax(scores, dim=-1)
                    top5_probs, top5_indices = torch.topk(probs, k=len(probs))

                    should_relabel = False
                    new_label = top5_indices[0].item()
                    
                    if strategy == 'aggressive':
                        # Always use top-1 if confidence > threshold
                        if top5_probs[0].item() > confidence_threshold:
                            should_relabel = True # Top-1 is already at position 0
                    
                    elif strategy == 'conservative':
                        # Only relabel if very confident (>0.9)
                        if top5_probs[0].item() > 0.9:
                            should_relabel = True
                    
                    elif strategy == 'adaptive':
                        # Multi-condition relabeling
                        top1_conf = top5_probs[0].item()
                        
                        if top1_conf > 0.8:
                            should_relabel = True
                        elif top1_conf > confidence_threshold:
                            should_relabel = True
                    
                    if should_relabel and new_label != current_label:
                        context_input_list.append(context_input[i])
                        new_label_input_list.append(new_label)
                        refined_count += 1
                    else:
                        context_input_list.append(context_input[i])
                        new_label_input_list.append(current_label)
                    
                    total_count += 1
        
        logger.info(f"Refined {refined_count}/{total_count} labels ({100*refined_count/total_count:.2f}%)")

        
        context_tensor = torch.stack(context_input_list, dim=0)  # [N, L]
        label_tensor   = torch.tensor(new_label_input_list)  # [N]
        sample_id_tensor   = torch.tensor(sample_id_list)
        

        refined_tensor_data = TensorDataset(context_tensor, label_tensor, sample_id_tensor)
        refined_sampler = RandomSampler(refined_tensor_data)
        refined_dataloader = DataLoader(
            refined_tensor_data, 
            sampler=refined_sampler, 
            batch_size= self.params["train_batch_size"]
        )

        return refined_count, refined_dataloader