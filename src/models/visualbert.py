"""VisualBERT model implementation."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    VisualBertForQuestionAnswering,
    VisualBertTokenizer,
    VisualBertModel
)


class VisualBERTModel(nn.Module):
    """VisualBERT model for Visual Question Answering.
    
    This implementation provides a clean interface for VisualBERT
    with proper configuration and forward pass handling.
    """
    
    def __init__(
        self,
        pretrained_model: str = "uclanlp/visualbert-nlvr2",
        num_labels: int = 2,
        freeze_vision: bool = False,
        freeze_text: bool = False,
        **kwargs
    ):
        """Initialize VisualBERT model.
        
        Args:
            pretrained_model: Name of pretrained model
            num_labels: Number of output labels
            freeze_vision: Whether to freeze vision encoder
            freeze_text: Whether to freeze text encoder
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.pretrained_model = pretrained_model
        self.num_labels = num_labels
        self.freeze_vision = freeze_vision
        self.freeze_text = freeze_text
        
        # Load pretrained model
        self.model = VisualBertForQuestionAnswering.from_pretrained(
            pretrained_model,
            num_labels=num_labels
        )
        
        # Freeze components if requested
        if freeze_vision:
            self._freeze_vision_encoder()
        
        if freeze_text:
            self._freeze_text_encoder()
    
    def _freeze_vision_encoder(self) -> None:
        """Freeze vision encoder parameters."""
        for param in self.model.visual_bert.embeddings.visual_projection.parameters():
            param.requires_grad = False
    
    def _freeze_text_encoder(self) -> None:
        """Freeze text encoder parameters."""
        for param in self.model.visual_bert.embeddings.word_embeddings.parameters():
            param.requires_grad = False
        for param in self.model.visual_bert.embeddings.position_embeddings.parameters():
            param.requires_grad = False
        for param in self.model.visual_bert.embeddings.token_type_embeddings.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        images: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        answer_input_ids: Optional[torch.Tensor] = None,
        answer_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            question_input_ids: Question token IDs [batch_size, seq_len]
            question_attention_mask: Question attention mask [batch_size, seq_len]
            answer_input_ids: Answer token IDs [batch_size, seq_len] (optional)
            answer_attention_mask: Answer attention mask [batch_size, seq_len] (optional)
            **kwargs: Additional arguments
            
        Returns:
            Dict: Model outputs including logits and loss
        """
        # Prepare inputs for VisualBERT
        inputs = {
            "input_ids": question_input_ids,
            "attention_mask": question_attention_mask,
            "visual_embeds": images,
            "visual_attention_mask": torch.ones(
                images.size(0), images.size(1), 
                device=images.device, dtype=torch.long
            )
        }
        
        # Add answer inputs if provided (for training)
        if answer_input_ids is not None:
            inputs["labels"] = answer_input_ids
        
        # Forward pass
        outputs = self.model(**inputs)
        
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else None,
            "hidden_states": outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            "attentions": outputs.attentions if hasattr(outputs, 'attentions') else None
        }
    
    def generate_answer(
        self,
        images: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        max_length: int = 20,
        **kwargs
    ) -> torch.Tensor:
        """Generate answer for given image and question.
        
        Args:
            images: Input images
            question_input_ids: Question token IDs
            question_attention_mask: Question attention mask
            max_length: Maximum answer length
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Generated answer token IDs
        """
        self.eval()
        
        with torch.no_grad():
            # Get model outputs
            outputs = self.forward(
                images=images,
                question_input_ids=question_input_ids,
                question_attention_mask=question_attention_mask
            )
            
            # For binary classification, return the predicted class
            if self.num_labels == 2:
                predicted_class = torch.argmax(outputs["logits"], dim=-1)
                return predicted_class
            
            # For generation tasks, you would implement proper generation logic here
            # This is a simplified version
            return outputs["logits"]
    
    def get_attention_weights(
        self,
        images: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Get attention weights for visualization.
        
        Args:
            images: Input images
            question_input_ids: Question token IDs
            question_attention_mask: Question attention mask
            layer_idx: Specific layer index (None for all layers)
            
        Returns:
            torch.Tensor: Attention weights
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(
                images=images,
                question_input_ids=question_input_ids,
                question_attention_mask=question_attention_mask
            )
            
            if outputs["attentions"] is not None:
                if layer_idx is not None:
                    return outputs["attentions"][layer_idx]
                else:
                    return outputs["attentions"]
            else:
                return None
