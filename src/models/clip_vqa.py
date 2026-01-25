"""CLIP-based VQA model implementation."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer


class CLIPVQAModel(nn.Module):
    """CLIP-based Visual Question Answering model.
    
    This implementation uses CLIP for vision-language understanding
    and adds a classification head for VQA tasks.
    """
    
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        num_labels: int = 2,
        freeze_clip: bool = False,
        hidden_size: int = 512,
        dropout: float = 0.1,
        **kwargs
    ):
        """Initialize CLIP VQA model.
        
        Args:
            clip_model_name: Name of CLIP model
            num_labels: Number of output labels
            freeze_clip: Whether to freeze CLIP parameters
            hidden_size: Hidden size for classification head
            dropout: Dropout rate
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.clip_model_name = clip_model_name
        self.num_labels = num_labels
        self.freeze_clip = freeze_clip
        self.hidden_size = hidden_size
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        
        # Get CLIP dimensions
        self.clip_dim = self.clip_model.config.projection_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.clip_dim * 2, hidden_size),  # *2 for concatenated features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # Freeze CLIP if requested
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using CLIP vision encoder.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Image embeddings
        """
        return self.clip_model.get_image_features(pixel_values=images)
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text using CLIP text encoder.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            torch.Tensor: Text embeddings
        """
        return self.clip_model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
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
            images: Input images
            question_input_ids: Question token IDs
            question_attention_mask: Question attention mask
            answer_input_ids: Answer token IDs (optional)
            answer_attention_mask: Answer attention mask (optional)
            **kwargs: Additional arguments
            
        Returns:
            Dict: Model outputs
        """
        # Encode images and questions
        image_features = self.encode_image(images)
        question_features = self.encode_text(question_input_ids, question_attention_mask)
        
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=-1)
        question_features = F.normalize(question_features, p=2, dim=-1)
        
        # Concatenate features
        combined_features = torch.cat([image_features, question_features], dim=-1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        # Compute loss if labels provided
        loss = None
        if answer_input_ids is not None:
            # For binary classification, convert answer to labels
            if self.num_labels == 2:
                # Simple mapping: "yes" -> 1, "no" -> 0, others -> 0
                labels = torch.zeros(answer_input_ids.size(0), dtype=torch.long, device=answer_input_ids.device)
                # This is a simplified approach - in practice, you'd have proper label mapping
                loss = F.cross_entropy(logits, labels)
        
        return {
            "logits": logits,
            "loss": loss,
            "image_features": image_features,
            "question_features": question_features,
            "combined_features": combined_features
        }
    
    def generate_answer(
        self,
        images: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Generate answer for given image and question.
        
        Args:
            images: Input images
            question_input_ids: Question token IDs
            question_attention_mask: Question attention mask
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Predicted answer class
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(
                images=images,
                question_input_ids=question_input_ids,
                question_attention_mask=question_attention_mask
            )
            
            predicted_class = torch.argmax(outputs["logits"], dim=-1)
            return predicted_class
    
    def compute_similarity(
        self,
        images: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute image-question similarity.
        
        Args:
            images: Input images
            question_input_ids: Question token IDs
            question_attention_mask: Question attention mask
            
        Returns:
            torch.Tensor: Similarity scores
        """
        self.eval()
        
        with torch.no_grad():
            image_features = self.encode_image(images)
            question_features = self.encode_text(question_input_ids, question_attention_mask)
            
            # Normalize features
            image_features = F.normalize(image_features, p=2, dim=-1)
            question_features = F.normalize(question_features, p=2, dim=-1)
            
            # Compute cosine similarity
            similarity = torch.sum(image_features * question_features, dim=-1)
            
            return similarity
