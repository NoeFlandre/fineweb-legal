"""Legal Quality Classifier model with LoRA on Gemma Embedding 300M."""

import logging
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class LegalClassifier(nn.Module):
    """Gemma Embedding 300M with LoRA adapters and classification head.
    
    Architecture:
    - Base: google/gemma-embedding-300m (308M params, 2048 token context)
    - LoRA: Applied to all linear projections (q, k, v, o_proj)
    - Head: Linear(768, 6) for 6-class classification (scores 0-5)
    
    Training uses CrossEntropyLoss, inference uses weighted probability average.
    """
    
    MODEL_NAME = "google/embeddinggemma-300m"
    EMBEDDING_DIM = 768
    NUM_CLASSES = 6
    MAX_LENGTH = 2048
    
    def __init__(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_lora: bool = True,
    ):
        """Initialize the classifier.
        
        Args:
            lora_r: LoRA rank (default 16)
            lora_alpha: LoRA scaling factor (default 32)
            lora_dropout: LoRA dropout rate (default 0.1)
            use_lora: Whether to apply LoRA adapters (set False for inference-only)
        """
        super().__init__()
        
        logger.info(f"Loading base model: {self.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.encoder = AutoModel.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        if use_lora:
            logger.info(f"Applying LoRA: r={lora_r}, alpha={lora_alpha}")
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.EMBEDDING_DIM, self.NUM_CLASSES),
        )
        
        # Move classifier to same dtype as encoder
        self.classifier = self.classifier.to(torch.bfloat16)
        
        logger.info("LegalClassifier initialized")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: encode text and classify.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Logits tensor [batch, 6] for 6-class classification
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Use CLS token or mean pooling depending on model
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # Mean pooling over non-padded tokens
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.dtype)
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        
        # Ensure consistent dtype with classifier
        pooled = pooled.to(torch.bfloat16)
        
        # Classification
        logits = self.classifier(pooled)
        return logits
    
    def predict_scores(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to continuous scores via weighted probability average.
        
        Formula: score = sum(P(class=i) * i) for i in 0..5
        
        Args:
            logits: Raw logits [batch, 6]
            
        Returns:
            Continuous scores [batch] in range [0.0, 5.0]
        """
        probs = torch.softmax(logits, dim=-1)
        weights = torch.arange(
            self.NUM_CLASSES,
            device=logits.device,
            dtype=logits.dtype,
        )
        scores = (probs * weights).sum(dim=-1)
        return scores
    
    def get_trainable_params(self) -> int:
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())
    
    def save_pretrained(self, save_path: str) -> None:
        """Save model checkpoint including LoRA weights and classifier head."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save LoRA weights
        self.encoder.save_pretrained(save_path)
        
        # Save classifier head separately
        torch.save(
            self.classifier.state_dict(),
            os.path.join(save_path, "classifier_head.pt"),
        )
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str) -> "LegalClassifier":
        """Load model from checkpoint."""
        import os
        from peft import PeftModel
        
        # Create base model without LoRA
        model = cls(use_lora=False)
        
        # Load LoRA weights
        model.encoder = PeftModel.from_pretrained(
            model.encoder,
            load_path,
            torch_dtype=torch.bfloat16,
        )
        
        # Load classifier head
        head_path = os.path.join(load_path, "classifier_head.pt")
        if os.path.exists(head_path):
            model.classifier.load_state_dict(torch.load(head_path, weights_only=True))
        
        logger.info(f"Model loaded from {load_path}")
        return model
