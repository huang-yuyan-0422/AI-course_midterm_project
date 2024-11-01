import tiktoken
import torch
from typing import List, Dict, Optional

class EnhancedTokenizer:
    def __init__(self):
        # basic tokenizer
        self.base_tokenizer = tiktoken.get_encoding("gpt2")
        
         # Special token starting IDs (after GPT-2 vocabulary size 50257)
        self.special_tokens = {
            '<start>': 50257,  # Sentence start
            '<end>': 50258,    # Sentence end
            '<punct>': 50259,  # Punctuation
        }
        
        # Cache common tokens
        self.period_token = self.base_tokenizer.encode('.')[0]
        self.space_token = self.base_tokenizer.encode(' ')[0]
        
        self.vocab_size = 50257 + len(self.special_tokens)
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if not add_special_tokens:
            return self.base_tokenizer.encode(text)
            
        sentences = text.split('.')
        encoded = []
        
        for sent in sentences:
            if sent.strip():
                # Base tokens
                tokens = self.base_tokenizer.encode(sent.strip())
                # Add special tokens
                if add_special_tokens:
                    tokens = [self.special_tokens['<start>']] + tokens + [self.special_tokens['<end>']]
                encoded.extend(tokens)
                
                # Add period if not the last sentence
                if sent != sentences[-1]:
                    encoded.append(self.period_token)
        
        return encoded
    
    def decode(self, tokens: List[int]) -> str:
        # Separate special tokens and normal tokens
        normal_tokens = []
        for token in tokens:
            if token < 50257: 
                normal_tokens.append(token)
                
        # Decode normal tokens
        text = self.base_tokenizer.decode(normal_tokens)
        return text
    
    # Get vocabulary size
    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    # Determine if it is a special token
    def is_special_token(self, token: int) -> bool:
        return token >= 50257
    
    # Get mask for special tokens
    def get_special_tokens_mask(self, tokens: List[int]) -> List[int]:
        return [1 if self.is_special_token(token) else 0 for token in tokens]

# Function to create a tokenizer instance
def create_tokenizer() -> EnhancedTokenizer:
    return EnhancedTokenizer()