# model/two_tower_enhanced.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTower(nn.Module):
    def __init__(self, config, vocab_size_tag, vocab_size_genre):
        super().__init__()
        
        # Store config for later use
        self.config = config
        
        # Embedding layers with proper initialization
        self.user_id_embedding = nn.Embedding(
            config["user_embedding_num"], 
            config["user_embedding_dim"],
            padding_idx=0  # Reserve 0 for unknown users
        )
        
        self.movie_id_embedding = nn.Embedding(
            config["item_embedding_num"], 
            config["item_embedding_dim"],
            padding_idx=0  # Reserve 0 for unknown movies
        )
        
        self.tag_embedding = nn.Embedding(
            vocab_size_tag, 
            config["user_embedding_dim"],
            padding_idx=0
        )
        
        self.genre_embedding = nn.Embedding(
            vocab_size_genre, 
            config["item_embedding_dim"],
            padding_idx=0
        )

        # Dense layer configurations
        self.user_dense = [2 * config["user_embedding_dim"]] + config["user_dense"]
        self.item_dense = [2 * config["item_embedding_dim"]] + config["item_dense"]

        # Build towers
        self.user_tower = self._make_mlp(self.user_dense, config["activation"])
        self.item_tower = self._make_mlp(self.item_dense, config["activation"])
        
        # Initialize embeddings properly
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings with proper scaling"""
        nn.init.normal_(self.user_id_embedding.weight, std=0.1)
        nn.init.normal_(self.movie_id_embedding.weight, std=0.1)
        nn.init.normal_(self.tag_embedding.weight, std=0.1)
        nn.init.normal_(self.genre_embedding.weight, std=0.1)
        
        # Zero out padding embeddings
        with torch.no_grad():
            self.user_id_embedding.weight[0].fill_(0)
            self.movie_id_embedding.weight[0].fill_(0)

    def _make_mlp(self, sizes, activation):
        """Create MLP with proper initialization and dropout"""
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            
            # Initialize weights
            nn.init.xavier_uniform_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
            
            # Add activation (except for last layer)
            if i < len(sizes) - 2:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                
                # Add dropout for regularization
                layers.append(nn.Dropout(0.1))
        
        return nn.Sequential(*layers)

    def user_embedding_vec(self, user_id, tag_input_ids):
        """Generate user embeddings with proper handling"""
        # User ID embedding
        u_id_emb = self.user_id_embedding(user_id)
        
        # Tag embedding with proper shape handling
        if tag_input_ids.dim() == 1:
            tag_input_ids = tag_input_ids.unsqueeze(0)
        
        # Average pooling over sequence dimension
        u_tag_emb = self.tag_embedding(tag_input_ids)
        
        # Create attention mask to ignore padding tokens
        attention_mask = (tag_input_ids != 0).float()
        if attention_mask.sum(dim=-1, keepdim=True).min() == 0:
            # If all tokens are padding, use uniform weights
            attention_mask = torch.ones_like(attention_mask)
        
        # Weighted average instead of simple mean
        attention_weights = attention_mask / attention_mask.sum(dim=-1, keepdim=True)
        u_tag_emb = (u_tag_emb * attention_weights.unsqueeze(-1)).sum(dim=-2)
        
        # Concatenate embeddings
        user_input = torch.cat([u_id_emb, u_tag_emb], dim=-1)
        
        # Pass through user tower
        return self.user_tower(user_input)

    def item_embedding_vec(self, movie_id, genre_input_ids):
        """Generate item embeddings with proper handling"""
        # Movie ID embedding
        i_id_emb = self.movie_id_embedding(movie_id)
        
        # Genre embedding with proper shape handling
        if genre_input_ids.dim() == 1:
            genre_input_ids = genre_input_ids.unsqueeze(0)
        
        # Average pooling over sequence dimension
        i_genre_emb = self.genre_embedding(genre_input_ids)
        
        # Create attention mask
        attention_mask = (genre_input_ids != 0).float()
        if attention_mask.sum(dim=-1, keepdim=True).min() == 0:
            attention_mask = torch.ones_like(attention_mask)
        
        # Weighted average
        attention_weights = attention_mask / attention_mask.sum(dim=-1, keepdim=True)
        i_genre_emb = (i_genre_emb * attention_weights.unsqueeze(-1)).sum(dim=-2)
        
        # Concatenate embeddings
        item_input = torch.cat([i_id_emb, i_genre_emb], dim=-1)
        
        # Pass through item tower
        return self.item_tower(item_input)

    def forward(self, user_id, tag_input_ids, movie_id, genre_input_ids):
        """Forward pass with improved shape handling"""
        # Get embeddings
        user_vec = self.user_embedding_vec(user_id, tag_input_ids)
        item_vec = self.item_embedding_vec(movie_id, genre_input_ids)

        # Ensure proper shapes for dot product
        if user_vec.dim() == 1:
            user_vec = user_vec.unsqueeze(0)
        if item_vec.dim() == 1:
            item_vec = item_vec.unsqueeze(0)

        # Compute similarity score
        score = (user_vec * item_vec).sum(dim=1)
        
        # Handle single sample case
        if score.size(0) == 1 and user_id.dim() == 0:
            score = score.squeeze(0)
            
        return score

    def get_user_embeddings(self, user_ids, tag_input_ids):
        """Get user embeddings for similarity computation"""
        return self.user_embedding_vec(user_ids, tag_input_ids)
    
    def get_item_embeddings(self, movie_ids, genre_input_ids):
        """Get item embeddings for similarity computation"""
        return self.item_embedding_vec(movie_ids, genre_input_ids)
    
    def predict_batch(self, user_ids, tag_input_ids, movie_ids, genre_input_ids):
        """Batch prediction for efficiency"""
        with torch.no_grad():
            return self.forward(user_ids, tag_input_ids, movie_ids, genre_input_ids)
    
    def get_model_info(self):
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "user_embedding_size": self.config["user_embedding_num"],
            "item_embedding_size": self.config["item_embedding_num"],
            "user_dense_layers": self.config["user_dense"],
            "item_dense_layers": self.config["item_dense"]
        }