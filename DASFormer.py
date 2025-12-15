
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from __future__ import annotations

"""
Implementation of DASFormer for Φ‑OTDR event classification.

This script defines all the building blocks necessary to reproduce
the architecture described in the paper “DASFormer: A Long
Sensing Sequence Classification and Recognition Model for
Phase‑Sensitive Optical Time Domain Reflectometers”.  It includes
hierarchical slicing, contextual positional encoding, a reduction
and adaptive multi‑head attention mechanism, a scale‑wise
feature fusion module, and the overall dual‑tower DASFormer
model.  A simple training loop is provided at the bottom to
illustrate how to instantiate the model and train it on dummy
data.  You can replace the dummy dataset with your own
pre‑processed Φ‑OTDR matrices and labels.

Note: This implementation aims to remain faithful to the paper’s
descriptions, but it simplifies certain aspects (e.g. DeepNorm) to
keep the code concise and easy to understand.  Hyperparameters
should be tuned according to your own dataset.
"""
from time import time
from tqdm import tqdm
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse
from mydataset import MyDataset


class ContextPositionalEncoding(nn.Module):
    """Contextual positional encoding using a depth‑wise 1D convolution.

    Instead of using static sinusoidal encodings, the DASFormer
    employs a learnable convolution that extracts local context and
    injects positional information into the sequence.  A depth‑wise
    convolution (groups=dim) ensures that each feature channel is
    processed independently.
    """

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] -> conv over L
        # permute to [B, C, L] for conv1d
        x_conv = self.conv(x.permute(0, 2, 1))
        # permute back to [B, L, C]
        x_conv = x_conv.permute(0, 2, 1)
        # add positional information
        return x + x_conv


class AdaptiveAttention(nn.Module):
    """Temporal/Spatial Reduction Adaptive Multi‑Head Attention.

    This module implements the T/SRAA mechanism described in the
    paper.  It first reduces the length of the key and value sequences
    by a factor `reduction` using a pooling operation.  The usual
    multi‑head attention is then applied between the full query
    sequence and the reduced keys/values.  To make the attention
    adaptive, a learnable gating parameter λ is computed from the
    mean of the queries and used to weight the attention outputs.
    """

    def __init__(self, dim: int, num_heads: int, reduction: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.reduction = max(1, reduction)
        # Multi‑head attention with batch_first=True for convenience
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        # Linear layer to compute λ for each head
        self.lambda_dense = nn.Linear(dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
    
        # --- FIX: ne pas pooler avec un kernel > L ---
        r = min(self.reduction, L)          # réduction effective
        if r > 1:
            kv = F.max_pool1d(
                x.permute(0, 2, 1),         # [B, C, L]
                kernel_size=r,
                stride=r
            ).permute(0, 2, 1)              # [B, L//r, C]
        else:
            kv = x                           # pas de réduction si L==1
    
        # Queries / Keys / Values
        Q = x
        K = kv
        V = kv
    
        attn_out, _ = self.attn(Q, K, V)
    
        # λ adaptatif par tête
        q_mean = Q.mean(dim=1)               # [B, C]
        lam = torch.sigmoid(self.lambda_dense(q_mean))  # [B, num_heads]
    
        head_dim = C // self.num_heads
        attn_out = attn_out.view(B, L, self.num_heads, head_dim)
        attn_out = attn_out * lam.view(B, 1, self.num_heads, 1)
        attn_out = attn_out.view(B, L, C)
        return attn_out



class FeedForward(nn.Module):
    """Position‑wise feed‑forward network with GELU activation."""

    def __init__(self, dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """A single DASFormer encoder layer.

    It consists of contextual positional encoding, adaptive attention,
    feed‑forward network, and simple residual connections with
    LayerNorm.  For brevity, we approximate DeepNorm by standard
    LayerNorm here.  In practice you can scale residuals using the
    factors described in the paper.
    """

    def __init__(self, dim: int, num_heads: int, reduction: int, ffn_dim: int, dropout: float = 0.6) -> None:
        super().__init__()
        # Contextual positional encoding (no learnable scale on residual)
        self.pos_enc = ContextPositionalEncoding(dim)
        # Adaptive attention and feed‑forward network
        self.attn = AdaptiveAttention(dim, num_heads, reduction)
        self.ffn = FeedForward(dim, ffn_dim)
        # LayerNorm layers for pre‑norm configuration
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # Dropout to improve stability and avoid overfitting
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a Transformer encoder layer in pre‑norm configuration.

        Pre‑norm (applying LayerNorm before attention and feed‑forward) has
        empirically been shown to improve numerical stability and prevent
        exploding activations in deep networks.  Dropout is applied on
        residual branches to further mitigate NaN issues.
        """
        # Add contextual positional encoding
        x = self.pos_enc(x)
        # Attention with pre‑norm and residual
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        x = x + self.drop(attn_out)
        # Feed‑forward with pre‑norm and residual
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.drop(ffn_out)
        return x


import torch
import torch.nn.functional as F

def hierarchical_slice(x, slice_size: int, stride: int = None):
    """
    x: [B, L, C]  (séquence, canaux)
    slice_size (Si): taille de fenêtre à agréger
    stride (Zi): pas de glissement; si None -> fenêtrage non-chevauchant (= slice_size)
    Retour: out [B, L_eff, slice_size * C], L_eff
    """
    if stride is None:
        stride = slice_size  # fenêtrage non-chevauchant par défaut

    B, L, C = x.shape

    # On évite tout problème de contiguïté en utilisant unfold sur la dim séquence.
    # On passe temporairement en [B, C, L] pour unfold sur L.
    x_u = x.permute(0, 2, 1).contiguous().unfold(dimension=2, size=slice_size, step=stride)
    # x_u: [B, C, L_eff, slice_size]

    # Revenir à [B, L_eff, slice_size*C] de manière contiguë et sûre
    x_u = x_u.permute(0, 2, 3, 1).contiguous().reshape(B, -1, slice_size * C)
    L_eff = x_u.shape[1]
    return x_u, L_eff



class ScaleFeatureFusion(nn.Module):
    """Scale‑wise feature fusion (SFF) module.

    It concatenates feature maps from multiple scales and applies a
    squeeze‑and‑excitation style re‑weighting across channels【323071144943756†L472-L501】.
    """

    def __init__(self, channel_sum: int, reduction: int = 4) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channel_sum, channel_sum // reduction)
        self.fc2 = nn.Linear(channel_sum // reduction, channel_sum)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Assume all feature tensors have shape [B, L_i, C_i]
        # Align them to the maximum length via interpolation
        max_len = max(f.shape[1] for f in features)
        aligned = []
        for f in features:
            B, L_i, C_i = f.shape
            if L_i != max_len:
                # interpolate along the sequence dimension
                f = F.interpolate(
                    f.permute(0, 2, 1), size=max_len, mode="linear", align_corners=False
                ).permute(0, 2, 1)
            aligned.append(f)
        # concatenate along channel dimension
        fused = torch.cat(aligned, dim=-1)  # [B, L, sum(C_i)]
        # squeeze: global pooling -> [B, sum(C_i)]
        squeeze = self.pool(fused.permute(0, 2, 1)).squeeze(-1)
        # excitation: two FC layers and sigmoid
        excitation = torch.sigmoid(self.fc2(F.relu(self.fc1(squeeze))))
        # reweight channels
        fused = fused * excitation.unsqueeze(1)
        return fused

from typing import Optional, List, Tuple

class DASFormerTower(nn.Module):
    """One tower of the DASFormer model (temporal or spatial).

    It performs hierarchical slicing across multiple stages, applies
    Transformer encoder layers with adaptive attention on each
    stage, fuses the multi‑scale features via SFF, and finally
    aggregates the features via max pooling over the sequence
    dimension to obtain a global representation.
    """

    def __init__(
        self,
        slice_sizes: List[int],
        hidden_dims: List[int],
        num_heads: List[int],
        reductions: List[int],
        ffn_expansion: int = 4,
        input_dim: Optional[int] = None,
    ) -> None:
        """Initialise a DASFormer tower.

        Args:
            slice_sizes: list of window sizes S_i for each stage.
            hidden_dims: list of hidden dimensions C_i for each stage.
            num_heads: number of attention heads H_i for each stage.
            reductions: reduction ratio γ_i for each stage in the adaptive attention.
            ffn_expansion: expansion factor for the feed‑forward networks (default 4).
            input_dim: the number of channels in the input sequence.  If
                provided, it is used to compute the correct input dimension
                for the first projection.  When None, the first projection
                assumes the same dimension as hidden_dims[0], which may
                lead to dimension mismatches.
        """
        super().__init__()
        assert (
            len(slice_sizes)
            == len(hidden_dims)
            == len(num_heads)
            == len(reductions)
        ), "Hyperparameter lists must have the same length"
        self.num_stages = len(slice_sizes)
        self.slice_sizes = slice_sizes
        # Create encoder layers for each stage
        self.encoders = nn.ModuleList(
            [
                EncoderLayer(
                    dim=hidden_dims[i],
                    num_heads=num_heads[i],
                    reduction=reductions[i],
                    ffn_dim=hidden_dims[i] * ffn_expansion,
                )
                for i in range(self.num_stages)
            ]
        )
        # Build projection layers with correct input dimensions.
        self.projections = nn.ModuleList()
        prev_dim = input_dim if input_dim is not None else hidden_dims[0]
        for i in range(self.num_stages):
            in_dim = slice_sizes[i] * prev_dim
            out_dim = hidden_dims[i]
            self.projections.append(nn.Linear(in_dim, out_dim))
            prev_dim = out_dim  # update for next stage
        # Fusion module
        self.fusion = ScaleFeatureFusion(sum(hidden_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C_in]
        features = []
        out = x
        for i in range(self.num_stages):
            # Hierarchical slice: group slice_size elements and project to hidden_dim
            out, new_len = hierarchical_slice(out, self.slice_sizes[i])
            # Project to hidden dimension
            out = self.projections[i](out)
            # Apply encoder layer (positional encoding + attention + FFN)
            out = self.encoders[i](out)
            features.append(out)
        # Fuse multi‑scale features
        fused = self.fusion(features)
        # Aggregate along the sequence dimension with adaptive max pooling
        # to produce a fixed‑size vector per sample
        pooled = F.adaptive_max_pool1d(fused.permute(0, 2, 1), output_size=1).squeeze(-1)
        return pooled  # [B, sum(hidden_dims)]


class DASFormer(nn.Module):
    """Dual‑tower DASFormer model.

    It instantiates two towers: one operating along the temporal
    dimension and the other along the spatial dimension (by
    transposing the input).  Their global representations are
    concatenated and passed through a final classifier.
    """

    def __init__(
        self,
        num_classes: int,
        # Temporal tower hyperparameters
        t_slice_sizes: List[int],
        t_hidden_dims: List[int],
        t_num_heads: List[int],
        t_reductions: List[int],
        # Spatial tower hyperparameters
        s_slice_sizes: List[int],
        s_hidden_dims: List[int],
        s_num_heads: List[int],
        s_reductions: List[int],
        # Input dimensions for towers
        time_input_dim: int,
        space_input_dim: int,
    ) -> None:
        super().__init__()
        # Temporal tower processes [B, L, S] where S=time_input_dim
        self.time_tower = DASFormerTower(
            t_slice_sizes, t_hidden_dims, t_num_heads, t_reductions, input_dim=time_input_dim
        )
        # Spatial tower processes [B, S, L] where L=space_input_dim
        self.space_tower = DASFormerTower(
            s_slice_sizes, s_hidden_dims, s_num_heads, s_reductions, input_dim=space_input_dim
        )
        # Final classifier: input dimension is sum(t_hidden_dims) + sum(s_hidden_dims)
        self.classifier = nn.Linear(
            sum(t_hidden_dims) + sum(s_hidden_dims), num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, S]
        t_repr = self.time_tower(x)
        # Transpose pour la tour spatiale -> rendre contigu pour éviter tout souci de stride
        s_repr = self.space_tower(x.transpose(1, 2).contiguous())
        out = torch.cat([t_repr, s_repr], dim=-1)
        return self.classifier(out)


# class RandomPhiOTDRDataset(Dataset):
#     """A dummy dataset generating random Φ‑OTDR sequences and labels.

#     This class is provided solely for demonstration purposes.  In a
#     real use case, replace it with a dataset that loads your own
#     pre‑processed Φ‑OTDR matrices (e.g. reading from disk) and labels.
#     Each sample is a tensor of shape [time_length, spatial_nodes].
#     """

#     def __init__(self, num_samples: int, time_length: int, spatial_nodes: int, num_classes: int) -> None:
#         self.num_samples = num_samples
#         self.time_length = time_length
#         self.spatial_nodes = spatial_nodes
#         self.num_classes = num_classes

#     def __len__(self) -> int:
#         return self.num_samples

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
#         # Generate random sequence and label
#         x = torch.randn(self.time_length, self.spatial_nodes)
#         label = torch.randint(0, self.num_classes, (1,)).item()
#         return x, label


def train_model(args=None) -> None:
    """
    Entraîne DASFormer sur des données réelles via MyDataset (format comme ton CNN).
    MyDataset doit renvoyer un dict: {'data': tensor[L,S], 'label': int}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    train_dataset = MyDataset(args.root,  args.txtpath,  transform=None)
    test_dataset  = MyDataset(args.root2, args.txtpath2, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    # Déduire (L, S) depuis un batch
    sample = next(iter(train_loader))
    sample_x = sample['data']            # [B, L, S]
    _, L, S = sample_x.shape

    # --- Hyperparamètres modèle ---
    num_classes = args.num_classes

    # Tour temporelle (L potentiellement grand)
    t_slice_sizes = [10, 5, 5]
    t_hidden_dims = [64, 128, 256]
    t_num_heads   = [4, 4, 8]
    t_reductions  = [2, 2, 1]    # dernier à 1 pour éviter pooling nul

    # Tour spatiale (S plus petit)
    s_slice_sizes = [2, 2, 2]
    s_hidden_dims = [32, 64, 128]
    s_num_heads   = [2, 4, 4]
    s_reductions  = [2, 2, 1]

    # --- Modèle ---
    model = DASFormer(
        num_classes=num_classes,
        t_slice_sizes=t_slice_sizes, t_hidden_dims=t_hidden_dims,
        t_num_heads=t_num_heads,     t_reductions=t_reductions,
        s_slice_sizes=s_slice_sizes, s_hidden_dims=s_hidden_dims,
        s_num_heads=s_num_heads,     s_reductions=s_reductions,
        time_input_dim=S,            # nb de nœuds spatiaux
        space_input_dim=L            # longueur temporelle
    ).to(device)

    # --- Optim & loss ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # --- Entraînement ---
    num_epochs = args.epochs
    batches_per_epoch = max(1, len(train_loader))
    for epoch in range(num_epochs):
        model.train()
        epoch_start = time()
        running_loss = 0.0
        seen = 0

        pbar = tqdm(enumerate(train_loader),
                    total=batches_per_epoch,
                    desc=f"Epoch {epoch+1}/{num_epochs}",
                    unit="batch")

        for batch_idx, batch in pbar:
            inputs = batch['data'].float().to(device)   # [B, L, S]
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)                     # [B, C]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bsz = inputs.size(0)
            running_loss += loss.item() * bsz
            seen += bsz
            avg_loss = running_loss / max(1, seen)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{avg_loss:.4f}"})

        epoch_time = time() - epoch_start
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs} | mean loss: {epoch_loss:.4f} | time: {epoch_time:.1f}s")

    # (optionnel) sauvegarde des poids
    torch.save(model.state_dict(), "dasformer_state_dict.pth")
    print("✅ Entraînement terminé. Poids sauvegardés dans dasformer_state_dict.pth")




# if __name__ == "__main__":
#     # Run the demonstration training when executing the script directly
#     train_model()