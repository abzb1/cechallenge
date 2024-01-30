import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

class model_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = 32000
        self.n_layers = 10

        self.tok_embeddings = VocabEmbedding(
            32000, 8192,
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id))

        self.freqs_cis = precompute_freqs_cis(
            8192 // 64, 4096 * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        return h

class model_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 10

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id+10))

        self.freqs_cis = precompute_freqs_cis(
            8192 // 64, 4096 * 2
        )

    @torch.inference_mode()
    def forward(self, h: torch.Tensor, start_pos: int):
        _bsz, seqlen, _ = h.shape
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=h.device
            )

            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=h.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        return h

class model_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 10

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id+20))

        self.freqs_cis = precompute_freqs_cis(
            8192 // 64, 4096 * 2
        )

    @torch.inference_mode()
    def forward(self, h: torch.Tensor, start_pos: int):
        _bsz, seqlen, _ = h.shape
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=h.device
            )

            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=h.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        return h
    
class model_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 10

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id+30))

        self.freqs_cis = precompute_freqs_cis(
            8192 // 64, 4096 * 2
        )

    @torch.inference_mode()
    def forward(self, h: torch.Tensor, start_pos: int):
        _bsz, seqlen, _ = h.shape
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=h.device
            )

            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=h.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        return h
    
class model_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 10

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id+40))

        self.freqs_cis = precompute_freqs_cis(
            8192 // 64, 4096 * 2
        )

    @torch.inference_mode()
    def forward(self, h: torch.Tensor, start_pos: int):
        _bsz, seqlen, _ = h.shape
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=h.device
            )

            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=h.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        return h
    
class model_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 10

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id+50))

        self.freqs_cis = precompute_freqs_cis(
            8192 // 64, 4096 * 2
        )

    @torch.inference_mode()
    def forward(self, h: torch.Tensor, start_pos: int):
        _bsz, seqlen, _ = h.shape
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=h.device
            )

            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=h.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        return h
    
class model_6(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layers = 10

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id+60))

        self.freqs_cis = precompute_freqs_cis(
            8192 // 64, 4096 * 2
        )

    @torch.inference_mode()
    def forward(self, h: torch.Tensor, start_pos: int):
        _bsz, seqlen, _ = h.shape
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=h.device
            )

            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=h.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        return h

class model_7(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = 32000
        self.n_layers = 10

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id+70))

        self.norm = RMSNorm(8192, eps=1e-6)
        self.output = noParallelLinear(
            8192, 32000, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            8192 // 64, 4096 * 2
        )

    @torch.inference_mode()
    def forward(self, h: torch.Tensor, start_pos: int):
        _bsz, seqlen, _ = h.shape
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=h.device
            )

            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=h.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        # *********************
        # output = self.output(h[:, -1, :])  # only compute last logits
        # *********************
        output = self.output(h)
        return output.float()

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()
        self.n_heads = 64
        self.dim = 8192
        self.head_dim = 8192 // 64
        self.attention = Attention()
        self.feed_forward = FeedForward(
            dim=8192, hidden_dim=4*8192, multiple_of=4096
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(8192, eps=1e-5)
        self.ffn_norm = RMSNorm(8192, eps=1e-5)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class VocabEmbedding(torch.nn.Module):
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> None:
        super(VocabEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = None
        # Divide the weight matrix along the vocaburaly dimension.
        
        self.num_embeddings = 32000

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore

        # Get the embeddings.
        output = F.embedding(
            input_,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        return output

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_local_heads = 64
        self.n_local_kv_heads = 8
        self.head_dim = 8192 // 64
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        self.wq = noParallelLinear(
            8192,
            64 * self.head_dim,
            bias=False,
        )
        self.wk = noParallelLinear(
            8192,
            1024,
            bias=False,
        )
        self.wv = noParallelLinear(
            8192,
            1024,
            bias=False,
        )
        self.wo = noParallelLinear(
            64 * self.head_dim,
            8192,
            bias=False,
        )

        self.cache_k = torch.zeros(
            (4, 1024, self.n_local_kv_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (4, 1024, self.n_local_kv_heads, self.head_dim)
        ).cuda()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)


        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = 28672

        self.w1 = noParallelLinear(
            dim, hidden_dim, bias=False
        )
        self.w2 = noParallelLinear(
            hidden_dim, dim, bias=False
        )
        self.w3 = noParallelLinear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class noParallelLinear(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ) -> None:
        super(noParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore

        # Matrix multiply.
        output = F.linear(input_, self.weight, self.bias)

        return output

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )