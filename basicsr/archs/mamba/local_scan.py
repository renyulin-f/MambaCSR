# The Dual-Interleaved, Cross-scale scanning methods. 
import math
import torch
import torch.nn.functional as F

def pad_tensor(x, w, H, W):
    if H % w == 0 and W % w == 0:
        return x, (H, W)
    B, C = x.shape[:2]
    if len(x.shape) == 3:
        x = x.view(B, C, H, W)
    
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    newH, newW = Hg * w, Wg * w
    x = F.pad(x, (0, newW - W, 0, newH - H))

    return x, (newH, newW)


"""PyTorch code for local scan and local reverse"""


def local_scan(x, w=7, H=14, W=14, flip=False, column_first=False):
    """Local windowed scan in LocalMamba
    Input: 
        x: [B, L, C]
        H, W: original width and height before padding
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, C, L]
    """
    B, L, C = x.shape
    x = x.view(B, H, W, C)
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    if H % w != 0 or W % w != 0:
        newH, newW = Hg * w, Wg * w
        x = F.pad(x, (0, 0, 0, newW - W, 0, newH - H))
    if column_first:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 5, 3, 1, 4, 2).reshape(B, C, -1)
    else:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, -1)
    if flip:
        x = x.flip([-1])
    return x


def local_scan_bchw(x, w=8, H=64, W=64, flip=False, column_first=False):
    """Local windowed scan in LocalMamba
    Input: 
        x: [B, C, H, W]
        H, W: original width and height before padding
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, C, L]
    """
    B, C, _, _ = x.shape
    x = x.view(B, C, H, W)
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    if H % w != 0 or W % w != 0:
        newH, newW = Hg * w, Wg * w
        x = F.pad(x, (0, newW - W, 0, newH - H))
    if column_first:
        x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C, -1)
    else:
        x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1) #(B,360,4096)
    if flip:
        x = x.flip([-1])
    return x


def local_reverse(x, w=8, H=64, W=64, flip=False, column_first=False):
    """Local windowed scan in LocalMamba
    Input: 
        x: [B, C, L]
        H, W: original width and height before padding
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, C, L]
    """
    B, C, L = x.shape
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    if flip:
        x = x.flip([-1])
    if H % w != 0 or W % w != 0:
        if column_first:
            x = x.view(B, C, Wg, Hg, w, w).permute(0, 1, 3, 5, 2, 4).reshape(B, C, Hg * w, Wg * w)
        else:
            x = x.view(B, C, Hg, Wg, w, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, Hg * w, Wg * w)
        x = x[:, :, :H, :W].reshape(B, C, -1)
    else:
        if column_first:
            x = x.view(B, C, Wg, Hg, w, w).permute(0, 1, 3, 5, 2, 4).reshape(B, C, L)
        else:
            x = x.view(B, C, Hg, Wg, w, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, L)
    return x


def nested_local_scan_bchw(x, w1=8, w2=4,H=64, W=64, flip=False, column_first=False):
    """
    Local windowed scan with nested windows in LocalMamba.

    Input:
        x: [B, C, H, W]
        w1: Outer window size
        w2: Inner window size
        H, W: original width and height before padding
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, C, L]
    """
    B, C, _, _ = x.shape
    # print(w1,w2)
    x = x.view(B, C, H, W)
    Hg, Wg = math.ceil(H / w1), math.ceil(W / w1)
    if column_first:
        x = x.view(B, C, Hg, w1, Wg, w1).permute(0, 1, 4, 2, 5, 3).reshape(B, C, -1)
    else:
        x = x.view(B, C, Hg, w1, Wg, w1).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)
    # x = x.view(B, C, Hg * Wg, w1 // w2, w1 // w2, w2, w2).permute(0, 1, 2, 3, 5, 4, 6).reshape(B, C, -1)
    Hg_inner, Wg_inner = math.ceil(w1 / w2), math.ceil(w1 / w2)
    x = x.view(B, C, Hg * Wg, Hg_inner, w2, Wg_inner, w2)
    x = x.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, C, -1)
    if flip:
        x = x.flip(-1)
    return x


def reverse_nested_local_scan_bchw(x, w1=8, w2=4, H=64, W=64, flip=False, column_first=False):
    """
    Reverse operation of nested_local_scan_bchw.
    
    Input:
        x: [B, C, L]
        w1: Outer window size
        w2: Inner window size
        H, W: original height and width
        column_first: column-wise scan first
    Return: [B, C, H, W]
    """
    B, C, L = x.shape
    Hg, Wg = math.ceil(H / w1), math.ceil(W / w1)
    Hg_inner, Wg_inner = math.ceil(w1 / w2), math.ceil(w1 / w2)
    if flip:
        x = x.flip(-1)
    # Reshape to recover the inner windows
    x = x.view(B, C, Hg * Wg, Hg_inner, Wg_inner, w2, w2)
    x = x.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, C, Hg, w1, Wg, w1)    
    # Reshape to recover the outer windows and original structure
    if column_first:
        x = x.view(B, C, Hg, Wg, w1, w1).permute(0, 1, 3, 5, 2, 4).reshape(B, C, L)
    else:
        x = x.view(B, C, Hg, Wg, w1, w1).permute(0, 1, 2, 4, 3, 5).reshape(B, C, L)
    return x


def merge_tensors(x1, x2):
    B, C, L = x1.shape
    assert x2.shape == (B, C, 4 * L), "Shape of x2 should be (B, C, 4 * L)"
    x2_reshaped = x2.view(B, C, L, 4)
    x1_expanded = x1.unsqueeze(-1)
    result = torch.cat((x1_expanded, x2_reshaped), dim=-1)
    result = result.view(B, C, -1)
    return result


def cross_scale_scan(x, x_down, w1=8, flip=False, column_first=False):
    B, C, H, W = x.shape
    w2 = w1 // 2
    seq_high = nested_local_scan_bchw(x, w1=w1, w2=2, H=H, W=W, flip=flip, column_first=column_first) #Position-Aligned Scanning Methods.
    seq_down = local_scan_bchw(x_down, w=w2, H=x_down.shape[2], W=x_down.shape[3], flip=flip, column_first=column_first)
    combined = merge_tensors(seq_down, seq_high)
    return combined


def cross_scale_reverse(seq, H, W, w1=8, flip=False, column_first=False):
    """
    Convert the output sequence of cross_scale_scan back to the original high-resolution and medium-resolution images.

    Parameters:
        seq: Input tensor with shape (B, C, L)
        H: Height of the high-resolution image
        W: Width of the high-resolution image
        w1: Window size of the high-resolution image (default 8)
        w2_size: Window size of the medium-resolution image (default 4)
        flip: Whether to reverse the scan direction
        column_first: Whether to scan column-first

    Returns:
        x1: High-resolution image with shape (B, C, H, W)
        x2: Medium-resolution image with shape (B, C, h2, w2)
    """
    B, C, L = seq.shape
    reshaped_seq = seq.view(B, C, L//5, 5)
    sliced_seq = reshaped_seq[:, :, :, 1:]  # 去掉每组的第一个元素
    seq_high = sliced_seq.reshape(B, C, -1).contiguous() 
    x2 = seq_high
    x1 = reverse_nested_local_scan_bchw(x2, w1=w1, w2=2, H=H, W=W, flip=flip, column_first=column_first)
    return x1
