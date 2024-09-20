import torch
from torch.nn import functional as F
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel

@MODEL_REGISTRY.register()
class MambaCSRModel(SRModel):
    """MambaCSR model for compressed image super-resolution."""
    # test by partitioning
    def test(self):
        _, C, h, w = self.lq.size()
        split_token_h = h // 192 +1   # number of horizontal cut sections 
        split_token_w = w // 192 +1   # number of vertical cut sections
        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()
        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition
        # overlapping
        shave_h = split_h // 4  
        shave_w = split_w // 4
        scale = self.opt.get('scale', 1)
        ral = H // split_h
        row = W // split_w
        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i*split_h, (i+1)*split_h+shave_h)
                elif i == ral - 1:
                    top = slice(i*split_h-shave_h, (i+1)*split_h)
                else:
                    top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j*split_w, (j+1)*split_w)
                elif j == 0:
                    left = slice(j*split_w, (j+1)*split_w+shave_w)
                elif j == row - 1:
                    left = slice(j*split_w-shave_w, (j+1)*split_w)
                else:
                    left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                temp = (top, left)
                slices.append(temp)
        img_chops = []  # list of partitions
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    padding_size = self.padding_size
                    _, _, h_old, w_old = chop.size()
                    h_pad = (h_old // padding_size + 1) * padding_size - h_old
                    w_pad = (w_old // padding_size + 1) * padding_size - w_old
                    chop = torch.cat([chop, torch.flip(chop, [2])], 2)[:, :, :h_old + h_pad, :]
                    chop = torch.cat([chop, torch.flip(chop, [3])], 3)[:, :, :, :w_old + w_pad]
                    out = self.net_g_ema(chop)  # image processing of each partition
                    out = out[..., :h_old * 4, :w_old * 4]
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                        if j == 0:
                            _left = slice(0, split_w*scale)
                        else:
                            _left = slice(shave_w*scale, (shave_w+split_w)*scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
        else:
            self.net_g.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    padding_size = self.padding_size
                    _, _, h_old, w_old = chop.size()
                    h_pad = (h_old // padding_size + 1) * padding_size - h_old
                    w_pad = (w_old // padding_size + 1) * padding_size - w_old
                    chop = torch.cat([chop, torch.flip(chop, [2])], 2)[:, :, :h_old + h_pad, :]
                    chop = torch.cat([chop, torch.flip(chop, [3])], 3)[:, :, :, :w_old + w_pad]
                    out = self.net_g(chop)  # image processing of each partition
                    out = out[..., :h_old * 4, :w_old * 4]
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                        if j == 0:
                            _left = slice(0, split_w * scale)
                        else:
                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
            self.net_g.train()
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]


