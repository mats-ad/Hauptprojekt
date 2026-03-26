# Evolutionäres Constrain basiertes NAS

- Dataset: EuroSAT
- Model: timm ViT-Small pretrained (ImageNet)
- Search space: depth, heads, patch_size (prepared), SPT/LSA flags (prepared),
                finetune strategy + optimizer hyperparams
- NAS: population -> elite selection -> mutation -> generations
- Constraints: soft penalty (val_acc - lambda*params - mu*time)

### Imports


```python
import time
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
```

    c:\Users\matsa\miniconda3\envs\dl_cuda\lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
```

    2.5.1+cu121
    12.1
    True



```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
```

    Using device: cuda



```python
def make_data_loaders(
    root="./data",
    image_size=224,
    batch_size=16,
    num_workers=2,
    seed=42,
):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])
    
    ds = datasets.EuroSAT(root=root, download=True, transform=tfm)
    
    n = len(ds)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=g
    )
    
    use_pin = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin,
    persistent_workers=(num_workers > 0))
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin
    )
    
    return train_loader, val_loader, test_loader, len(ds.classes)

train_loader, val_loader, test_loader, num_classes = make_data_loaders()
print("Classes:", num_classes)
```

    Downloading https://cas-bridge.xethub.hf.co/xet-bridge-us/63d93fa519aa7b8ed272fc38/37dfb877f3bf87d402835aba93060b4d8a53e02d48394614a99c1831d540d2f0?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20260223%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260223T102653Z&X-Amz-Expires=3600&X-Amz-Signature=365a1531446aa637a7ea413042a36d154d24916d73ad338120f782966357ba67&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=public&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27EuroSAT.zip%3B+filename%3D%22EuroSAT.zip%22%3B&response-content-type=application%2Fzip&x-id=GetObject&Expires=1771846013&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3MTg0NjAxM319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82M2Q5M2ZhNTE5YWE3YjhlZDI3MmZjMzgvMzdkZmI4NzdmM2JmODdkNDAyODM1YWJhOTMwNjBiNGQ4YTUzZTAyZDQ4Mzk0NjE0YTk5YzE4MzFkNTQwZDJmMCoifV19&Signature=L0qeWEIXfAfL%7Eagda3Uwt0xWlBg5D5s9rfOxjBJR0j3dw1IiuEiKh3QTQnQ4BDaA3gywgVKIeAx-3-QyoKanu4mhclzGIUDjhCaB49cV-Wi%7EgVukVE9GYKbwx7QCEtc9-AZj7RsSbA9KkMTSiNlThwv2bTotifJNLGqBUkpInV1wJdbqj-pLSX0jSzOBJfKCJ7tAkDBavZN6MYx%7EsnbRqVsPbDd6HVi0hwayaSFaKz9dQndJ9-9VWGE-GH7XQH-LpFnIUgpeQi6Y-ZaO0jPzlZBZdfJosZZoimbPi8ZNj6Xwmvq10Lphe1xn5c5SkuibqkVgddpXxk46t9RffU%7E4yQ__&Key-Pair-Id=K2L8F4GPSG1IFC to ./data\eurosat\EuroSAT.zip


     19%|█▊        | 17.6M/94.3M [00:00<00:02, 33.3MB/s]



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[4], line 45
         39     test_loader = DataLoader(
         40         test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin
         41     )
         43     return train_loader, val_loader, test_loader, len(ds.classes)
    ---> 45 train_loader, val_loader, test_loader, num_classes = make_data_loaders()
         46 print("Classes:", num_classes)


    Cell In[4], line 17, in make_data_loaders(root, image_size, batch_size, num_workers, seed)
          1 def make_data_loaders(
          2     root="./data",
          3     image_size=224,
       (...)
          6     seed=42,
          7 ):
          8     tfm = transforms.Compose([
          9         transforms.Resize((image_size, image_size)),
         10         transforms.ToTensor(),
       (...)
         14         ),
         15     ])
    ---> 17     ds = datasets.EuroSAT(root=root, download=True, transform=tfm)
         19     n = len(ds)
         20     n_train = int(0.7 * n)


    File c:\Users\matsa\miniconda3\envs\dl_cuda\lib\site-packages\torchvision\datasets\eurosat.py:38, in EuroSAT.__init__(self, root, transform, target_transform, download)
         35 self._data_folder = os.path.join(self._base_folder, "2750")
         37 if download:
    ---> 38     self.download()
         40 if not self._check_exists():
         41     raise RuntimeError("Dataset not found. You can use download=True to download it")


    File c:\Users\matsa\miniconda3\envs\dl_cuda\lib\site-packages\torchvision\datasets\eurosat.py:58, in EuroSAT.download(self)
         55     return
         57 os.makedirs(self._base_folder, exist_ok=True)
    ---> 58 download_and_extract_archive(
         59     "https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip",
         60     download_root=self._base_folder,
         61     md5="c8fa014336c82ac7804f0398fcb19387",
         62 )


    File c:\Users\matsa\miniconda3\envs\dl_cuda\lib\site-packages\torchvision\datasets\utils.py:395, in download_and_extract_archive(url, download_root, extract_root, filename, md5, remove_finished)
        392 if not filename:
        393     filename = os.path.basename(url)
    --> 395 download_url(url, download_root, filename, md5)
        397 archive = os.path.join(download_root, filename)
        398 print(f"Extracting {archive} to {extract_root}")


    File c:\Users\matsa\miniconda3\envs\dl_cuda\lib\site-packages\torchvision\datasets\utils.py:132, in download_url(url, root, filename, md5, max_redirect_hops)
        130 try:
        131     print("Downloading " + url + " to " + fpath)
    --> 132     _urlretrieve(url, fpath)
        133 except (urllib.error.URLError, OSError) as e:  # type: ignore[attr-defined]
        134     if url[:5] == "https":


    File c:\Users\matsa\miniconda3\envs\dl_cuda\lib\site-packages\torchvision\datasets\utils.py:30, in _urlretrieve(url, filename, chunk_size)
         28 with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
         29     with open(filename, "wb") as fh, tqdm(total=response.length, unit="B", unit_scale=True) as pbar:
    ---> 30         while chunk := response.read(chunk_size):
         31             fh.write(chunk)
         32             pbar.update(len(chunk))


    File c:\Users\matsa\miniconda3\envs\dl_cuda\lib\http\client.py:466, in HTTPResponse.read(self, amt)
        463 if self.length is not None and amt > self.length:
        464     # clip the read to the "end of response"
        465     amt = self.length
    --> 466 s = self.fp.read(amt)
        467 if not s and amt:
        468     # Ideally, we would raise IncompleteRead if the content-length
        469     # wasn't satisfied, but it might break compatibility.
        470     self._close_conn()


    File c:\Users\matsa\miniconda3\envs\dl_cuda\lib\socket.py:717, in SocketIO.readinto(self, b)
        715 while True:
        716     try:
    --> 717         return self._sock.recv_into(b)
        718     except timeout:
        719         self._timeout_occurred = True


    File c:\Users\matsa\miniconda3\envs\dl_cuda\lib\ssl.py:1307, in SSLSocket.recv_into(self, buffer, nbytes, flags)
       1303     if flags != 0:
       1304         raise ValueError(
       1305           "non-zero flags not allowed in calls to recv_into() on %s" %
       1306           self.__class__)
    -> 1307     return self.read(nbytes, buffer)
       1308 else:
       1309     return super().recv_into(buffer, nbytes, flags)


    File c:\Users\matsa\miniconda3\envs\dl_cuda\lib\ssl.py:1163, in SSLSocket.read(self, len, buffer)
       1161 try:
       1162     if buffer is not None:
    -> 1163         return self._sslobj.read(len, buffer)
       1164     else:
       1165         return self._sslobj.read(len)


    KeyboardInterrupt: 


## Cifar100


```python
def make_cifar100_loaders(
    root="./data",
    image_size=224,
    batch_size=64,
    num_workers=4,
    seed=42,
):
    tfm_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    tfm_eval = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    full_train = datasets.CIFAR100(root=root, train=True, download=True, transform=tfm_train)
    test_ds = datasets.CIFAR100(root=root, train=False, download=True, transform=tfm_eval)

    # train/val split from training set
    n = len(full_train)
    n_train = int(0.9 * n)
    n_val = n - n_train

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=g)

    # IMPORTANT: val_ds currently uses tfm_train because it's a Subset of full_train.
    # Quick fix: wrap val subset with eval transform
    val_ds.dataset.transform = tfm_eval

    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader, 100

train_loader, val_loader, test_loader, num_classes = make_cifar100_loaders()
print(num_classes)
```

    Downloading https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz to ./data\cifar-100-python.tar.gz


    100%|██████████| 169M/169M [00:08<00:00, 19.2MB/s] 


    Extracting ./data\cifar-100-python.tar.gz to ./data
    Files already downloaded and verified
    100


# Shifted Patch Tokenization


```python
class SPTPatchEmbed(nn.Module):
    """
    Shifted Patch Tokenization (SPT) as Patch Embedding replacement for ViT.
    Produces patch tokens with increased locality inductive bias by concatenating
    shifted versions of the input image along channel dimension.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        vanilla=False,
        use_ln=True,
        eps=1e-6
    ):
        super().__init__()
        self.img_size = img_size if isinstance(img_size, int) else img_size[0]
        self.patch_size = patch_size if isinstance(patch_size, int) else patch_size[0]
        self.half_patch = self.patch_size // 2
        self.vanilla = vanilla
        self.use_ln = use_ln and (not vanilla)
        
        effective_in_chans = in_chans if vanilla else in_chans * 5
        
        self.proj = nn.Conv2d(
            effective_in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        self.norm = nn.LayerNorm(embed_dim, eps=eps) if self.use_ln else nn.Identity()
        
    def _crop_shift_pad(self, x, mode: str):
        """
        Create diagonally shifted image by cropping and padding.
        x: (B, C, H, W)
        """
        hp = self.half_patch
        B, C, H, W = x.shape

        if mode == "left-up":
            top_crop, left_crop = hp, hp
            top_pad, left_pad = 0, 0
        elif mode == "left-down":
            top_crop, left_crop = 0, hp
            top_pad, left_pad = hp, 0
        elif mode == "right-up":
            top_crop, left_crop = hp, 0
            top_pad, left_pad = 0, hp
        elif mode == "right-down":
            top_crop, left_crop = 0, 0
            top_pad, left_pad = hp, hp
        else:
            raise ValueError(mode)

        x_crop = x[:, :, 
                top_crop:top_crop + (H - hp),
                left_crop:left_crop + (W - hp)]

        pad_bottom = H - x_crop.shape[2] - top_pad
        pad_right = W - x_crop.shape[3] - left_pad

        x_pad = F.pad(
            x_crop,
            (left_pad, pad_right, top_pad, pad_bottom)
        )

        return x_pad
    
    def forward(self, x):
        if not self.vanilla:
            x = torch.cat(
                [
                x, 
                self._crop_shift_pad(x, "left-up"),
                self._crop_shift_pad(x, "left-down"),
                self._crop_shift_pad(x, "right-up"),
                self._crop_shift_pad(x, "right-down"),
                ],
                dim = 1
            )
        
        x = self.proj(x)
        
        x = x.flatten(2).transpose(1, 2)
        
        x = self.norm(x)
        
        return x
        
    
```


```python
def enable_spt(model, img_size=224, patch_size=16, vanilla=False):
    """
    Replace model.patch_embed with SPTPatchEmbed while keeping everything else identical.
    Assumes ViT-Small (embed_dim=384).
    """
    embed_dim = getattr(model, "embed_dim", 384)
    in_chans = 3
    
    model.patch_embed = SPTPatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        vanilla=vanilla,
        use_ln=True
    )
    
    return model
```

# Local Self-Attention


```python
class LSA_Attention(nn.Module):
    """
    Drop-in replacement for timm ViT Attention with:
      - diagonal mask (no self-token attention)
      - learnable per-head scale (temperature)
    Based on common SPT+LSA implementations.  [oai_citation:1‡GitHub](https://raw.githubusercontent.com/aanna0701/SPT_LSA_ViT/main/models/vit.py)
    """
    def __init__(self, attn_module: nn.Module):
        super().__init__()
        
        # Grab needed parts from timm's Attention
        self.num_heads = attn_module.num_heads
        self.head_dim = attn_module.head_dim
        self.attn_drop = attn_module.attn_drop
        self.proj = attn_module.proj
        self.proj_drop = attn_module.proj_drop
        
        # timm attention uses a single scale
        # LSA makes it learnable per head
        init_scale = getattr(attn_module, "scale", self.head_dim ** -0.5)
        self.scale = nn.Parameter(torch.ones(self.num_heads) * float(init_scale))
        
        # qkv projection
        self.qkv = attn_module.qkv
        
        # small cache for diagonal masks by token-length (optional speed)
        self._mask_cache = {}

    def _diag_mask(self, n: int, device):
        key = (n, device)
        if key in self._mask_cache:
            return self._mask_cache[key]

        m = torch.eye(n, device=device, dtype=torch.bool)
        self._mask_cache[key] = m
        return m

    def forward(self, x, attn_mask=None, **kwargs):
        # attn_mask is passed by some timm blocks; we can ignore it for LSA
        # or optionally apply it (see section 2).
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))
        attn = attn * self.scale.view(1, self.num_heads, 1, 1)

        # LSA: mask diagonal (no self-attention)
        diag = self._diag_mask(N, x.device)
        attn = attn.masked_fill(diag.view(1, 1, N, N), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out
        
```


```python
def enable_lsa(model):
    for blk in model.blocks:
        blk.attn = LSA_Attention(blk.attn)
    return model
```

# ViT-Small Implementation

Builds a ViT-Small using timm with variable depth and num_heads (if supported by your timm version).

Notes:
    - patch_size is prepared in cfg but not applied yet due to pretrained compatibility complexity.
    - use_spt/use_lsa are flags prepared for later module integration.


```python
def build_vit_from_cfg(num_classes, cfg):
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=True,
        num_classes=num_classes,
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
    )

    if cfg["use_spt"] == 1:
        model = enable_spt(model, img_size=224, patch_size=16, vanilla=False)
        
    if cfg["use_lsa"] == 1: model = enable_lsa(model)
    
    # TODO: Mixed Resolution Tokenization

    mode = cfg.get("finetune_mode", "last2")

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Train head always
    for p in model.head.parameters():
        p.requires_grad = True

    # Finetune modes
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "last2":
        for blk in model.blocks[-2:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in model.norm.parameters():
            p.requires_grad = True
    elif mode == "head":
        pass
    else:
        raise ValueError(mode)

    return model.to(device)

def count_params(model):
    return sum(p.numel() for p in model.parameters())
```


```python
def accuracy_top1(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def train_one_epoch(model, loader, optimizer):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n

@torch.no_grad()
def eval_one_epoch(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n

def quick_finetune(model, train_loader, val_loader, epochs=2, lr=3e-4, weight_decay=1e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    start = time.time()
    best_val_acc = 0.0

    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer)
        _, val_acc = eval_one_epoch(model, val_loader)
        best_val_acc = max(best_val_acc, val_acc)

    return best_val_acc, time.time() - start
```

# For Mixed Precision (PyTorch AMP)


```python
def accuracy_top1(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def train_one_epoch_amp(model, loader, optimizer, scaler):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n

@torch.no_grad()
def eval_one_epoch_amp(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda",enabled=(device == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n

def quick_finetune(model, train_loader, val_loader, epochs=20, lr=3e-4, weight_decay=1e-4, patience=None):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    start = time.time()
    best_val_acc = 0.0
    bad = 0

    for _ in range(epochs):
        train_one_epoch_amp(model, train_loader, optimizer, scaler)
        _, val_acc = eval_one_epoch_amp(model, val_loader)

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            bad = 0
        else:
            bad += 1
            if patience is not None and bad >= patience:
                break

    return best_val_acc, time.time() - start
```

## Suchraum (SEARCH_SPACE)

Definiert den diskreten Architektur- und Trainingsraum für das evolutionäre NAS.

**Architekturparameter:**
- `patch_size` – Patch-Zerlegung der Eingabebilder  
- `depth` – Anzahl der Transformer-Blöcke  
- `num_heads` – Anzahl Attention-Heads (muss 384 teilen)  
- `use_spt` – Aktivierung von Shifted Patch Tokenization  
- `use_lsa` – Aktivierung von Local Self-Attention  
- `finetune_mode` – Umfang des Fine-Tunings (`last2`, `full`)

**Optimierungsparameter:**
- `lr` – Lernrate  
- `weight_decay` – Regularisierung  
- `e` – Proxy-Trainingsepochen  

→ Der Suchraum kombiniert strukturelle Architekturentscheidungen mit Trainingshyperparametern.

---

## sample_config

- Erzeugt eine zufällige Konfiguration aus dem definierten Suchraum  
- Stellt sicher, dass `num_heads` die Embedding-Dimension (384) teilt  
- Dient zur Initialisierung der Startpopulation  

→ Implementiert die zufällige Startverteilung im evolutionären NAS.

---

## mutate_config

- Erzeugt ein Kind-Modell durch zufällige Mutation einzelner Parameter  
- Jeder Parameter wird mit Wahrscheinlichkeit `p_mut` verändert  
- Validiert erneut architektonische Konsistenz (`num_heads`-Constraint)

→ Realisiert den explorativen Schritt der Evolution durch kontrollierte Variation.


```python
SEARCH_SPACE = {
    "patch_size": [16],            # later: [16, 8] with proper pretrained adaptation
    "depth": [6, 8, 10, 12],
    "num_heads": [4, 6, 8],        # must divide embed_dim=384
    "use_spt": [0, 1],             
    "use_lsa": [0, 1],             
    "finetune_mode": ["last2", "full"],
    "lr": [1e-4, 3e-4, 1e-3],
    "weight_decay": [1e-4, 5e-4],
    "proxy_epochs": [5],
}

def sample_config(space=SEARCH_SPACE):
    cfg = {k: random.choice(v) for k, v in space.items()}

    # ensure heads divides embed_dim
    if 384 % cfg["num_heads"] != 0:
        valid = [h for h in space["num_heads"] if 384 % h == 0]
        cfg["num_heads"] = random.choice(valid)

    return cfg

def mutate_config(cfg, space=SEARCH_SPACE, p_mut=0.35):
    child = cfg.copy()
    for key, choices in space.items():
        if random.random() < p_mut:
            options = [c for c in choices if c != child[key]]
            child[key] = random.choice(options) if options else child[key]

    # ensure validity again
    if 384 % child["num_heads"] != 0:
        valid = [h for h in space["num_heads"] if 384 % h == 0]
        child["num_heads"] = random.choice(valid)

    return child
```

## Fitness-Funktion

- Bewertet jede Architektur mittels kombinierter Zielfunktion  
- Ziel: hohe Validation Accuracy bei geringer Modellgröße und Trainingszeit  
- Formel:  
  `fitness = val_acc − λ_params · params − λ_time · time`

→ Implementiert ein **constraint-basiertes NAS**, das Genauigkeit und Effizienz gemeinsam optimiert.

---

## normalize_cfg

- Ergänzt fehlende Hyperparameter durch Default-Werte  
- Setzt Trainingsparameter (Proxy-Epochen, LR, Weight Decay)  
- Definiert Architekturparameter (Depth, Heads, Patch Size, SPT, LSA)  
- Prüft architektonische Gültigkeit (z. B. Teilbarkeit der Embedding-Dimension durch `num_heads`)

→ Stellt sicher, dass nur valide und konsistente ViT-Konfigurationen trainiert werden.

---

## evaluate_cfg

- Baut das Vision-Transformer-Modell aus der Konfiguration  
- Optional: übernimmt kompatible Gewichte vom Elternmodell (Weight Inheritance)  
- Führt Proxy-Fine-Tuning durch  
- Berechnet Fitness-Wert  
- Gibt Architektur + Performance-Metriken zurück  

→ Zentrale Evaluationskomponente im evolutionären NAS-Prozess.


```python
def fitness(val_acc, params, time_sec, lam_params=1e-8, lam_time=0.02):
    return val_acc - lam_params * params - lam_time * (time_sec / 60.0)

def normalize_cfg(cfg):
    cfg = dict(cfg)  

    cfg.setdefault("proxy_epochs", 5)
    cfg.setdefault("lr", 3e-4)
    cfg.setdefault("weight_decay", 1e-4)
    cfg.setdefault("finetune_mode", "head")

    cfg.setdefault("patch_size", 16)
    cfg.setdefault("depth", 12)
    cfg.setdefault("num_heads", 6)
    cfg.setdefault("use_spt", 0)
    cfg.setdefault("use_lsa", 0)

    # validity
    if 384 % cfg["num_heads"] != 0:
        cfg["num_heads"] = 6

    return cfg

def evaluate_cfg(cfg, parent_state_dict=None):
    cfg = normalize_cfg(cfg)

    model = build_vit_from_cfg(num_classes, cfg)
    
    if parent_state_dict is not None:
        model.load_state_dict(parent_state_dict, strict=False)
    
    params = count_params(model)

    val_acc, secs = quick_finetune(
        model, train_loader, val_loader,
        epochs=cfg.get("proxy_epochs", 2),
        lr=cfg.get("lr", 3e-4),
        weight_decay=cfg.get("weight_decay", 1e-4)
    )

    fit = fitness(val_acc, params, secs)

    return {
        **cfg,
        "val_acc": val_acc,
        "time_sec": secs,
        "params": params,
        "fitness": fit
    }
```

# Evolutionäres NAS

Hierbei erbt jede Generation Gewichte von Elite Modellen

Neue Layer werden automatisch neu initialisiert

Fitness bleibt constraint-basiert

Proxy-Training bleibt erhalten

### Funktionsweise des evolutionären NAS

In diesem Projekt wird ein evolutionäres, constraint-basiertes Neural Architecture Search (NAS) eingesetzt, um geeignete Vision-Transformer-Architekturen für die EuroSAT-Klassifikation zu finden.

Ablauf
1.	Initialisierung
Eine Population zufälliger Architekturkonfigurationen wird aus dem definierten Suchraum (SEARCH_SPACE) erzeugt. \n


2.	Evaluation
Jede Architektur wird für wenige Epochen trainiert (Proxy-Training) und anhand einer Fitness-Funktion bewertet.
Die Fitness berücksichtigt:
- Validation Accuracy
- Parameteranzahl
- Trainingszeit

3.	Selektion (Elite Selection)
Die besten k Architekturen werden als „Eliten“ ausgewählt.
4.	Mutation
Neue Kandidaten entstehen durch Mutation architektureller Parameter wie:
- depth (Anzahl Transformer-Blöcke)
- num_heads
- use_spt
- use_lsa
- finetune_mode

5.	Iteration
Der Prozess wiederholt sich über mehrere Generationen.

⸻

### Weight Inheritance

Um die Rechenkosten zu reduzieren, wird Weight Inheritance eingesetzt:
- Kinderarchitekturen übernehmen – sofern möglich – die Gewichte des Elternmodells.
- Nicht kompatible Layer werden neu initialisiert (strict=False).

Dadurch wird Training beschleunigt, da Modelle nicht immer vollständig neu trainiert werden müssen.

⸻

### Wichtige Einschränkung bei Weight Inheritance

Weight Inheritance ist nur sinnvoll, wenn die zugrunde liegende Architektur strukturell kompatibel bleibt.

Problematisch ist es, Gewichte zu übernehmen, wenn sich zentrale Architekturkomponenten ändern, z. B.:
- Aktivierung/Deaktivierung von Shifted Patch Tokenization (SPT)
- Aktivierung/Deaktivierung von Local Self-Attention (LSA)
- Änderung der Embedding-Dimension oder Attention-Struktur

Solche Änderungen verändern die Feature-Verteilung oder Tensorformen erheblich.
Wird dennoch vererbt, kann dies:
- zu instabilem Training führen
- bestimmte Architekturvarianten systematisch benachteiligen
- die NAS-Suche verzerren

Daher sollte Weight Inheritance nur bei kompatiblen Mutationen (z. B. identischer Depth und Head-Anzahl) angewendet werden


```python
def evolutionary_nas(
    pop_size=10,
    generations=10,
    elite_k=3,
    p_mut=0.35,
    seed=42,
    log_csv_path="enas_cifar100.csv"
):
    random.seed(seed)

    population = [sample_config() for _ in range(pop_size)]
    all_results = []

    fieldnames = list(SEARCH_SPACE.keys()) + ["val_acc", "time_sec", "params", "fitness", "generation", "rank"]

    with open(log_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for gen in range(generations):
            print(f"\n=== Generation {gen} ===")
            gen_results = []

            for i, cfg in enumerate(population):
                parent_state = cfg.get("state_dict", None)
                
                res = evaluate_cfg(cfg)
                gen_results.append(res)
                print(
                    f"  [{i:02d}] "
                    f"val_acc={res['val_acc']:.4f} "
                    f"fitness={res['fitness']:.4f} "
                    f"params={res['params']/1e6:.1f}M "
                    f"time={res['time_sec']:.1f}s "
                    f"cfg(depth={res['depth']}, heads={res['num_heads']}, spt={res['use_spt']}, lsa={res['use_lsa']}, ft={res['finetune_mode']})"
                )

            gen_results = sorted(gen_results, key=lambda x: x["fitness"], reverse=True)

            for rank, res in enumerate(gen_results):
                row = {k: res[k] for k in SEARCH_SPACE.keys()}
                row.update({
                    "val_acc": res["val_acc"],
                    "time_sec": res["time_sec"],
                    "params": res["params"],
                    "fitness": res["fitness"],
                    "generation": gen,
                    "rank": rank
                })
                writer.writerow(row)

            all_results.extend([{**r, "generation": gen} for r in gen_results])

            elites = gen_results[:elite_k]

            # next population: mutate elites
            new_population = []
            while len(new_population) < pop_size:
                parent = random.choice(elites)
                parent_cfg = {k: parent[k] for k in SEARCH_SPACE.keys()}
                child_cfg = mutate_config(parent_cfg, p_mut=p_mut)
                
                if (
                    parent["depth"] == child_cfg["depth"] and
                    parent["num_heads"] == child_cfg["num_heads"] and
                    parent["use_spt"] == child_cfg["use_spt"] and
                    parent["use_lsa"] == child_cfg["use_lsa"]
                    and "state_dict" in parent
                ):
                    child_cfg["state_dict"] = parent["state_dict"]
                
                new_population.append(child_cfg)

            population = new_population

    return all_results
```


```python
results = evolutionary_nas(
    pop_size=10,
    generations=10,
    elite_k=3,
    p_mut=0.35,
    log_csv_path="enas_cifar100.csv"
)

print("\nDone. Results written to CSV")
```

    
    === Generation 0 ===


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [00] val_acc=0.3846 fitness=0.1965 params=12.2M time=197.2s cfg(depth=6, heads=8, spt=1, lsa=0, ft=last2)


    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [01] val_acc=0.6022 fitness=0.4310 params=11.1M time=182.0s cfg(depth=6, heads=4, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [02] val_acc=0.3320 fitness=0.1007 params=15.8M time=220.3s cfg(depth=8, heads=6, spt=1, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [03] val_acc=0.3270 fitness=0.0922 params=15.8M time=230.9s cfg(depth=8, heads=4, spt=1, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [04] val_acc=0.5348 fitness=0.1611 params=19.3M time=541.1s cfg(depth=10, heads=4, spt=1, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [05] val_acc=0.6182 fitness=0.4290 params=11.1M time=235.8s cfg(depth=6, heads=4, spt=0, lsa=1, ft=last2)
      [06] val_acc=0.4858 fitness=0.0513 params=22.9M time=617.0s cfg(depth=12, heads=8, spt=1, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [07] val_acc=0.6952 fitness=0.4141 params=14.6M time=405.1s cfg(depth=8, heads=8, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [08] val_acc=0.6410 fitness=0.3956 params=11.1M time=404.3s cfg(depth=6, heads=4, spt=0, lsa=1, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [09] val_acc=0.5412 fitness=0.1620 params=15.8M time=664.0s cfg(depth=8, heads=8, spt=1, lsa=1, ft=full)
    
    === Generation 1 ===


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [00] val_acc=0.5970 fitness=0.4260 params=11.1M time=181.3s cfg(depth=6, heads=4, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [01] val_acc=0.6210 fitness=0.3840 params=14.6M time=272.7s cfg(depth=8, heads=4, spt=0, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [02] val_acc=0.3862 fitness=0.1630 params=12.2M time=302.3s cfg(depth=6, heads=8, spt=1, lsa=1, ft=last2)
      [03] val_acc=0.2422 fitness=-0.0817 params=22.9M time=285.1s cfg(depth=12, heads=4, spt=1, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [04] val_acc=0.4384 fitness=0.0878 params=18.2M time=507.0s cfg(depth=10, heads=4, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [05] val_acc=0.5050 fitness=0.1360 params=19.3M time=527.0s cfg(depth=10, heads=8, spt=1, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [06] val_acc=0.6898 fitness=0.4714 params=11.1M time=323.6s cfg(depth=6, heads=4, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [07] val_acc=0.3856 fitness=0.1640 params=12.2M time=297.6s cfg(depth=6, heads=8, spt=1, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [08] val_acc=0.6808 fitness=0.5130 params=11.1M time=171.6s cfg(depth=6, heads=6, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [09] val_acc=0.6352 fitness=0.3987 params=14.6M time=271.4s cfg(depth=8, heads=4, spt=0, lsa=1, ft=last2)
    
    === Generation 2 ===


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [00] val_acc=0.5980 fitness=0.4107 params=11.1M time=230.0s cfg(depth=6, heads=4, spt=0, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [01] val_acc=0.3084 fitness=-0.0171 params=19.3M time=396.6s cfg(depth=10, heads=8, spt=1, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [02] val_acc=0.4104 fitness=0.1721 params=12.2M time=347.6s cfg(depth=6, heads=4, spt=1, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [03] val_acc=0.6022 fitness=0.4314 params=11.1M time=180.7s cfg(depth=6, heads=4, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [04] val_acc=0.6114 fitness=0.4400 params=11.1M time=182.5s cfg(depth=6, heads=4, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [05] val_acc=0.4928 fitness=0.2534 params=12.2M time=351.2s cfg(depth=6, heads=4, spt=1, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [06] val_acc=0.7344 fitness=0.4882 params=14.6M time=300.3s cfg(depth=8, heads=6, spt=0, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [07] val_acc=0.7048 fitness=0.4322 params=11.1M time=486.1s cfg(depth=6, heads=8, spt=0, lsa=1, ft=full)
      [08] val_acc=0.7070 fitness=0.2899 params=21.7M time=600.2s cfg(depth=12, heads=4, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [09] val_acc=0.7272 fitness=0.5151 params=14.6M time=198.3s cfg(depth=8, heads=6, spt=0, lsa=0, ft=last2)
    
    === Generation 3 ===


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [00] val_acc=0.7198 fitness=0.4727 params=14.6M time=303.1s cfg(depth=8, heads=6, spt=0, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [01] val_acc=0.7352 fitness=0.4152 params=14.6M time=521.7s cfg(depth=8, heads=4, spt=0, lsa=1, ft=full)
      [02] val_acc=0.7904 fitness=0.3852 params=21.7M time=564.5s cfg(depth=12, heads=6, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [03] val_acc=0.7192 fitness=0.5071 params=11.1M time=304.6s cfg(depth=6, heads=6, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [04] val_acc=0.3428 fitness=0.1540 params=12.2M time=199.2s cfg(depth=6, heads=4, spt=1, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [05] val_acc=0.6408 fitness=0.3959 params=11.1M time=403.0s cfg(depth=6, heads=4, spt=0, lsa=1, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [06] val_acc=0.7274 fitness=0.4809 params=14.6M time=301.4s cfg(depth=8, heads=6, spt=0, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [07] val_acc=0.3518 fitness=0.1634 params=12.2M time=198.2s cfg(depth=6, heads=4, spt=1, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [08] val_acc=0.6960 fitness=0.5277 params=11.1M time=173.1s cfg(depth=6, heads=6, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [09] val_acc=0.6712 fitness=0.5029 params=11.1M time=173.1s cfg(depth=6, heads=6, spt=0, lsa=0, ft=last2)
    
    === Generation 4 ===


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [00] val_acc=0.6808 fitness=0.4847 params=11.1M time=256.6s cfg(depth=6, heads=6, spt=0, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [01] val_acc=0.3710 fitness=-0.0931 params=19.3M time=812.1s cfg(depth=10, heads=8, spt=1, lsa=1, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [02] val_acc=0.5060 fitness=0.2923 params=11.1M time=309.5s cfg(depth=6, heads=8, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [03] val_acc=0.7168 fitness=0.5038 params=11.1M time=307.2s cfg(depth=6, heads=6, spt=0, lsa=0, ft=full)
      [04] val_acc=0.8640 fitness=0.4586 params=21.7M time=565.2s cfg(depth=12, heads=6, spt=0, lsa=0, ft=full)
      [05] val_acc=0.8114 fitness=0.4649 params=21.7M time=388.5s cfg(depth=12, heads=6, spt=0, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [06] val_acc=0.6554 fitness=0.2971 params=15.8M time=601.4s cfg(depth=8, heads=6, spt=1, lsa=1, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [07] val_acc=0.6468 fitness=0.4323 params=11.1M time=311.8s cfg(depth=6, heads=8, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [08] val_acc=0.7570 fitness=0.5452 params=14.6M time=197.4s cfg(depth=8, heads=6, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [09] val_acc=0.3442 fitness=0.1551 params=12.2M time=200.3s cfg(depth=6, heads=4, spt=1, lsa=0, ft=last2)
    
    === Generation 5 ===


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [00] val_acc=0.4806 fitness=0.2698 params=11.1M time=300.7s cfg(depth=6, heads=6, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [01] val_acc=0.3932 fitness=0.2082 params=12.2M time=187.9s cfg(depth=6, heads=6, spt=1, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [02] val_acc=0.7050 fitness=0.4235 params=14.6M time=406.3s cfg(depth=8, heads=8, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [03] val_acc=0.8132 fitness=0.5564 params=18.2M time=225.9s cfg(depth=10, heads=6, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [04] val_acc=0.7214 fitness=0.4646 params=11.1M time=438.7s cfg(depth=6, heads=6, spt=0, lsa=1, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [05] val_acc=0.6988 fitness=0.5313 params=11.1M time=170.8s cfg(depth=6, heads=6, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [06] val_acc=0.7286 fitness=0.3920 params=14.6M time=571.7s cfg(depth=8, heads=6, spt=0, lsa=1, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [07] val_acc=0.7772 fitness=0.3829 params=18.2M time=638.3s cfg(depth=10, heads=4, spt=0, lsa=1, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [08] val_acc=0.7014 fitness=0.5054 params=11.1M time=256.2s cfg(depth=6, heads=6, spt=0, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [09] val_acc=0.6138 fitness=0.4257 params=11.1M time=232.5s cfg(depth=6, heads=4, spt=0, lsa=1, ft=last2)
    
    === Generation 6 ===


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [00] val_acc=0.6990 fitness=0.5306 params=11.1M time=173.4s cfg(depth=6, heads=6, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [01] val_acc=0.2978 fitness=-0.0161 params=19.3M time=361.6s cfg(depth=10, heads=6, spt=1, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [02] val_acc=0.6752 fitness=0.4629 params=11.1M time=305.1s cfg(depth=6, heads=6, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [03] val_acc=0.8108 fitness=0.5538 params=18.2M time=226.3s cfg(depth=10, heads=6, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [04] val_acc=0.2814 fitness=0.0068 params=19.3M time=243.9s cfg(depth=10, heads=6, spt=1, lsa=0, ft=last2)
      [05] val_acc=0.8144 fitness=0.5141 params=21.7M time=249.7s cfg(depth=12, heads=6, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [06] val_acc=0.6460 fitness=0.4448 params=11.1M time=272.0s cfg(depth=6, heads=8, spt=0, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [07] val_acc=0.6982 fitness=0.5050 params=11.1M time=247.8s cfg(depth=6, heads=6, spt=0, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [08] val_acc=0.7280 fitness=0.5181 params=11.1M time=298.1s cfg(depth=6, heads=6, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [09] val_acc=0.4002 fitness=0.2159 params=12.2M time=185.7s cfg(depth=6, heads=6, spt=1, lsa=0, ft=last2)
    
    === Generation 7 ===
      [00] val_acc=0.2518 fitness=-0.0650 params=22.9M time=263.8s cfg(depth=12, heads=6, spt=1, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [01] val_acc=0.6924 fitness=0.5259 params=11.1M time=167.7s cfg(depth=6, heads=6, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [02] val_acc=0.5744 fitness=0.3160 params=18.2M time=230.7s cfg(depth=10, heads=4, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [03] val_acc=0.7938 fitness=0.4518 params=18.2M time=481.4s cfg(depth=10, heads=8, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [04] val_acc=0.7224 fitness=0.4687 params=11.1M time=429.4s cfg(depth=6, heads=6, spt=0, lsa=1, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [05] val_acc=0.7700 fitness=0.5257 params=14.6M time=294.8s cfg(depth=8, heads=6, spt=0, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [06] val_acc=0.3398 fitness=0.0768 params=15.8M time=315.3s cfg(depth=8, heads=6, spt=1, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [07] val_acc=0.7960 fitness=0.5025 params=18.2M time=335.9s cfg(depth=10, heads=6, spt=0, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [08] val_acc=0.8312 fitness=0.4190 params=18.2M time=692.0s cfg(depth=10, heads=6, spt=0, lsa=1, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [09] val_acc=0.5752 fitness=0.3163 params=18.2M time=232.0s cfg(depth=10, heads=4, spt=0, lsa=0, ft=last2)
    
    === Generation 8 ===


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [00] val_acc=0.7652 fitness=0.4879 params=14.6M time=393.6s cfg(depth=8, heads=8, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [01] val_acc=0.5900 fitness=0.2362 params=15.8M time=587.9s cfg(depth=8, heads=6, spt=1, lsa=1, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [02] val_acc=0.6934 fitness=0.5268 params=11.1M time=168.0s cfg(depth=6, heads=6, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [03] val_acc=0.7588 fitness=0.5483 params=14.6M time=193.2s cfg(depth=8, heads=6, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [04] val_acc=0.3406 fitness=0.1121 params=15.8M time=211.8s cfg(depth=8, heads=6, spt=1, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [05] val_acc=0.6962 fitness=0.5297 params=11.1M time=167.8s cfg(depth=6, heads=6, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [06] val_acc=0.3518 fitness=0.0799 params=15.8M time=342.2s cfg(depth=8, heads=8, spt=1, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [07] val_acc=0.7032 fitness=0.5366 params=11.1M time=168.0s cfg(depth=6, heads=6, spt=0, lsa=0, ft=last2)
      [08] val_acc=0.5214 fitness=0.1907 params=21.7M time=341.0s cfg(depth=12, heads=4, spt=0, lsa=1, ft=last2)
      [09] val_acc=0.3620 fitness=-0.1499 params=22.9M time=849.2s cfg(depth=12, heads=6, spt=1, lsa=1, ft=full)
    
    === Generation 9 ===


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [00] val_acc=0.6710 fitness=0.4613 params=11.1M time=297.4s cfg(depth=6, heads=6, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [01] val_acc=0.6634 fitness=0.4504 params=11.1M time=307.2s cfg(depth=6, heads=8, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [02] val_acc=0.6678 fitness=0.4583 params=11.1M time=296.9s cfg(depth=6, heads=6, spt=0, lsa=0, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [03] val_acc=0.8052 fitness=0.5507 params=18.2M time=218.8s cfg(depth=10, heads=6, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [04] val_acc=0.3850 fitness=0.1733 params=12.2M time=268.0s cfg(depth=6, heads=6, spt=1, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [05] val_acc=0.6988 fitness=0.5313 params=11.1M time=170.8s cfg(depth=6, heads=6, spt=0, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [06] val_acc=0.3864 fitness=0.1950 params=12.2M time=207.1s cfg(depth=6, heads=8, spt=1, lsa=0, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [07] val_acc=0.6688 fitness=0.4145 params=11.1M time=431.2s cfg(depth=6, heads=6, spt=0, lsa=1, ft=full)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [08] val_acc=0.3498 fitness=0.0877 params=15.8M time=312.7s cfg(depth=8, heads=6, spt=1, lsa=1, ft=last2)


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


      [09] val_acc=0.6170 fitness=0.4311 params=11.1M time=226.0s cfg(depth=6, heads=4, spt=0, lsa=1, ft=last2)
    
    Done. Results written to CSV



```python
enas_df = pd.read_csv("enas_cifar100.csv")
enas_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patch_size</th>
      <th>depth</th>
      <th>num_heads</th>
      <th>use_spt</th>
      <th>use_lsa</th>
      <th>finetune_mode</th>
      <th>lr</th>
      <th>weight_decay</th>
      <th>proxy_epochs</th>
      <th>val_acc</th>
      <th>time_sec</th>
      <th>params</th>
      <th>fitness</th>
      <th>generation</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0010</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.6022</td>
      <td>182.015210</td>
      <td>11057380</td>
      <td>0.430954</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.6182</td>
      <td>235.795175</td>
      <td>11057404</td>
      <td>0.429028</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>8</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>full</td>
      <td>0.0003</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.6952</td>
      <td>405.057583</td>
      <td>14606308</td>
      <td>0.414118</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>full</td>
      <td>0.0003</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.6410</td>
      <td>404.334749</td>
      <td>11057404</td>
      <td>0.395648</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>6</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.3846</td>
      <td>197.214382</td>
      <td>12237796</td>
      <td>0.196484</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
enas_df.sort_values("fitness", ascending=False).head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patch_size</th>
      <th>depth</th>
      <th>num_heads</th>
      <th>use_spt</th>
      <th>use_lsa</th>
      <th>finetune_mode</th>
      <th>lr</th>
      <th>weight_decay</th>
      <th>proxy_epochs</th>
      <th>val_acc</th>
      <th>time_sec</th>
      <th>params</th>
      <th>fitness</th>
      <th>generation</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.8132</td>
      <td>225.859128</td>
      <td>18155236</td>
      <td>0.556361</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>60</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.8108</td>
      <td>226.331010</td>
      <td>18155236</td>
      <td>0.553804</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>90</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.8052</td>
      <td>218.764402</td>
      <td>18155236</td>
      <td>0.550726</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>16</td>
      <td>8</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.7588</td>
      <td>193.229803</td>
      <td>14606308</td>
      <td>0.548327</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>16</td>
      <td>8</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.7570</td>
      <td>197.358562</td>
      <td>14606308</td>
      <td>0.545151</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>81</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.7032</td>
      <td>168.039326</td>
      <td>11057380</td>
      <td>0.536613</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>91</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.6988</td>
      <td>170.794423</td>
      <td>11057380</td>
      <td>0.531295</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.6988</td>
      <td>170.807199</td>
      <td>11057380</td>
      <td>0.531290</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>61</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.6990</td>
      <td>173.448973</td>
      <td>11057380</td>
      <td>0.530610</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>82</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.6962</td>
      <td>167.824127</td>
      <td>11057380</td>
      <td>0.529685</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>30</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.6960</td>
      <td>173.060886</td>
      <td>11057380</td>
      <td>0.527739</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.6934</td>
      <td>168.049723</td>
      <td>11057380</td>
      <td>0.526810</td>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <th>70</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.6924</td>
      <td>167.684403</td>
      <td>11057380</td>
      <td>0.525931</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>71</th>
      <td>16</td>
      <td>8</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.7700</td>
      <td>294.821623</td>
      <td>14606356</td>
      <td>0.525663</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>62</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>full</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.7280</td>
      <td>298.080886</td>
      <td>11057380</td>
      <td>0.518066</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>16</td>
      <td>8</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0010</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.7272</td>
      <td>198.257193</td>
      <td>14606308</td>
      <td>0.515051</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>63</th>
      <td>16</td>
      <td>12</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.8144</td>
      <td>249.676049</td>
      <td>21704164</td>
      <td>0.514133</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0010</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.6808</td>
      <td>171.588939</td>
      <td>11057380</td>
      <td>0.513030</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>full</td>
      <td>0.0001</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.7192</td>
      <td>304.567210</td>
      <td>11057380</td>
      <td>0.507104</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>52</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.7014</td>
      <td>256.237370</td>
      <td>11057416</td>
      <td>0.505413</td>
      <td>5</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>






```python
enas_df.sort_values("val_acc", ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patch_size</th>
      <th>depth</th>
      <th>num_heads</th>
      <th>use_spt</th>
      <th>use_lsa</th>
      <th>finetune_mode</th>
      <th>lr</th>
      <th>weight_decay</th>
      <th>proxy_epochs</th>
      <th>val_acc</th>
      <th>time_sec</th>
      <th>params</th>
      <th>fitness</th>
      <th>generation</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44</th>
      <td>16</td>
      <td>12</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>full</td>
      <td>0.0001</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.8640</td>
      <td>565.185635</td>
      <td>21704164</td>
      <td>0.458563</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>75</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>full</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.8312</td>
      <td>692.030400</td>
      <td>18155296</td>
      <td>0.418970</td>
      <td>7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>63</th>
      <td>16</td>
      <td>12</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.8144</td>
      <td>249.676049</td>
      <td>21704164</td>
      <td>0.514133</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>50</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.8132</td>
      <td>225.859128</td>
      <td>18155236</td>
      <td>0.556361</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>16</td>
      <td>12</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.8114</td>
      <td>388.460479</td>
      <td>21704236</td>
      <td>0.464871</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>60</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.8108</td>
      <td>226.331010</td>
      <td>18155236</td>
      <td>0.553804</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>90</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.8052</td>
      <td>218.764402</td>
      <td>18155236</td>
      <td>0.550726</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.7960</td>
      <td>335.902688</td>
      <td>18155296</td>
      <td>0.502479</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>74</th>
      <td>16</td>
      <td>10</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>full</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.7938</td>
      <td>481.401350</td>
      <td>18155236</td>
      <td>0.451781</td>
      <td>7</td>
      <td>4</td>
    </tr>
    <tr>
      <th>37</th>
      <td>16</td>
      <td>12</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>full</td>
      <td>0.0003</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.7904</td>
      <td>564.507250</td>
      <td>21704164</td>
      <td>0.385189</td>
      <td>3</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
enas_df.groupby("depth")["fitness"].mean()
```




    depth
    6     0.396740
    8     0.336351
    10    0.269662
    12    0.205804
    Name: fitness, dtype: float64




```python
enas_df.groupby("depth")["val_acc"].mean().plot(kind="bar")
plt.title("Mean validation value by depth")
plt.ylabel("Val Accuracy")
plt.show()
```


    
![png](hauptprojekt_files/hauptprojekt_32_0.png)
    



```python
enas_df.groupby("use_spt")["val_acc"].max()

```




    use_spt
    0    0.8640
    1    0.6554
    Name: val_acc, dtype: float64




```python
enas_df.groupby("use_lsa")["val_acc"].max()
```




    use_lsa
    0    0.8640
    1    0.8312
    Name: val_acc, dtype: float64




```python
plt.scatter(enas_df["params"] / 1e6, enas_df["val_acc"])
plt.xlabel("Params (Million)")
plt.ylabel("Validation Accuracy")
plt.title("Accuracy vs Model Size")
plt.show()
```


    
![png](hauptprojekt_files/hauptprojekt_35_0.png)
    



```python
plt.figure(figsize=(8,6))
plt.scatter(enas_df["params"] / 1e6, enas_df["val_acc"], alpha=0.7)

plt.xlabel("Parameters (Million)")
plt.ylabel("Validation Accuracy")
plt.title("Accuracy vs Model Size")
plt.grid(True)
plt.show()
```


```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(enas_df["depth"], enas_df["num_heads"], enas_df["val_acc"])

ax.set_xlabel("Depth")
ax.set_ylabel("Heads")
ax.set_zlabel("Val Acc")

plt.show()
```


    
![png](hauptprojekt_files/hauptprojekt_37_0.png)
    



```python
best_per_gen = enas_df[enas_df["rank"] == 0]
best_per_gen
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patch_size</th>
      <th>depth</th>
      <th>num_heads</th>
      <th>use_spt</th>
      <th>use_lsa</th>
      <th>finetune_mode</th>
      <th>lr</th>
      <th>weight_decay</th>
      <th>proxy_epochs</th>
      <th>val_acc</th>
      <th>time_sec</th>
      <th>params</th>
      <th>fitness</th>
      <th>generation</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0010</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.6022</td>
      <td>182.015210</td>
      <td>11057380</td>
      <td>0.430954</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0010</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.6808</td>
      <td>171.588939</td>
      <td>11057380</td>
      <td>0.513030</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>16</td>
      <td>8</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0010</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.7272</td>
      <td>198.257193</td>
      <td>14606308</td>
      <td>0.515051</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.6960</td>
      <td>173.060886</td>
      <td>11057380</td>
      <td>0.527739</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>16</td>
      <td>8</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.7570</td>
      <td>197.358562</td>
      <td>14606308</td>
      <td>0.545151</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.8132</td>
      <td>225.859128</td>
      <td>18155236</td>
      <td>0.556361</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>60</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.8108</td>
      <td>226.331010</td>
      <td>18155236</td>
      <td>0.553804</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>70</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.6924</td>
      <td>167.684403</td>
      <td>11057380</td>
      <td>0.525931</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>16</td>
      <td>8</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.7588</td>
      <td>193.229803</td>
      <td>14606308</td>
      <td>0.548327</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>90</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.8052</td>
      <td>218.764402</td>
      <td>18155236</td>
      <td>0.550726</td>
      <td>9</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
top5 = (
    enas_df.sort_values("fitness", ascending=False).drop_duplicates(subset=["depth", "num_heads", "use_spt", "use_lsa", "finetune_mode"]).head(5)
)

top5
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patch_size</th>
      <th>depth</th>
      <th>num_heads</th>
      <th>use_spt</th>
      <th>use_lsa</th>
      <th>finetune_mode</th>
      <th>lr</th>
      <th>weight_decay</th>
      <th>proxy_epochs</th>
      <th>val_acc</th>
      <th>time_sec</th>
      <th>params</th>
      <th>fitness</th>
      <th>generation</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.8132</td>
      <td>225.859128</td>
      <td>18155236</td>
      <td>0.556361</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>16</td>
      <td>8</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.7588</td>
      <td>193.229803</td>
      <td>14606308</td>
      <td>0.548327</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>81</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.7032</td>
      <td>168.039326</td>
      <td>11057380</td>
      <td>0.536613</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71</th>
      <td>16</td>
      <td>8</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.7700</td>
      <td>294.821623</td>
      <td>14606356</td>
      <td>0.525663</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>62</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>full</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.7280</td>
      <td>298.080886</td>
      <td>11057380</td>
      <td>0.518066</td>
      <td>6</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
top10 = (
    enas_df.sort_values("val_acc", ascending=False).drop_duplicates(subset=["depth", "num_heads", "use_spt", "use_lsa", "finetune_mode"]).head(10)
)

top10
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patch_size</th>
      <th>depth</th>
      <th>num_heads</th>
      <th>use_spt</th>
      <th>use_lsa</th>
      <th>finetune_mode</th>
      <th>lr</th>
      <th>weight_decay</th>
      <th>proxy_epochs</th>
      <th>val_acc</th>
      <th>time_sec</th>
      <th>params</th>
      <th>fitness</th>
      <th>generation</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44</th>
      <td>16</td>
      <td>12</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>full</td>
      <td>0.0001</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.8640</td>
      <td>565.185635</td>
      <td>21704164</td>
      <td>0.458563</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>75</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>full</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.8312</td>
      <td>692.030400</td>
      <td>18155296</td>
      <td>0.418970</td>
      <td>7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>63</th>
      <td>16</td>
      <td>12</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.8144</td>
      <td>249.676049</td>
      <td>21704164</td>
      <td>0.514133</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>50</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.8132</td>
      <td>225.859128</td>
      <td>18155236</td>
      <td>0.556361</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>16</td>
      <td>12</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.8114</td>
      <td>388.460479</td>
      <td>21704236</td>
      <td>0.464871</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>72</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.7960</td>
      <td>335.902688</td>
      <td>18155296</td>
      <td>0.502479</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>74</th>
      <td>16</td>
      <td>10</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>full</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.7938</td>
      <td>481.401350</td>
      <td>18155236</td>
      <td>0.451781</td>
      <td>7</td>
      <td>4</td>
    </tr>
    <tr>
      <th>57</th>
      <td>16</td>
      <td>10</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>full</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.7772</td>
      <td>638.310735</td>
      <td>18155276</td>
      <td>0.382877</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>71</th>
      <td>16</td>
      <td>8</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.7700</td>
      <td>294.821623</td>
      <td>14606356</td>
      <td>0.525663</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>84</th>
      <td>16</td>
      <td>8</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>full</td>
      <td>0.0001</td>
      <td>0.0005</td>
      <td>5</td>
      <td>0.7652</td>
      <td>393.633783</td>
      <td>14606308</td>
      <td>0.487926</td>
      <td>8</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
worst3 = (
    enas_df.sort_values("fitness", ascending=True).drop_duplicates(subset=["depth", "num_heads", "use_spt", "use_lsa", "finetune_mode"]).head(3)
)

worst3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patch_size</th>
      <th>depth</th>
      <th>num_heads</th>
      <th>use_spt</th>
      <th>use_lsa</th>
      <th>finetune_mode</th>
      <th>lr</th>
      <th>weight_decay</th>
      <th>proxy_epochs</th>
      <th>val_acc</th>
      <th>time_sec</th>
      <th>params</th>
      <th>fitness</th>
      <th>generation</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>89</th>
      <td>16</td>
      <td>12</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>full</td>
      <td>0.001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.3620</td>
      <td>849.200845</td>
      <td>22884652</td>
      <td>-0.149913</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>49</th>
      <td>16</td>
      <td>10</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>full</td>
      <td>0.001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.3710</td>
      <td>812.141948</td>
      <td>19335732</td>
      <td>-0.093071</td>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <th>19</th>
      <td>16</td>
      <td>12</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>last2</td>
      <td>0.001</td>
      <td>0.0001</td>
      <td>5</td>
      <td>0.2422</td>
      <td>285.128413</td>
      <td>22884580</td>
      <td>-0.081689</td>
      <td>1</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
top_configs = top5[[
    "patch_size",
    "depth",
    "num_heads",
    "use_spt",
    "use_lsa",
    "finetune_mode",
    "lr",
    "weight_decay"
]].to_dict(orient="records")

top_configs
```




    [{'patch_size': 16,
      'depth': 10,
      'num_heads': 6,
      'use_spt': 0,
      'use_lsa': 0,
      'finetune_mode': 'last2',
      'lr': 0.0001,
      'weight_decay': 0.0005},
     {'patch_size': 16,
      'depth': 8,
      'num_heads': 6,
      'use_spt': 0,
      'use_lsa': 0,
      'finetune_mode': 'last2',
      'lr': 0.0003,
      'weight_decay': 0.0005},
     {'patch_size': 16,
      'depth': 6,
      'num_heads': 6,
      'use_spt': 0,
      'use_lsa': 0,
      'finetune_mode': 'last2',
      'lr': 0.0001,
      'weight_decay': 0.0001},
     {'patch_size': 16,
      'depth': 8,
      'num_heads': 6,
      'use_spt': 0,
      'use_lsa': 1,
      'finetune_mode': 'last2',
      'lr': 0.0001,
      'weight_decay': 0.0005},
     {'patch_size': 16,
      'depth': 6,
      'num_heads': 6,
      'use_spt': 0,
      'use_lsa': 0,
      'finetune_mode': 'full',
      'lr': 0.0001,
      'weight_decay': 0.0001}]




```python
def full_train(
    model,
    train_loader,
    val_loader,
    test_loader,
    epochs=100,
    lr=5e-5,
    weight_decay=1e-3
):

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )

    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    best_val = 0.0
    best_state = None

    history = []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_amp(
            model, train_loader, optimizer, scaler
        )

        val_loss, val_acc = eval_one_epoch_amp(
            model, val_loader
        )

        history.append({
            "epoch": epoch,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()

    # restore best model
    model.load_state_dict(best_state)

    _, test_acc = eval_one_epoch_amp(model, test_loader)

    return best_val, test_acc, history
```


```python
final_results = []

for i, cfg in enumerate(top_configs):
    print(f"\n=== True Training Model {i} ===")

    model = build_vit_from_cfg(num_classes, cfg)

    best_val, test_acc, history = full_train(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs=20,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    final_results.append({
        **cfg,
        "best_val": best_val,
        "test_acc": test_acc
    })

pd.DataFrame(final_results)
```

    
    === True Training Model 0 ===


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


    Epoch 001/20 | train_acc=0.6428 | val_acc=0.7758
    Epoch 002/20 | train_acc=0.8454 | val_acc=0.8082
    Epoch 003/20 | train_acc=0.9158 | val_acc=0.8056
    Epoch 004/20 | train_acc=0.9624 | val_acc=0.8124
    Epoch 005/20 | train_acc=0.9857 | val_acc=0.8180
    Epoch 006/20 | train_acc=0.9946 | val_acc=0.8164
    Epoch 007/20 | train_acc=0.9969 | val_acc=0.8138
    Epoch 008/20 | train_acc=0.9974 | val_acc=0.8118
    Epoch 009/20 | train_acc=0.9898 | val_acc=0.7980
    Epoch 010/20 | train_acc=0.9941 | val_acc=0.8064
    Epoch 011/20 | train_acc=0.9976 | val_acc=0.8114
    Epoch 012/20 | train_acc=0.9986 | val_acc=0.8094
    Epoch 013/20 | train_acc=0.9944 | val_acc=0.8008
    Epoch 014/20 | train_acc=0.9927 | val_acc=0.8078
    Epoch 015/20 | train_acc=0.9977 | val_acc=0.8066
    Epoch 016/20 | train_acc=0.9952 | val_acc=0.8074
    Epoch 017/20 | train_acc=0.9964 | val_acc=0.8010
    Epoch 018/20 | train_acc=0.9961 | val_acc=0.7996
    Epoch 019/20 | train_acc=0.9960 | val_acc=0.8020
    Epoch 020/20 | train_acc=0.9980 | val_acc=0.8042
    
    === True Training Model 1 ===


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


    Epoch 001/20 | train_acc=0.6228 | val_acc=0.7332
    Epoch 002/20 | train_acc=0.8217 | val_acc=0.7580
    Epoch 003/20 | train_acc=0.9114 | val_acc=0.7530
    Epoch 004/20 | train_acc=0.9586 | val_acc=0.7586
    Epoch 005/20 | train_acc=0.9739 | val_acc=0.7490
    Epoch 006/20 | train_acc=0.9691 | val_acc=0.7450
    Epoch 007/20 | train_acc=0.9705 | val_acc=0.7338
    Epoch 008/20 | train_acc=0.9763 | val_acc=0.7408
    Epoch 009/20 | train_acc=0.9796 | val_acc=0.7466
    Epoch 010/20 | train_acc=0.9784 | val_acc=0.7492
    Epoch 011/20 | train_acc=0.9816 | val_acc=0.7536
    Epoch 012/20 | train_acc=0.9802 | val_acc=0.7604
    Epoch 013/20 | train_acc=0.9820 | val_acc=0.7596
    Epoch 014/20 | train_acc=0.9836 | val_acc=0.7586
    Epoch 015/20 | train_acc=0.9837 | val_acc=0.7590
    Epoch 016/20 | train_acc=0.9829 | val_acc=0.7502
    Epoch 017/20 | train_acc=0.9846 | val_acc=0.7496
    Epoch 018/20 | train_acc=0.9858 | val_acc=0.7462
    Epoch 019/20 | train_acc=0.9901 | val_acc=0.7486
    Epoch 020/20 | train_acc=0.9841 | val_acc=0.7452
    
    === True Training Model 2 ===


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


    Epoch 001/20 | train_acc=0.4436 | val_acc=0.6046
    Epoch 002/20 | train_acc=0.6846 | val_acc=0.6694
    Epoch 003/20 | train_acc=0.7723 | val_acc=0.6872
    Epoch 004/20 | train_acc=0.8427 | val_acc=0.6990
    Epoch 005/20 | train_acc=0.9071 | val_acc=0.7066
    Epoch 006/20 | train_acc=0.9559 | val_acc=0.7110
    Epoch 007/20 | train_acc=0.9829 | val_acc=0.7118
    Epoch 008/20 | train_acc=0.9933 | val_acc=0.7110
    Epoch 009/20 | train_acc=0.9963 | val_acc=0.7116
    Epoch 010/20 | train_acc=0.9946 | val_acc=0.6874
    Epoch 011/20 | train_acc=0.9883 | val_acc=0.6930
    Epoch 012/20 | train_acc=0.9970 | val_acc=0.6988
    Epoch 013/20 | train_acc=0.9976 | val_acc=0.6838
    Epoch 014/20 | train_acc=0.9791 | val_acc=0.7022
    Epoch 015/20 | train_acc=0.9981 | val_acc=0.7164
    Epoch 016/20 | train_acc=0.9993 | val_acc=0.7176
    Epoch 017/20 | train_acc=0.9995 | val_acc=0.7176
    Epoch 018/20 | train_acc=0.9996 | val_acc=0.7232
    Epoch 019/20 | train_acc=0.9688 | val_acc=0.6988
    Epoch 020/20 | train_acc=0.9918 | val_acc=0.6982
    
    === True Training Model 3 ===


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


    Epoch 001/20 | train_acc=0.5534 | val_acc=0.6980
    Epoch 002/20 | train_acc=0.7815 | val_acc=0.7390
    Epoch 003/20 | train_acc=0.8679 | val_acc=0.7554
    Epoch 004/20 | train_acc=0.9334 | val_acc=0.7632
    Epoch 005/20 | train_acc=0.9741 | val_acc=0.7688
    Epoch 006/20 | train_acc=0.9926 | val_acc=0.7722
    Epoch 007/20 | train_acc=0.9971 | val_acc=0.7772
    Epoch 008/20 | train_acc=0.9975 | val_acc=0.7712
    Epoch 009/20 | train_acc=0.9945 | val_acc=0.7524
    Epoch 010/20 | train_acc=0.9944 | val_acc=0.7540
    Epoch 011/20 | train_acc=0.9978 | val_acc=0.7670
    Epoch 012/20 | train_acc=0.9986 | val_acc=0.7704
    Epoch 013/20 | train_acc=0.9888 | val_acc=0.7600
    Epoch 014/20 | train_acc=0.9966 | val_acc=0.7600
    Epoch 015/20 | train_acc=0.9994 | val_acc=0.7726
    Epoch 016/20 | train_acc=0.9981 | val_acc=0.7568
    Epoch 017/20 | train_acc=0.9885 | val_acc=0.7606
    Epoch 018/20 | train_acc=0.9967 | val_acc=0.7570
    Epoch 019/20 | train_acc=0.9984 | val_acc=0.7686
    Epoch 020/20 | train_acc=0.9985 | val_acc=0.7480
    
    === True Training Model 4 ===


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight, blocks.6.attn.proj.bias, blocks.6.attn.proj.weight, blocks.6.attn.qkv.bias, blocks.6.attn.qkv.weight, blocks.6.mlp.fc1.bias, blocks.6.mlp.fc1.weight, blocks.6.mlp.fc2.bias, blocks.6.mlp.fc2.weight, blocks.6.norm1.bias, blocks.6.norm1.weight, blocks.6.norm2.bias, blocks.6.norm2.weight, blocks.7.attn.proj.bias, blocks.7.attn.proj.weight, blocks.7.attn.qkv.bias, blocks.7.attn.qkv.weight, blocks.7.mlp.fc1.bias, blocks.7.mlp.fc1.weight, blocks.7.mlp.fc2.bias, blocks.7.mlp.fc2.weight, blocks.7.norm1.bias, blocks.7.norm1.weight, blocks.7.norm2.bias, blocks.7.norm2.weight, blocks.8.attn.proj.bias, blocks.8.attn.proj.weight, blocks.8.attn.qkv.bias, blocks.8.attn.qkv.weight, blocks.8.mlp.fc1.bias, blocks.8.mlp.fc1.weight, blocks.8.mlp.fc2.bias, blocks.8.mlp.fc2.weight, blocks.8.norm1.bias, blocks.8.norm1.weight, blocks.8.norm2.bias, blocks.8.norm2.weight, blocks.9.attn.proj.bias, blocks.9.attn.proj.weight, blocks.9.attn.qkv.bias, blocks.9.attn.qkv.weight, blocks.9.mlp.fc1.bias, blocks.9.mlp.fc1.weight, blocks.9.mlp.fc2.bias, blocks.9.mlp.fc2.weight, blocks.9.norm1.bias, blocks.9.norm1.weight, blocks.9.norm2.bias, blocks.9.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


    Epoch 001/20 | train_acc=0.4745 | val_acc=0.6206
    Epoch 002/20 | train_acc=0.7319 | val_acc=0.6806
    Epoch 003/20 | train_acc=0.8381 | val_acc=0.7002
    Epoch 004/20 | train_acc=0.9159 | val_acc=0.7034
    Epoch 005/20 | train_acc=0.9630 | val_acc=0.7168
    Epoch 006/20 | train_acc=0.9775 | val_acc=0.7136
    Epoch 007/20 | train_acc=0.9756 | val_acc=0.6870
    Epoch 008/20 | train_acc=0.9712 | val_acc=0.7156
    Epoch 009/20 | train_acc=0.9790 | val_acc=0.7216
    Epoch 010/20 | train_acc=0.9827 | val_acc=0.7114
    Epoch 011/20 | train_acc=0.9758 | val_acc=0.7034
    Epoch 012/20 | train_acc=0.9805 | val_acc=0.7202
    Epoch 013/20 | train_acc=0.9867 | val_acc=0.6922
    Epoch 014/20 | train_acc=0.9772 | val_acc=0.7130
    Epoch 015/20 | train_acc=0.9847 | val_acc=0.7234
    Epoch 016/20 | train_acc=0.9861 | val_acc=0.7030
    Epoch 017/20 | train_acc=0.9834 | val_acc=0.7128
    Epoch 018/20 | train_acc=0.9838 | val_acc=0.7168
    Epoch 019/20 | train_acc=0.9854 | val_acc=0.7102
    Epoch 020/20 | train_acc=0.9867 | val_acc=0.6990





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patch_size</th>
      <th>depth</th>
      <th>num_heads</th>
      <th>use_spt</th>
      <th>use_lsa</th>
      <th>finetune_mode</th>
      <th>lr</th>
      <th>weight_decay</th>
      <th>best_val</th>
      <th>test_acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16</td>
      <td>10</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0005</td>
      <td>0.8180</td>
      <td>0.8092</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16</td>
      <td>8</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0003</td>
      <td>0.0005</td>
      <td>0.7604</td>
      <td>0.7527</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.7232</td>
      <td>0.6950</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>8</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>last2</td>
      <td>0.0001</td>
      <td>0.0005</td>
      <td>0.7772</td>
      <td>0.7497</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>full</td>
      <td>0.0001</td>
      <td>0.0001</td>
      <td>0.7234</td>
      <td>0.7111</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(final_results)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patch_size</th>
      <th>depth</th>
      <th>num_heads</th>
      <th>use_spt</th>
      <th>use_lsa</th>
      <th>finetune_mode</th>
      <th>lr</th>
      <th>weight_decay</th>
      <th>best_val</th>
      <th>test_acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16</td>
      <td>12</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>full</td>
      <td>0.001</td>
      <td>0.0001</td>
      <td>0.412</td>
      <td>0.3914</td>
    </tr>
  </tbody>
</table>
</div>




```python
worst_configs = worst3[[
    "patch_size",
    "depth",
    "num_heads",
    "use_spt",
    "use_lsa",
    "finetune_mode",
    "lr",
    "weight_decay"
]].to_dict(orient="records")

worst_configs

worst_config = worst_configs[2]
```


```python
final_results = []

for i, cfg in enumerate(worst_configs):
    print(f"\n=== True Training Model {i} ===")

    model = build_vit_from_cfg(num_classes, cfg)

    best_val, test_acc, history = full_train(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs=20,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    final_results.append({
        **cfg,
        "best_val": best_val,
        "test_acc": test_acc
    })

pd.DataFrame(final_results)
```

    
    === True Training Model 0 ===
    Epoch 001/20 | train_acc=0.1031 | val_acc=0.1746
    Epoch 002/20 | train_acc=0.1995 | val_acc=0.2140
    Epoch 003/20 | train_acc=0.2616 | val_acc=0.3012
    Epoch 004/20 | train_acc=0.3188 | val_acc=0.3188
    Epoch 005/20 | train_acc=0.3639 | val_acc=0.3496
    Epoch 006/20 | train_acc=0.4104 | val_acc=0.3650
    Epoch 007/20 | train_acc=0.4601 | val_acc=0.3898
    Epoch 008/20 | train_acc=0.5144 | val_acc=0.3904
    Epoch 009/20 | train_acc=0.5778 | val_acc=0.4120
    Epoch 010/20 | train_acc=0.6478 | val_acc=0.4054
    Epoch 011/20 | train_acc=0.7221 | val_acc=0.4040
    Epoch 012/20 | train_acc=0.7918 | val_acc=0.4028
    Epoch 013/20 | train_acc=0.8437 | val_acc=0.3970
    Epoch 014/20 | train_acc=0.8759 | val_acc=0.3940
    Epoch 015/20 | train_acc=0.8949 | val_acc=0.3940
    Epoch 016/20 | train_acc=0.9056 | val_acc=0.4032
    Epoch 017/20 | train_acc=0.9166 | val_acc=0.4094
    Epoch 018/20 | train_acc=0.9245 | val_acc=0.3974
    Epoch 019/20 | train_acc=0.9298 | val_acc=0.4072
    Epoch 020/20 | train_acc=0.9345 | val_acc=0.3934
    
    === True Training Model 1 ===


    Unexpected keys (blocks.10.attn.proj.bias, blocks.10.attn.proj.weight, blocks.10.attn.qkv.bias, blocks.10.attn.qkv.weight, blocks.10.mlp.fc1.bias, blocks.10.mlp.fc1.weight, blocks.10.mlp.fc2.bias, blocks.10.mlp.fc2.weight, blocks.10.norm1.bias, blocks.10.norm1.weight, blocks.10.norm2.bias, blocks.10.norm2.weight, blocks.11.attn.proj.bias, blocks.11.attn.proj.weight, blocks.11.attn.qkv.bias, blocks.11.attn.qkv.weight, blocks.11.mlp.fc1.bias, blocks.11.mlp.fc1.weight, blocks.11.mlp.fc2.bias, blocks.11.mlp.fc2.weight, blocks.11.norm1.bias, blocks.11.norm1.weight, blocks.11.norm2.bias, blocks.11.norm2.weight) found while loading pretrained weights. This may be expected if model is being adapted.


    Epoch 001/20 | train_acc=0.1040 | val_acc=0.1724



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[83], line 8
          4     print(f"\n=== True Training Model {i} ===")
          6     model = build_vit_from_cfg(num_classes, cfg)
    ----> 8     best_val, test_acc, history = full_train(
          9         model,
         10         train_loader,
         11         val_loader,
         12         test_loader,
         13         epochs=20,
         14         lr=cfg["lr"],
         15         weight_decay=cfg["weight_decay"]
         16     )
         18     final_results.append({
         19         **cfg,
         20         "best_val": best_val,
         21         "test_acc": test_acc
         22     })
         24 pd.DataFrame(final_results)


    Cell In[67], line 25, in full_train(model, train_loader, val_loader, test_loader, epochs, lr, weight_decay)
         22 history = []
         24 for epoch in range(epochs):
    ---> 25     train_loss, train_acc = train_one_epoch_amp(
         26         model, train_loader, optimizer, scaler
         27     )
         29     val_loss, val_acc = eval_one_epoch_amp(
         30         model, val_loader
         31     )
         33     history.append({
         34         "epoch": epoch,
         35         "train_acc": train_acc,
         36         "val_acc": val_acc
         37     })


    Cell In[21], line 17, in train_one_epoch_amp(model, loader, optimizer, scaler)
         14     logits = model(x)
         15     loss = loss_fn(logits, y)
    ---> 17 scaler.scale(loss).backward()
         18 scaler.step(optimizer)
         19 scaler.update()


    File c:\Users\matsa\miniconda3\envs\dl_cuda\lib\site-packages\torch\_tensor.py:581, in Tensor.backward(self, gradient, retain_graph, create_graph, inputs)
        571 if has_torch_function_unary(self):
        572     return handle_torch_function(
        573         Tensor.backward,
        574         (self,),
       (...)
        579         inputs=inputs,
        580     )
    --> 581 torch.autograd.backward(
        582     self, gradient, retain_graph, create_graph, inputs=inputs
        583 )


    File c:\Users\matsa\miniconda3\envs\dl_cuda\lib\site-packages\torch\autograd\__init__.py:347, in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
        342     retain_graph = create_graph
        344 # The reason we repeat the same comment below is that
        345 # some Python versions print out the first line of a multi-line function
        346 # calls in the traceback and some print out the last line
    --> 347 _engine_run_backward(
        348     tensors,
        349     grad_tensors_,
        350     retain_graph,
        351     create_graph,
        352     inputs,
        353     allow_unreachable=True,
        354     accumulate_grad=True,
        355 )


    File c:\Users\matsa\miniconda3\envs\dl_cuda\lib\site-packages\torch\autograd\graph.py:825, in _engine_run_backward(t_outputs, *args, **kwargs)
        823     unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
        824 try:
    --> 825     return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
        826         t_outputs, *args, **kwargs
        827     )  # Calls into the C++ engine to run the backward pass
        828 finally:
        829     if attach_logging_hooks:


    KeyboardInterrupt: 



```python

```
