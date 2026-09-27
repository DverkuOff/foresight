"""Последовательная модель задержки: GRU / маленький Transformer по окну телеметрии + табличные признаки.

Вход — последовательность :mod:`shared.sequences` ``(B, SEQ_LEN, N_CHANNELS)`` и табличные признаки v1
``(B, F)`` «как есть» (NaN — пропуск). Нормировка табличных признаков (медиана / IQR обучающей выборки,
индикатор пропуска) — часть сети, поэтому ONNX-модель принимает сырые признаки. Выход — остаток задержки к
базе (``cur_dev_s``), секунды.

Обучение — PyTorch на GPU (extra ``train``): L1-loss, AdamW, косинусное расписание на ``max_epochs``,
число эпох выбирается по OOF MAE (см. ``ml/train_v2.py``). Экспорт — ONNX (fp32), затем INT8 dynamic
quantization ONNX Runtime. GRU при экспорте разворачивается по шагам (:class:`UnrolledGRU`, та же
математика, что ``nn.GRU``): так все умножения — ``MatMul``, и INT8-квантование покрывает всю сеть, а не
только голову (оператор ONNX ``GRU`` динамическая квантизация ONNX Runtime не поддерживает).

Модуль импортирует torch и нужен только для обучения; инференс (``ml/inference.py``) работает через ONNX
Runtime без torch.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml.inference import ONNX_OUTPUT as OUTPUT
from ml.inference import ONNX_SEQ_INPUT as SEQ_INPUT
from ml.inference import ONNX_TAB_INPUT as TAB_INPUT


@dataclass(frozen=True)
class SeqConfig:
    """Вариант последовательной модели.

    Attributes:
        name: имя варианта.
        kind: ``gru`` | ``transformer`` | ``mlp`` (только табличные признаки — абляция) |
            ``gru_seq`` (только последовательность — абляция).
        hidden: размер скрытого состояния GRU / модели Transformer.
        layers: слоёв Transformer.
        heads: голов внимания Transformer.
        tab_hidden: размер эмбеддинга табличных признаков.
        dropout: dropout.
        lr: learning rate AdamW.
        weight_decay: weight decay AdamW.
        batch_size: размер батча.
        max_epochs: длина косинусного расписания (и максимум эпох).
        synth_weight: вес синтетических ТС в loss.
        base: колонка-база остатка.
        target_scale: масштаб выхода, с.
        seeds: сиды (итог — среднее по сидам).
    """

    name: str
    kind: str = "gru"
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    tab_hidden: int = 64
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 80
    synth_weight: float = 0.5
    base: str | None = "cur_dev_s"
    target_scale: float = 100.0
    seeds: tuple[int, ...] = field(default=(1, 2, 3))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        return d


def set_seed(seed: int) -> None:
    """Детерминизм: сиды python / numpy / torch, детерминированные алгоритмы cuDNN / cuBLAS."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def robust_scaling(tab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Центр (медиана) и масштаб (IQR / 1.35, иначе std, минимум 1e-3) по колонкам с NaN."""
    with np.errstate(all="ignore"):
        center = np.nanmedian(tab, axis=0)
        q75, q25 = np.nanpercentile(tab, [75, 25], axis=0)
        scale = (q75 - q25) / 1.35
        std = np.nanstd(tab, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-3), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-3), scale, 1.0)
    center = np.where(np.isfinite(center), center, 0.0)
    return center.astype(np.float32), scale.astype(np.float32)


class TabEncoder(nn.Module):
    """Табличные признаки: робастная нормировка, пропуск → 0 + индикатор, линейный слой."""

    def __init__(self, center: np.ndarray, scale: np.ndarray, hidden: int, dropout: float) -> None:
        super().__init__()
        n = len(center)
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(2 * n, hidden), nn.GELU(), nn.Dropout(dropout))

    def forward(self, tab: torch.Tensor) -> torch.Tensor:
        miss = torch.isnan(tab)
        filled = torch.where(miss, self.center.expand_as(tab), tab)
        z = torch.clamp((filled - self.center) / self.scale, -10.0, 10.0)
        return self.net(torch.cat([z, miss.to(z.dtype)], dim=1))


class GRUEncoder(nn.Module):
    """Однослойный GRU; выход — последнее скрытое состояние."""

    def __init__(self, n_channels: int, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden, batch_first=True)
        self.out_dim = hidden

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(seq)
        return h[-1]


class UnrolledGRU(nn.Module):
    """Та же сеть, что :class:`GRUEncoder`, развёрнутая по шагам для экспорта в ONNX.

    Порядок гейтов PyTorch ``(r, z, n)``::

        r = σ(W_ir x + b_ir + W_hr h + b_hr)
        z = σ(W_iz x + b_iz + W_hz h + b_hz)
        n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))
        h' = (1 − z) ⊙ n + z ⊙ h = n + z ⊙ (h − n)

    Проекция входа считается одним ``MatMul`` на все шаги, на шаге — один ``MatMul`` ``h × W_hh`` с
    постоянными весами и статические срезы гейтов (без вычисления форм в графе). INT8-квантование ONNX
    Runtime превращает эти умножения в ``MatMulInteger`` и покрывает рекуррентную часть.
    """

    def __init__(self, enc: GRUEncoder) -> None:
        super().__init__()
        gru = enc.gru
        hd = gru.hidden_size
        self.hidden = hd
        w_ih, w_hh = gru.weight_ih_l0.detach(), gru.weight_hh_l0.detach()
        b_ih, b_hh = gru.bias_ih_l0.detach(), gru.bias_hh_l0.detach()
        b_i = b_ih.clone()
        b_i[: 2 * hd] += b_hh[: 2 * hd]  # b_hr, b_hz складываются с входными; b_hn остаётся внутри r ⊙ (…)
        b_h = torch.zeros_like(b_hh)
        b_h[2 * hd :] = b_hh[2 * hd :]
        self.w_i = nn.Parameter(w_ih.t().contiguous())
        self.b_i = nn.Parameter(b_i)
        self.w_h = nn.Parameter(w_hh.t().contiguous())
        self.b_h = nn.Parameter(b_h)
        self.out_dim = hd

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        hd = self.hidden
        gi = (seq @ self.w_i + self.b_i).transpose(0, 1)  # (L, B, 3H)
        h = torch.zeros_like(gi[0, :, :hd])
        for k in range(seq.shape[1]):
            g = gi[k]
            gh = h @ self.w_h + self.b_h
            r = torch.sigmoid(g[:, :hd] + gh[:, :hd])
            z = torch.sigmoid(g[:, hd : 2 * hd] + gh[:, hd : 2 * hd])
            n = torch.tanh(g[:, 2 * hd :] + r * gh[:, 2 * hd :])
            h = n + z * (h - n)
        return h


class ExportLinear(nn.Module):
    """``nn.Linear`` как ``MatMul`` + ``Add`` (без ``Gemm``): так его квантует ONNX Runtime."""

    def __init__(self, lin: nn.Linear) -> None:
        super().__init__()
        self.w = nn.Parameter(lin.weight.detach().t().contiguous())
        self.b = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.w
        return y + self.b if self.b is not None else y


def _replace_linear(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, ExportLinear(child))
        else:
            _replace_linear(child)


class _Block(nn.Module):
    """Pre-norm блок Transformer (внимание на Linear/MatMul — экспортируется и квантуется)."""

    def __init__(self, d: int, heads: int, ff: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, d = x.shape[1], x.shape[2]
        qkv = self.qkv(self.n1(x)).reshape(-1, seq_len, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dh), dim=-1)
        y = (att @ v).transpose(1, 2).reshape(-1, seq_len, d)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff(self.n2(x)))


class TransformerEncoder(nn.Module):
    """Маленький Transformer: проекция каналов + позиционный эмбеддинг, блоки, [последний шаг; среднее]."""

    def __init__(
        self, n_channels: int, seq_len: int, d: int, heads: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.inp = nn.Linear(n_channels, d)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d) * 0.02)
        self.blocks = nn.ModuleList([_Block(d, heads, 2 * d, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out_dim = 2 * d

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.inp(seq) + self.pos
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return torch.cat([x[:, -1], x.mean(dim=1)], dim=1)


class SeqNet(nn.Module):
    """Последовательность + табличные признаки → остаток задержки к базе, секунды."""

    def __init__(
        self, cfg: SeqConfig, n_channels: int, seq_len: int, center: np.ndarray, scale: np.ndarray
    ) -> None:
        super().__init__()
        self.kind = cfg.kind
        self.target_scale = float(cfg.target_scale)
        self.seq: nn.Module | None
        if cfg.kind in ("gru", "gru_seq"):
            self.seq = GRUEncoder(n_channels, cfg.hidden)
        elif cfg.kind == "transformer":
            self.seq = TransformerEncoder(n_channels, seq_len, cfg.hidden, cfg.heads, cfg.layers, cfg.dropout)
        elif cfg.kind == "mlp":
            self.seq = None
        else:
            raise ValueError(f"unknown kind {cfg.kind!r}")
        use_tab = cfg.kind != "gru_seq"
        self.tab = TabEncoder(center, scale, cfg.tab_hidden, cfg.dropout) if use_tab else None
        d_in = (self.seq.out_dim if self.seq is not None else 0) + (cfg.tab_hidden if use_tab else 0)
        self.head = nn.Sequential(nn.Linear(d_in, 64), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(64, 1))

    def forward(self, seq: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.seq is not None:
            parts.append(self.seq(seq))
        if self.tab is not None:
            parts.append(self.tab(tab))
        return self.head(torch.cat(parts, dim=1)).squeeze(1) * self.target_scale

    def for_export(self) -> SeqNet:
        """Копия для ONNX (eval, веса те же): GRU → :class:`UnrolledGRU`, Linear → :class:`ExportLinear`."""
        import copy

        net = copy.deepcopy(self).eval()
        if isinstance(net.seq, GRUEncoder):
            net.seq = UnrolledGRU(net.seq)
        _replace_linear(net)
        return net


class SeedEnsemble(nn.Module):
    """Среднее сетей (разные сиды) — один ONNX-граф."""

    def __init__(self, nets: list[nn.Module]) -> None:
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, seq: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        return torch.stack([n(seq, tab) for n in self.nets], dim=0).mean(dim=0)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_net(net: nn.Module, seq: torch.Tensor, tab: torch.Tensor, batch: int = 1024) -> np.ndarray:
    """Прогноз сети (eval) по батчам; тензоры уже на устройстве сети."""
    net.eval()
    out = [net(seq[i : i + batch], tab[i : i + batch]) for i in range(0, len(seq), batch)]
    return torch.cat(out).float().cpu().numpy().astype(np.float64) if out else np.zeros(0)


def train_net(
    cfg: SeqConfig,
    seed: int,
    seq: np.ndarray,
    tab: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    epochs: int | None = None,
    eval_sets: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    dev: torch.device | None = None,
    on_epoch: Callable[[int, dict[str, np.ndarray]], None] | None = None,
) -> SeqNet:
    """Обучить сеть одного сида.

    Args:
        cfg: вариант модели.
        seed: сид (инициализация и порядок батчей).
        seq: последовательности ``(N, L, C)``.
        tab: табличные признаки ``(N, F)`` с NaN.
        target: остаток задержки к базе, с.
        weight: веса строк (синтетика — ``cfg.synth_weight``).
        epochs: остановиться после стольких эпох (расписание lr всё равно на ``cfg.max_epochs``).
        eval_sets: ``{имя: (seq, tab)}`` — прогноз после каждой эпохи передаётся в ``on_epoch``.
        dev: устройство (по умолчанию CUDA, если есть).
        on_epoch: ``on_epoch(epoch, {имя: прогноз})`` после каждой эпохи (с 1).

    Returns:
        Обученная сеть (eval, на ``dev``).
    """
    dev = dev or device()
    set_seed(seed)
    center, scale = robust_scaling(tab)
    net = SeqNet(cfg, seq.shape[2], seq.shape[1], center, scale).to(dev)
    xs = torch.as_tensor(np.array(seq, dtype=np.float32), device=dev)
    xt = torch.as_tensor(np.array(tab, dtype=np.float32), device=dev)
    y = torch.as_tensor(target / cfg.target_scale, dtype=torch.float32, device=dev)
    w = torch.as_tensor(weight, dtype=torch.float32, device=dev)
    evals = {
        k: (
            torch.as_tensor(np.array(s, dtype=np.float32), device=dev),
            torch.as_tensor(np.array(t, dtype=np.float32), device=dev),
        )
        for k, (s, t) in (eval_sets or {}).items()
    }
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = len(xs)
    stop = min(epochs or cfg.max_epochs, cfg.max_epochs)
    for epoch in range(1, stop + 1):
        net.train()
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            pred = net(xs[idx], xt[idx]) / cfg.target_scale
            wb = w[idx]
            loss = (wb * (pred - y[idx]).abs()).sum() / wb.sum().clamp_min(1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        if on_epoch is not None:
            on_epoch(epoch, {k: predict_net(net, s, t) for k, (s, t) in evals.items()})
    return net.eval()


def export_onnx(model: nn.Module, path: Path, seq_len: int, n_channels: int, n_tab: int) -> Path:
    """Экспорт сети (``for_export()`` / :class:`SeedEnsemble`) в ONNX fp32 с динамическим батчем.

    Входы ``seq`` ``(B, L, C)`` и ``tab`` ``(B, F)`` float32, выход ``delay_resid`` ``(B,)``.
    """
    model = model.cpu().eval()
    seq = torch.zeros(2, seq_len, n_channels)
    tab = torch.zeros(2, n_tab)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (seq, tab),
        str(path),
        input_names=[SEQ_INPUT, TAB_INPUT],
        output_names=[OUTPUT],
        dynamic_axes={SEQ_INPUT: {0: "batch"}, TAB_INPUT: {0: "batch"}, OUTPUT: {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return path


def quantize_int8(src: Path, dst: Path) -> Path:
    """INT8 dynamic quantization ONNX Runtime: веса MatMul/Gemm — int8, активации квантуются на лету.

    Перед квантованием — рекомендованная ONNX Runtime подготовка (вывод форм и оптимизация графа).
    """
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prep = dst.with_suffix(".prep.onnx")
    try:
        quant_pre_process(str(src), str(prep), skip_symbolic_shape=False)
        quantize_dynamic(
            str(prep),
            str(dst),
            weight_type=QuantType.QInt8,
            per_channel=False,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )
    finally:
        prep.unlink(missing_ok=True)
    return dst
