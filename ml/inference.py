"""Библиотека инференса: единый интерфейс моделей для ml-service, офлайн-сабмита и тестов.

Контракт — ``docs/api-contract.md`` §4::

    bundle = load_bundle("v2")                    # или путь к каталогу версии / manifest.json
    bundle.version, bundle.features, bundle.precision, bundle.capabilities
    out = bundle.predict(features_df, sequences)  # pred_delay_s, p10, p50, p90, p_late, expected_abs_error_s
    factors = bundle.explain(features_df, top=3)  # вклад признаков (CatBoost SHAP), секунды

``sequences`` — последовательности телеметрии ``(len(df), SEQ_LEN, N_CHANNELS)`` из
:func:`shared.sequences.build_sequences` (тот же контекст признаков, что у ``build_features``); нужны только
версиям с возможностью ``sequence``. Без них такая версия считает прогноз без последовательной модели (вес
остальных компонент нормируется), остальные выходы не меняются.

Артефакты версии — каталог ``<models_dir>/<version>/``: ``manifest.json`` + файлы моделей. ``models_dir`` —
переменная окружения ``FORESIGHT_MODELS_DIR`` или ``artifacts/models`` в корне репозитория. Формат манифеста
(``format = "foresight-model/1"``):

* ``version``, ``created_at`` (UTC), ``git_commit`` (код обучения), ``features`` (порядок входа модели),
  ``features_version`` (хэш кода признаков, :func:`ml.dataset.code_version`), ``components``, ``precision``,
  ``capabilities``, ``metrics`` (``cv_mae``, ``test_mae``, …), ``late_threshold_s``;
* ``members`` — модели: ``file``, ``component`` (``catboost`` | ``gru`` | ``transformer``), ``target``
  (``delay`` — задержка, ``quantiles`` — P10/P50/P90 (CatBoost MultiQuantile, поле ``alphas``), ``p_late`` —
  вероятность ``delay > late_threshold_s`` (CatBoost Logloss, поле ``calibration``: ``none`` | ``platt`` |
  ``isotonic`` | ``stack`` — вместе с прогнозом задержки ансамбля), ``abs_error`` — ожидаемая абсолютная
  ошибка прогноза (CatBoost по out-of-fold ошибкам ансамбля)), ``base`` (колонка-база,
  к которой модель предсказывает остаток, или ``null``). У последовательных моделей (ONNX) ещё
  ``fp32_file`` (необязательно), ``precision`` и ``sequence`` (``len``, ``step_s``, ``channels``,
  ``version``);
* ``ensemble`` — объединение членов ``target = delay``: ``{"method": "mean"}`` (среднее) или
  ``{"method": "weighted", "weights": {компонент: вес}}`` (взвешенное среднее средних по компонентам);
* ``intervals`` — построение P10/P50/P90: ``center`` (``q50`` — медиана модели квантилей, ``pred`` —
  прогноз ансамбля), ``cqr_offset_s`` (поправка CQR, расширяет интервал), ``abs_error_k``
  (``expected_abs_error_s = k · (p90 − p10)``, если в версии нет члена ``abs_error``);
* ``sequence_version`` — хэш кода последовательностей (:func:`shared.sequences.sequence_version`);
* ``files`` — ``{имя: {sha256, bytes}}``, проверяется при загрузке;
* ``online_calibration`` (необязательно) — поправка дообучения на потоке (:mod:`ml.online`): ``a``, ``b``;
  прогноз и P10/P50/P90 версии — ``a + b · выход базовой модели`` (``p_late`` и вклад признаков — базовой).

Работает на CPU (CatBoost; ONNX Runtime CPU — для последовательных моделей, без torch). Возможность, которой у
версии нет (например, квантили у v1), в ответе — ``NaN`` (в JSON ml-service — ``null``).

Загрузка отказывает, если код признаков или последовательностей изменился с момента обучения
(``features_version`` / ``sequence_version`` манифеста ≠ текущему), — как ``ml/submit.py``: иначе модель
молча получит другие входы, чем при обучении.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from ml.feature_labels import label

MANIFEST_NAME = "manifest.json"
FORMAT = "foresight-model/1"
MODELS_DIR_ENV = "FORESIGHT_MODELS_DIR"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "models"
LATE_THRESHOLD_S = 120.0

OUTPUT_COLUMNS = ["pred_delay_s", "p10", "p50", "p90", "p_late", "expected_abs_error_s"]
EXPECTED_COLUMN = "_expected"

CAP_QUANTILES = "quantiles"
CAP_P_LATE = "p_late"
CAP_FACTORS = "factors"
CAP_SEQUENCE = "sequence"
CAPABILITIES = frozenset({CAP_QUANTILES, CAP_P_LATE, CAP_FACTORS, CAP_SEQUENCE})

SEQ_COMPONENTS = frozenset({"gru", "transformer"})
SUPPORTED_COMPONENTS = frozenset({"catboost"}) | SEQ_COMPONENTS
TARGET_DELAY, TARGET_QUANTILES, TARGET_P_LATE = "delay", "quantiles", "p_late"
TARGET_ABS_ERROR = "abs_error"
SUPPORTED_TARGETS = frozenset({TARGET_DELAY, TARGET_QUANTILES, TARGET_P_LATE, TARGET_ABS_ERROR})
COMPONENT_TARGETS = {
    "catboost": SUPPORTED_TARGETS,
    **{c: frozenset({TARGET_DELAY}) for c in SEQ_COMPONENTS},
}
SUPPORTED_ENSEMBLES = frozenset({"mean", "weighted"})
PRECISIONS = frozenset({"fp32", "int8"})
CALIBRATIONS = frozenset({"none", "platt", "isotonic", "stack"})
SEQUENCE_COLUMN = "_sequence"
# входы / выход ONNX последовательной модели (ml/seq_model.py)
ONNX_SEQ_INPUT, ONNX_TAB_INPUT, ONNX_OUTPUT = "seq", "tab", "delay_resid"
ORT_THREADS_ENV = "FORESIGHT_ORT_THREADS"
# SHAP CatBoost по умолчанию: Regular на 4 × 1000 деревьев — ≈ 0.65 с на вызов, Approximate — десятки мс
SHAP_CALC_TYPE = "Approximate"


class BundleError(ValueError):
    """Артефакты версии модели отсутствуют, повреждены или не поддерживаются."""


class FeaturesVersionError(BundleError):
    """Код признаков изменился с момента обучения модели."""


def models_dir() -> Path:
    """Каталог версий моделей: ``$FORESIGHT_MODELS_DIR`` или ``artifacts/models`` в корне репозитория."""
    env = os.environ.get(MODELS_DIR_ENV)
    return Path(env) if env else DEFAULT_MODELS_DIR


def file_sha256(path: Path) -> str:
    """SHA-256 файла (hex)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(bundle_dir: Path) -> dict[str, Any]:
    """Прочитать ``manifest.json`` каталога версии.

    Raises:
        BundleError: нет манифеста или он не в формате :data:`FORMAT`.
    """
    path = bundle_dir / MANIFEST_NAME
    if not path.is_file():
        raise BundleError(f"no {MANIFEST_NAME} in {bundle_dir}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise BundleError(f"{path}: format {manifest.get('format')!r} != {FORMAT!r}")
    return manifest


def list_bundles(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Манифесты всех версий в каталоге моделей, по возрастанию ``created_at``.

    Каталоги без корректного манифеста пропускаются. К каждому манифесту добавляется ``path``.
    """
    root = Path(root) if root else models_dir()
    out = []
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            manifest = read_manifest(d)
        except (BundleError, json.JSONDecodeError):
            continue
        out.append({**manifest, "path": str(d)})
    return sorted(out, key=lambda m: (str(m.get("created_at", "")), str(m.get("version", ""))))


def resolve_bundle_dir(path_or_version: str | Path, root: str | Path | None = None) -> Path:
    """Каталог версии по имени версии, пути к каталогу или пути к ``manifest.json``.

    Строка без ``/`` — имя версии в каталоге моделей (``latest`` — последняя по ``created_at``).
    """
    if isinstance(path_or_version, Path) or "/" in path_or_version or os.sep in path_or_version:
        p = Path(path_or_version)
        return p.parent if p.name == MANIFEST_NAME else p
    root = Path(root) if root else models_dir()
    if path_or_version == "latest":
        bundles = list_bundles(root)
        if not bundles:
            raise BundleError(f"no model versions in {root}")
        return Path(bundles[-1]["path"])
    return root / path_or_version


def current_features_version() -> str:
    """Хэш текущего кода признаков (тот же, что пишется в кэш признаков и в манифест при обучении)."""
    from ml.dataset import code_version  # ленивый импорт: тянет детектор и признаки

    return code_version()


def current_sequence_version() -> str:
    """Хэш текущего кода последовательностей (:func:`shared.sequences.sequence_version`)."""
    from shared.sequences import sequence_version

    return sequence_version()


def calibrate(
    cal: Mapping[str, Any] | None, p: np.ndarray, pred: np.ndarray | None = None, threshold_s: float = 0.0
) -> np.ndarray:
    """Калибровка вероятности по описанию из манифеста.

    ``none`` — без изменений; ``platt`` — ``σ(a · logit p + b)``; ``isotonic`` — кусочно-линейная функция по
    точкам ``x → y`` (как ``sklearn.isotonic.IsotonicRegression``, вне диапазона — крайние значения);
    ``stack`` — логистическая регрессия по logit вероятности классификатора и прогнозу задержки:
    ``σ(a · logit p + c · (pred − threshold_s) / pred_scale + b)`` (``pred`` обязателен).
    """
    p = np.asarray(p, dtype=np.float64)
    method = (cal or {}).get("method", "none")
    if method == "none":
        return p
    q = np.clip(p, 1e-6, 1 - 1e-6)
    if method == "platt":
        z = cal["a"] * np.log(q / (1 - q)) + cal["b"]
        return 1.0 / (1.0 + np.exp(-z))
    if method == "stack":
        if pred is None:
            raise ValueError("calibration 'stack' needs the delay prediction")
        x = (np.asarray(pred, dtype=np.float64) - threshold_s) / float(cal["pred_scale"])
        z = cal["a"] * np.log(q / (1 - q)) + cal["c"] * x + cal["b"]
        return 1.0 / (1.0 + np.exp(-z))
    if method == "isotonic":
        return np.interp(p, np.asarray(cal["x"], dtype=np.float64), np.asarray(cal["y"], dtype=np.float64))
    raise BundleError(f"unknown calibration {method!r}")


def features_frame(rows: Iterable[Mapping[str, float | None]], features: list[str]) -> pd.DataFrame:
    """Таблица признаков из строк ``{имя: значение}`` (как в ``POST /predict``); отсутствующие — NaN."""
    frame = pd.DataFrame(list(rows)).reindex(columns=features)
    return frame.apply(pd.to_numeric, errors="raise").astype(np.float64)


@dataclass(frozen=True)
class Member:
    """Модель версии.

    Attributes:
        name: имя (вариант обучения и сид).
        model: CatBoost-модель или сессия ONNX Runtime.
        base: колонка-база остатка (прогноз = база + модель) или ``None``.
        component: ``catboost`` | ``gru`` | ``transformer``.
        target: ``delay`` | ``quantiles`` | ``p_late`` | ``abs_error``.
        meta: описание члена из манифеста.
    """

    name: str
    model: Any
    base: str | None = None
    component: str = "catboost"
    target: str = TARGET_DELAY
    meta: Mapping[str, Any] | None = None


class ModelBundle:
    """Загруженная версия модели. Создаётся :func:`load_bundle`.

    Attributes:
        path: каталог версии.
        manifest: манифест целиком.
        version: имя версии (``v1``, ``v2``, …).
        created_at: момент обучения (ISO 8601, UTC).
        features: имена входных признаков в порядке модели.
        features_version: хэш кода признаков при обучении.
        precision: ``fp32`` или ``int8`` (точность последовательной модели, которая загружена).
        capabilities: подмножество :data:`CAPABILITIES`.
        components: компоненты ансамбля (``catboost``, ``gru``).
        metrics: метрики обучения (``cv_mae``, ``test_mae``, …).
        late_threshold_s: порог опоздания для ``p_late``, с.
        features_version_ok: совпадают ли ``features_version`` и ``sequence_version`` с текущим кодом.
        sequence_shape: ``(SEQ_LEN, N_CHANNELS)`` входа последовательной модели или ``None``.
    """

    def __init__(
        self,
        path: Path,
        manifest: dict[str, Any],
        members: list[Member],
        features_ok: bool,
        precision: str | None = None,
    ) -> None:
        self.path = path
        self.manifest = manifest
        self.version: str = str(manifest["version"])
        self.created_at: str = str(manifest.get("created_at", ""))
        self.features: list[str] = list(manifest["features"])
        self.features_version: str = str(manifest.get("features_version", ""))
        self.precision: str = precision or str(manifest.get("precision", "fp32"))
        self.capabilities: frozenset[str] = frozenset(manifest.get("capabilities", []))
        self.components: list[str] = list(manifest.get("components", []))
        self.metrics: dict[str, Any] = dict(manifest.get("metrics", {}))
        self.late_threshold_s = float(manifest.get("late_threshold_s", LATE_THRESHOLD_S))
        self.features_version_ok = features_ok
        self._members = members
        ens = manifest.get("ensemble", {})
        self._method: str = ens.get("method", "mean")
        self._weights: dict[str, float] = {k: float(v) for k, v in ens.get("weights", {}).items()}
        self._intervals: dict[str, Any] = dict(manifest.get("intervals", {}))
        cal = manifest.get("online_calibration") or {}
        self.online_shift: tuple[float, float] | None = (float(cal["a"]), float(cal["b"])) if cal else None
        self._delay = [m for m in members if m.target == TARGET_DELAY]
        self._quantiles = next((m for m in members if m.target == TARGET_QUANTILES), None)
        self._late = next((m for m in members if m.target == TARGET_P_LATE), None)
        self._abs_error = next((m for m in members if m.target == TARGET_ABS_ERROR), None)
        seq = [m for m in self._delay if m.component in SEQ_COMPONENTS]
        self.sequence_shape: tuple[int, int] | None = None
        if seq:
            spec = seq[0].meta["sequence"]
            self.sequence_shape = (int(spec["len"]), len(spec["channels"]))

    def __repr__(self) -> str:
        return (
            f"ModelBundle(version={self.version!r}, members={len(self._members)}, "
            f"precision={self.precision!r}, capabilities={sorted(self.capabilities)})"
        )

    def info(self) -> dict[str, Any]:
        """Описание версии для ``GET /model/info``."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "features": list(self.features),
            "features_version": self.features_version,
            "metrics": dict(self.metrics),
            "precision": self.precision,
            "components": list(self.components),
            "capabilities": sorted(self.capabilities),
            "sequence": None
            if self.sequence_shape is None
            else {
                "len": self.sequence_shape[0],
                "channels": self.sequence_shape[1],
                "version": self.manifest.get("sequence_version"),
            },
        }

    def matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вход модели: колонки :attr:`features` в порядке модели, float64; отсутствующие — NaN."""
        return df.reindex(columns=self.features).astype(np.float64)

    @staticmethod
    def _base(df: pd.DataFrame, member: Member) -> np.ndarray:
        if member.base is None or member.base not in df.columns:
            return np.zeros(len(df))
        return df[member.base].astype(np.float64).fillna(0.0).to_numpy(dtype=np.float64)

    def _check_sequences(self, sequences: Any, n: int) -> np.ndarray | None:
        if sequences is None or self.sequence_shape is None:
            return None
        arr = np.asarray(sequences, dtype=np.float32)
        if arr.shape != (n, *self.sequence_shape):
            raise ValueError(f"sequences shape {arr.shape} != {(n, *self.sequence_shape)}")
        return arr

    def _member_delay(
        self, m: Member, df: pd.DataFrame, x: pd.DataFrame, seq: np.ndarray | None
    ) -> np.ndarray:
        if m.component == "catboost":
            return m.model.predict(x) + self._base(df, m)
        feeds = {ONNX_SEQ_INPUT: seq, ONNX_TAB_INPUT: x.to_numpy(dtype=np.float32)}
        out = np.asarray(m.model.run([ONNX_OUTPUT], feeds)[0], dtype=np.float64).reshape(-1)
        return out + self._base(df, m)

    def _group_weights(self, has_seq: bool) -> dict[str, float]:
        """Нормированные веса компонент задержки, доступных при данном входе."""
        comps = [
            c for c in dict.fromkeys(m.component for m in self._delay) if has_seq or c not in SEQ_COMPONENTS
        ]
        if self._method == "mean":
            counts = {c: sum(m.component == c for m in self._delay) for c in comps}
            total = sum(counts.values())
            return {c: counts[c] / total for c in comps}
        w = {c: self._weights.get(c, 0.0) for c in comps}
        total = sum(w.values())
        if total <= 0:
            raise ValueError("no delay components with positive weight for this input")
        return {c: v / total for c, v in w.items()}

    def _delay_parts(
        self, df: pd.DataFrame, x: pd.DataFrame, seq: np.ndarray | None
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
        """Прогноз задержки, средние по компонентам и их нормированные веса."""
        weights = self._group_weights(seq is not None)
        groups: dict[str, np.ndarray] = {}
        for comp in weights:
            ms = [m for m in self._delay if m.component == comp]
            # тот же порядок операций, что в ml.train.predict_ensemble: v1 совпадает с ним бит в бит
            groups[comp] = np.mean([self._member_delay(m, df, x, seq) for m in ms], axis=0)
        if len(groups) == 1:
            return next(iter(groups.values())), groups, weights
        pred = np.sum([weights[c] * groups[c] for c in weights], axis=0)
        return pred, groups, weights

    def predict(self, df: pd.DataFrame, sequences: Any = None) -> pd.DataFrame:
        """Прогноз задержки на целевой остановке, интервал P10–P90, вероятность опоздания.

        Args:
            df: признаки точек (лишние колонки игнорируются, отсутствующие признаки — NaN).
            sequences: последовательности ``(len(df), SEQ_LEN, N_CHANNELS)``
                (:func:`shared.sequences.build_sequences`) для версий с возможностью ``sequence``; ``None`` —
                прогноз без последовательной модели. Версии без неё последовательности игнорируют.

        Returns:
            DataFrame с индексом ``df`` и колонками :data:`OUTPUT_COLUMNS` (float64); возможности, которых у
            версии нет, — NaN.

        Raises:
            ValueError: форма ``sequences`` не совпадает с ``(len(df), *sequence_shape)``.
        """
        out = pd.DataFrame(np.nan, index=df.index, columns=OUTPUT_COLUMNS, dtype=np.float64)
        seq = self._check_sequences(sequences, len(df))
        if len(df) == 0:
            return out
        x = self.matrix(df)
        pred, _, _ = self._delay_parts(df, x, seq)
        out["pred_delay_s"] = pred
        if self._quantiles is not None:
            m = self._quantiles
            q = np.asarray(m.model.predict(x), dtype=np.float64).reshape(len(df), -1)
            q = np.sort(q + self._base(df, m)[:, None], axis=1)
            iv = self._intervals
            center = pred if iv.get("center", "q50") == "pred" else q[:, 1]
            off = float(iv.get("cqr_offset_s", 0.0))
            lo = np.minimum(center + (q[:, 0] - q[:, 1]) - off, center)
            hi = np.maximum(center + (q[:, 2] - q[:, 1]) + off, center)
            out["p10"], out["p50"], out["p90"] = lo, center, hi
            if iv.get("abs_error_k") is not None and self._abs_error is None:
                out["expected_abs_error_s"] = float(iv["abs_error_k"]) * (hi - lo)
        if self._abs_error is not None:
            err = np.asarray(self._abs_error.model.predict(x), dtype=np.float64).reshape(-1)
            out["expected_abs_error_s"] = np.maximum(err, 0.0)
        if self._late is not None:
            p = np.asarray(self._late.model.predict_proba(x)[:, 1], dtype=np.float64)
            cal = (self._late.meta or {}).get("calibration")
            out["p_late"] = np.clip(calibrate(cal, p, pred, self.late_threshold_s), 0.0, 1.0)
        if self.online_shift is not None:  # b > 0: порядок P10 ≤ P50 ≤ P90 сохраняется
            a, b = self.online_shift
            for col in ("pred_delay_s", "p10", "p50", "p90"):
                out[col] = a + b * out[col]
        return out

    def contributions(
        self, df: pd.DataFrame, shap_calc_type: str = SHAP_CALC_TYPE, sequences: Any = None
    ) -> pd.DataFrame:
        """Вклад признаков в прогноз ``pred_delay_s`` (CatBoost SHAP), секунды.

        ``shap_calc_type`` — ``Approximate`` (по умолчанию, быстро) или ``Regular`` (точные значения Шепли;
        CatBoost заново готовит таблицы на каждый вызов — ≈ 150 мс на модель из 1000 деревьев).

        SHAP CatBoost-членов задержки усредняется и умножается на вес CatBoost в ансамбле. База остатка
        (``cur_dev_s``) входит в прогноз напрямую и добавляется к вкладу своего признака. Вклад
        последовательной модели (без базы) — колонка :data:`SEQUENCE_COLUMN` (есть у версий с ``sequence``;
        0, если ``sequences`` не переданы). Сумма по строке с колонкой :data:`EXPECTED_COLUMN` равна
        ``pred_delay_s``.

        Returns:
            DataFrame с индексом ``df``: колонки :attr:`features`, [:data:`SEQUENCE_COLUMN`],
            :data:`EXPECTED_COLUMN`.
        """
        has_seq_member = self.sequence_shape is not None
        cols = [*self.features, *([SEQUENCE_COLUMN] if has_seq_member else []), EXPECTED_COLUMN]
        seq = self._check_sequences(sequences, len(df))
        if len(df) == 0:
            return pd.DataFrame(columns=cols, index=df.index, dtype=np.float64)
        x = self.matrix(df)
        weights = self._group_weights(seq is not None)
        members = [m for m in self._delay if m.component in weights]
        per_comp = {c: sum(m.component == c for m in members) for c in weights}
        pool = Pool(x)
        n_feat = len(self.features)
        total = np.zeros((len(df), n_feat + 1))
        seq_col = np.zeros(len(df))
        for m in members:
            # доля члена в прогнозе: среднее по всем (mean) или вес компоненты / число её членов (weighted)
            share = weights[m.component] / per_comp[m.component]
            if m.component == "catboost":
                shap = m.model.get_feature_importance(pool, type="ShapValues", shap_calc_type=shap_calc_type)
                shap = np.asarray(shap, dtype=np.float64)
                if m.base is not None and m.base in self.features:
                    shap[:, self.features.index(m.base)] += self._base(df, m)
                total += share * shap
                continue
            base = self._base(df, m)
            seq_col += share * (self._member_delay(m, df, x, seq) - base)
            if m.base is not None and m.base in self.features:
                total[:, self.features.index(m.base)] += share * base
        blocks = [total[:, :n_feat], *([seq_col[:, None]] if has_seq_member else []), total[:, n_feat:]]
        return pd.DataFrame(np.hstack(blocks), index=df.index, columns=cols)

    def explain(
        self, df: pd.DataFrame, top: int = 3, shap_calc_type: str = SHAP_CALC_TYPE, sequences: Any = None
    ) -> list[list[dict]]:
        """Топ признаков по модулю вклада для каждой строки.

        Ранжируются только признаки (вклад последовательной модели :data:`SEQUENCE_COLUMN` не входит).

        Returns:
            Список (по строкам ``df``) списков ``{"feature", "label", "contribution_s"}``, по убыванию модуля
            вклада; ``label`` — подпись из :mod:`ml.feature_labels` (или имя признака).
        """
        if CAP_FACTORS not in self.capabilities:
            return [[] for _ in range(len(df))]
        contrib = self.contributions(df, shap_calc_type=shap_calc_type, sequences=sequences)
        contrib = contrib[self.features].to_numpy()
        names = self.features
        k = max(0, min(top, len(names)))
        out = []
        for row in contrib:
            order = np.argsort(-np.abs(row), kind="stable")[:k]
            out.append(
                [
                    {"feature": names[j], "label": label(names[j]), "contribution_s": round(float(row[j]), 2)}
                    for j in order
                ]
            )
        return out


def _check_manifest(manifest: dict[str, Any], bundle_dir: Path) -> None:
    for key in ("version", "features", "members"):
        if key not in manifest:
            raise BundleError(f"{bundle_dir}: manifest has no {key!r}")
    caps = set(manifest.get("capabilities", []))
    if not caps <= CAPABILITIES:
        raise BundleError(f"{bundle_dir}: unknown capabilities {sorted(caps - CAPABILITIES)}")
    if manifest.get("precision", "fp32") not in PRECISIONS:
        raise BundleError(f"{bundle_dir}: unknown precision {manifest.get('precision')!r}")
    ens = manifest.get("ensemble", {})
    method = ens.get("method", "mean")
    if method not in SUPPORTED_ENSEMBLES:
        raise BundleError(f"{bundle_dir}: ensemble method {method!r} is not supported by ml/inference.py")
    if not manifest["members"]:
        raise BundleError(f"{bundle_dir}: manifest has no members")
    delay_comps = set()
    for item in manifest["members"]:
        comp, target = item.get("component"), item.get("target")
        if comp not in SUPPORTED_COMPONENTS or target not in COMPONENT_TARGETS[comp]:
            raise BundleError(
                f"{bundle_dir}: member {item.get('file')!r} ({comp}/{target}) "
                f"is not supported by this ml/inference.py"
            )
        if target == TARGET_DELAY:
            delay_comps.add(comp)
        if comp in SEQ_COMPONENTS:
            spec = item.get("sequence") or {}
            if not {"len", "channels"} <= set(spec):
                raise BundleError(f"{bundle_dir}: sequence member {item.get('file')!r} has no sequence spec")
            if item.get("precision", "fp32") not in PRECISIONS:
                raise BundleError(f"{bundle_dir}: unknown precision of {item.get('file')!r}")
        if target == TARGET_QUANTILES and len(item.get("alphas", [])) != 3:
            raise BundleError(f"{bundle_dir}: quantiles member {item.get('file')!r} needs 3 alphas")
        if target == TARGET_P_LATE:
            cal = item.get("calibration") or {}
            if cal.get("method", "none") not in CALIBRATIONS:
                raise BundleError(f"{bundle_dir}: unknown calibration {cal.get('method')!r}")
            need = {"platt": {"a", "b"}, "stack": {"a", "b", "c", "pred_scale"}, "isotonic": {"x", "y"}}
            if not need.get(cal.get("method", "none"), set()) <= set(cal):
                raise BundleError(f"{bundle_dir}: calibration {cal} has no {sorted(need[cal['method']])}")
    if not delay_comps:
        raise BundleError(f"{bundle_dir}: manifest has no delay members")
    if method == "weighted":
        weights = ens.get("weights") or {}
        if set(weights) != delay_comps or any(float(v) < 0 for v in weights.values()):
            raise BundleError(
                f"{bundle_dir}: weights {weights} do not match delay components {sorted(delay_comps)}"
            )
        if sum(float(weights[c]) for c in delay_comps if c not in SEQ_COMPONENTS) <= 0:
            raise BundleError(f"{bundle_dir}: components without sequences need a positive weight")


def _member_file(item: Mapping[str, Any], precision: str | None) -> str:
    if item.get("component") in SEQ_COMPONENTS and precision == "fp32" and item.get("fp32_file"):
        return str(item["fp32_file"])
    return str(item["file"])


def _check_files(
    manifest: dict[str, Any], bundle_dir: Path, verify_hashes: bool, precision: str | None = None
) -> None:
    files = manifest.get("files", {})
    for item in manifest["members"]:
        name = _member_file(item, precision)
        path = bundle_dir / name
        if not path.is_file():
            raise BundleError(f"{bundle_dir}: missing model file {name}")
        if name not in files:
            raise BundleError(f"{bundle_dir}: model file {name} is not listed in manifest files")
        if verify_hashes and file_sha256(path) != files[name]["sha256"]:
            raise BundleError(f"{bundle_dir}: sha256 mismatch for {name}")


def _ort_session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    threads = os.environ.get(ORT_THREADS_ENV)
    if threads:
        opts.intra_op_num_threads = int(threads)
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _load_member(
    item: Mapping[str, Any], bundle_dir: Path, features: list[str], precision: str | None
) -> Member:
    comp, target = item["component"], item["target"]
    name = _member_file(item, precision)
    path = bundle_dir / name
    base = item.get("base")
    if base is not None and base not in features:
        raise BundleError(f"{bundle_dir}: base {base!r} of {name} is not a model feature")
    if comp in SEQ_COMPONENTS:
        try:
            sess = _ort_session(path)
        except Exception as e:  # повреждённый / не-ONNX файл
            raise BundleError(f"{bundle_dir}: cannot load ONNX model {name}: {e}") from e
        inputs = {i.name: i.shape for i in sess.get_inputs()}
        spec = item["sequence"]
        want_seq = [int(spec["len"]), len(spec["channels"])]
        if set(inputs) != {ONNX_SEQ_INPUT, ONNX_TAB_INPUT} or list(inputs[ONNX_SEQ_INPUT][1:]) != want_seq:
            raise BundleError(f"{bundle_dir}: {name} inputs {inputs} do not match sequence spec {want_seq}")
        if inputs[ONNX_TAB_INPUT][1] != len(features):
            raise BundleError(f"{bundle_dir}: {name} expects {inputs[ONNX_TAB_INPUT][1]} features")
        return Member(str(item.get("name", name)), sess, base, comp, target, item)
    model = CatBoostClassifier() if target == TARGET_P_LATE else CatBoostRegressor()
    model.load_model(str(path))
    if model.feature_names_ and list(model.feature_names_) != features:
        raise BundleError(f"{bundle_dir}: {name} features differ from manifest features")
    loss = str(model.get_all_params().get("loss_function", ""))
    expected = {
        TARGET_DELAY: not loss.startswith("MultiQuantile") and loss != "Logloss",
        TARGET_QUANTILES: loss.startswith("MultiQuantile"),
        TARGET_P_LATE: loss in ("Logloss", "CrossEntropy"),
        TARGET_ABS_ERROR: not loss.startswith("MultiQuantile") and loss not in ("Logloss", "CrossEntropy"),
    }[target]
    if not expected:
        raise BundleError(f"{bundle_dir}: {name} (loss {loss}) cannot serve target {target!r}")
    if target == TARGET_QUANTILES:
        alphas = [float(a) for a in loss.partition("alpha=")[2].split(",") if a]
        if not np.allclose(alphas, [float(a) for a in item["alphas"]]):
            raise BundleError(f"{bundle_dir}: {name} alphas {alphas} != manifest {item['alphas']}")
    return Member(str(item.get("name", name)), model, base, comp, target, item)


def load_bundle(
    path_or_version: str | Path,
    *,
    root: str | Path | None = None,
    strict: bool = True,
    verify_hashes: bool = True,
    precision: str | None = None,
) -> ModelBundle:
    """Загрузить версию модели.

    Args:
        path_or_version: имя версии (``v1``, ``latest``) в каталоге моделей, путь к каталогу версии или к
            ``manifest.json``.
        root: каталог моделей (по умолчанию :func:`models_dir`).
        strict: отказать, если код признаков (или последовательностей) изменился с момента обучения;
            ``False`` — загрузить с ``features_version_ok = False``.
        verify_hashes: сверить SHA-256 файлов моделей с манифестом.
        precision: точность последовательной модели: ``None`` — из манифеста, ``fp32`` — файл ``fp32_file``
            (если есть), ``int8`` — основной файл.

    Raises:
        BundleError: нет версии, файлы повреждены или компонент не поддерживается.
        FeaturesVersionError: ``strict`` и ``features_version`` / ``sequence_version`` не совпадают с текущим
            кодом.
    """
    if precision is not None and precision not in PRECISIONS:
        raise BundleError(f"unknown precision {precision!r}")
    bundle_dir = resolve_bundle_dir(path_or_version, root)
    manifest = read_manifest(bundle_dir)
    _check_manifest(manifest, bundle_dir)
    current = current_features_version()
    features_ok = manifest.get("features_version") == current
    if strict and not features_ok:
        raise FeaturesVersionError(
            f"model {manifest['version']}: features code changed since training "
            f"({manifest.get('features_version')} != {current}); retrain or load with strict=False"
        )
    has_seq = any(m.get("component") in SEQ_COMPONENTS for m in manifest["members"])
    if has_seq:
        seq_now = current_sequence_version()
        seq_ok = manifest.get("sequence_version") == seq_now
        if strict and not seq_ok:
            raise FeaturesVersionError(
                f"model {manifest['version']}: sequence code changed since training "
                f"({manifest.get('sequence_version')} != {seq_now}); retrain or load with strict=False"
            )
        features_ok = features_ok and seq_ok
    _check_files(manifest, bundle_dir, verify_hashes, precision)
    features = list(manifest["features"])
    members = [_load_member(item, bundle_dir, features, precision) for item in manifest["members"]]
    effective = None
    if has_seq:
        seq_items = [m for m in manifest["members"] if m.get("component") in SEQ_COMPONENTS]
        loaded = {
            ("fp32" if _member_file(m, precision) == m.get("fp32_file") else m.get("precision", "fp32"))
            for m in seq_items
        }
        effective = "int8" if "int8" in loaded else "fp32"
    return ModelBundle(bundle_dir, manifest, members, features_ok, precision=effective)
