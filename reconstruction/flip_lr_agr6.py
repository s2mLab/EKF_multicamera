"""Detecteur de flip gauche/droite `epipolar_agr6` (variante « A6 », sans terme temporel).

Meme contrat que `vitpose_ekf_pipeline.detect_left_right_flip_diagnostics` :
`detect_agr6(...)` retourne `(suspect_mask (n_cam, n_frames) bool, diagnostics, detail_arrays)`.

Formes et unites : points `(n_cam, n_frames, 17, 2)` en pixels de l'image brute (ordre COCO17 inchange), scores
`(n_cam, n_frames, 17)`. NaN ou score <= 0 = point absent (exclu des hypotheses nominale et permutee). Les couts sont des
erreurs de Sampson en pixels ; le FPS n'intervient pas (aucun terme temporel) ; les vues exclues ne sont pas gerees
specifiquement (comme le detecteur `epipolar` : elles sont absentes par NaN / score <= 0).

Methode, par camera et par image :

1. Pour chaque keypoint, erreur de Sampson vers chaque autre vue, hypothese nominale et hypothese « gauche/droite permutes »
   (les 8 paires COCO17 gauche/droite echangees dans la vue testee).
2. Cout par paire symetrique (epaules, coudes, poignets, hanches, genoux, chevilles) : mediane ponderee (poids de paire de
   baseline x poids de keypoint du depot x min des scores) des erreurs de la paire de keypoints vers les autres vues.
3. Cout de vue = MOYENNE des couts des paires informatives (nominal et permute disponibles), au moins `MIN_INFORMATIVE_PAIRS`.
4. Candidats du depot (cout brut ou lisse sur 5 images >= max(5 px, percentile 85 de la camera), ou critere « legacy » idem) ;
   une image candidate est marquee si `cout_permute < RATIO * cout_nominal` et `cout_nominal - cout_permute >= GAIN_PX`.

Mesure preliminaire (leave-one-sequence-out, 83 vrais flips) : precision 75 %, rappel 54 %, F1 0,63 (detecteur `epipolar` du
depot : 78 %, 35 %, 0,48). Voir `docs/architecture/LLM_CONTEXT.md`.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

FLIP_AGR6_METHOD = "epipolar_agr6"
FLIP_AGR6_RATIO = 0.7
FLIP_AGR6_GAIN_PX = 3.0
FLIP_AGR6_SCORING_VERSION = 1
CANDIDATE_PERCENTILE = 85.0
CANDIDATE_FLOOR_PX = 5.0
CANDIDATE_SMOOTH_WINDOW = 5
MIN_INFORMATIVE_PAIRS = 3
# Paires COCO17 gauche/droite : yeux, oreilles, epaules, coudes, poignets, hanches, genoux, chevilles.
LR_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
# Paires utilisees par A6 : epaules, coudes, poignets, hanches, genoux, chevilles (pas la tete).
SYMMETRIC_PAIRS = LR_PAIRS[2:]
MIRROR = np.arange(17)
for _left, _right in LR_PAIRS:
    MIRROR[_left], MIRROR[_right] = _right, _left


def flip_agr6_parameters() -> dict[str, float | int]:
    """Parametres observables de A6 (stockes dans les metadonnees de cache et les diagnostics)."""

    return {
        "improvement_ratio": FLIP_AGR6_RATIO,
        "min_gain_px": FLIP_AGR6_GAIN_PX,
        "candidate_percentile": CANDIDATE_PERCENTILE,
        "candidate_floor_px": CANDIDATE_FLOOR_PX,
        "candidate_smooth_window": CANDIDATE_SMOOTH_WINDOW,
        "min_informative_pairs": MIN_INFORMATIVE_PAIRS,
        "scoring_version": FLIP_AGR6_SCORING_VERSION,
    }


def sampson_tensors(points: np.ndarray, fundamental: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Erreurs de Sampson (px) `(n_cam, n_cam, n_frames, 17)` de la vue c vers la vue o.

    Le premier tenseur utilise les points nominaux de c, le second ses points permutes gauche/droite.
    """

    n_cam, n_frames, n_kp = points.shape[:3]
    homog = np.concatenate([points, np.ones((n_cam, n_frames, n_kp, 1))], axis=-1)
    nominal = np.full((n_cam, n_cam, n_frames, n_kp), np.nan)
    swapped = np.full_like(nominal, np.nan)
    with np.errstate(all="ignore"):
        for c in range(n_cam):
            for o in range(n_cam):
                if o == c:
                    continue
                fmat = fundamental[c, o]
                f_xa = np.einsum("ij,fkj->fki", fmat, homog[c])
                ft_xb = np.einsum("fki,ij->fkj", homog[o], fmat)
                den_b = ft_xb[..., 0] ** 2 + ft_xb[..., 1] ** 2
                num = np.abs(np.sum(homog[o] * f_xa, axis=-1))
                den = np.sqrt(f_xa[..., 0] ** 2 + f_xa[..., 1] ** 2 + den_b)
                nominal[c, o] = np.where(den > 1e-12, num / den, np.nan)
                f_xs = f_xa[:, MIRROR]
                num_s = np.abs(np.sum(homog[o] * f_xs, axis=-1))
                den_s = np.sqrt(f_xs[..., 0] ** 2 + f_xs[..., 1] ** 2 + den_b)
                swapped[c, o] = np.where(den_s > 1e-12, num_s / den_s, np.nan)
    return nominal, swapped


def _weighted_and_plain_median(values: np.ndarray, weights: np.ndarray, valid: np.ndarray):
    """`(..., M) -> (mediane ponderee, mediane simple)` ; NaN si aucune valeur valide."""

    sortable = np.where(valid, values, np.inf)
    w = np.where(valid, weights, 0.0)
    order = np.argsort(sortable, axis=-1, kind="stable")
    sorted_values = np.take_along_axis(sortable, order, -1)
    cumulative = np.cumsum(np.take_along_axis(w, order, -1), -1)
    n_valid = valid.sum(-1)
    m = values.shape[-1]
    index = np.minimum((cumulative < 0.5 * cumulative[..., -1:]).sum(-1), m - 1)
    weighted = np.take_along_axis(sorted_values, index[..., None], -1)[..., 0]
    low = np.take_along_axis(sorted_values, np.clip((n_valid - 1) // 2, 0, m - 1)[..., None], -1)[..., 0]
    high = np.take_along_axis(sorted_values, np.clip(n_valid // 2, 0, m - 1)[..., None], -1)[..., 0]
    empty = n_valid == 0
    return np.where(empty | ~np.isfinite(weighted), np.nan, weighted), np.where(empty, np.nan, 0.5 * (low + high))


def _view_costs(
    errors: np.ndarray,
    scores: np.ndarray,
    finite_own: np.ndarray,
    finite_other: np.ndarray,
    pair_weights: np.ndarray,
    keypoint_weights: np.ndarray,
    keypoints: Sequence[int] | None = None,
    min_other: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """`(n_cam, n_frames)` : cout pondere et cout « legacy » (mediane simple) d'une vue, sur un sous-ensemble de keypoints."""

    n_cam = errors.shape[0]
    sl = slice(None) if keypoints is None else list(keypoints)
    errors, kp_scores = errors[..., sl], scores[..., sl]
    valid = (finite_own & (scores > 0))[..., sl][:, None] & (finite_other & (scores > 0))[..., sl][None, :]
    valid &= ~np.eye(n_cam, dtype=bool)[:, :, None, None]
    informative = valid.any(-1).sum(1)
    valid &= np.isfinite(errors)
    confidence = np.maximum(np.minimum(kp_scores[:, None], kp_scores[None, :]), 1e-6)
    weights = np.maximum(pair_weights[:, :, None, None] * keypoint_weights[sl][None, None, None, :] * confidence, 1e-9)

    def flatten(array: np.ndarray) -> np.ndarray:
        return np.moveaxis(array, 2, 1).reshape(n_cam, errors.shape[2], -1)

    weighted, median = _weighted_and_plain_median(
        flatten(errors), flatten(np.broadcast_to(weights, errors.shape)), flatten(valid)
    )
    too_few = informative < min_other
    return np.where(too_few, np.nan, weighted), np.where(too_few, np.nan, median)


def _candidate_mask(cost: np.ndarray, legacy: np.ndarray) -> np.ndarray:
    """Candidats du depot : cout brut ou lisse >= max(plancher, percentile de la camera), ou critere legacy idem."""

    candidates = np.zeros(cost.shape, bool)
    kernel = np.ones(CANDIDATE_SMOOTH_WINDOW)
    for c in range(cost.shape[0]):
        finite = np.isfinite(cost[c])
        if not finite.any():
            continue
        total = np.convolve(np.where(finite, cost[c], 0.0), kernel, "same")
        count = np.convolve(finite.astype(float), kernel, "same")
        smooth = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        threshold = max(CANDIDATE_FLOOR_PX, np.percentile(cost[c][finite], CANDIDATE_PERCENTILE))
        candidates[c] = (finite & (cost[c] >= threshold)) | (np.isfinite(smooth) & (smooth >= threshold))
        legacy_values = legacy[c][np.isfinite(legacy[c])]
        if legacy_values.size:
            legacy_threshold = max(CANDIDATE_FLOOR_PX, np.percentile(legacy_values, CANDIDATE_PERCENTILE))
            candidates[c] |= np.isfinite(legacy[c]) & (legacy[c] >= legacy_threshold)
    return candidates


def _swap_rule(nominal: np.ndarray, swapped: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (
            np.isfinite(nominal)
            & np.isfinite(swapped)
            & (nominal > 0)
            & (swapped < FLIP_AGR6_RATIO * nominal)
            & ((nominal - swapped) >= FLIP_AGR6_GAIN_PX)
        )


def _pair_mean_costs(pair_nominal: np.ndarray, pair_swapped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cout de vue = moyenne des couts par paire informatifs (nominal et permute finis), `(n_cam, n_frames)`."""

    ok = np.isfinite(pair_nominal) & np.isfinite(pair_swapped)
    n_pairs = ok.sum(-1)
    enough = n_pairs >= MIN_INFORMATIVE_PAIRS
    with np.errstate(all="ignore"):
        nominal = np.where(enough, np.where(ok, pair_nominal, 0.0).sum(-1) / np.maximum(n_pairs, 1), np.nan)
        swapped = np.where(enough, np.where(ok, pair_swapped, 0.0).sum(-1) / np.maximum(n_pairs, 1), np.nan)
    return nominal, swapped


def detect_agr6(
    pose_data,
    calibrations,
    build_fundamental_matrix_array: Callable,
    build_pair_weight_array: Callable,
    keypoint_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    """Detecte les images ou une vue serait plus coherente avec les autres une fois gauche/droite permutes (A6).

    Les fonctions de geometrie du depot (matrices fondamentales, poids de paire) sont passees en argument pour eviter une
    dependance circulaire avec `vitpose_ekf_pipeline`.
    """

    ordered = [calibrations[name] for name in pose_data.camera_names]
    points = np.asarray(pose_data.keypoints, float)
    scores = np.asarray(pose_data.scores, float)
    pair_weights = build_pair_weight_array(ordered)
    weights = np.asarray(keypoint_weights, float)
    nominal_errors, swapped_errors = sampson_tensors(points, build_fundamental_matrix_array(ordered))
    finite = np.all(np.isfinite(points), -1)
    finite_swapped = np.all(np.isfinite(points[:, :, MIRROR]), -1)
    gn, ln = _view_costs(nominal_errors, scores, finite, finite, pair_weights, weights)
    gw, lw = _view_costs(swapped_errors, scores, finite_swapped, finite, pair_weights, weights)
    pair_nominal = np.stack(
        [
            _view_costs(nominal_errors, scores, finite, finite, pair_weights, weights, pair)[0]
            for pair in SYMMETRIC_PAIRS
        ],
        -1,
    )
    pair_swapped = np.stack(
        [
            _view_costs(swapped_errors, scores, finite_swapped, finite, pair_weights, weights, pair)[0]
            for pair in SYMMETRIC_PAIRS
        ],
        -1,
    )
    mean_nominal, mean_swapped = _pair_mean_costs(pair_nominal, pair_swapped)
    candidates = _candidate_mask(gn, ln)
    mask = candidates & _swap_rule(mean_nominal, mean_swapped)

    frames = np.asarray(pose_data.frames)
    names = pose_data.camera_names
    any_frame = mask.any(0)
    diagnostics = {
        "method": FLIP_AGR6_METHOD,
        "variant_parameters": flip_agr6_parameters(),
        "temporal_weight": 0.0,
        "n_frames_with_any_flip_suspect": int(any_frame.sum()),
        "n_camera_frame_flip_suspects": int(mask.sum()),
        "n_candidate_camera_frames_tested": int(candidates.sum()),
        "frames_with_any_flip_suspect": frames[any_frame].astype(int).tolist(),
        "camera_frame_flip_suspects": {names[c]: int(mask[c].sum()) for c in range(len(names))},
        "camera_frame_flip_suspect_frames": {names[c]: frames[mask[c]].astype(int).tolist() for c in range(len(names))},
    }
    detail = {
        "nominal_geometric_costs": gn,
        "swapped_geometric_costs": gw,
        "nominal_legacy_costs": ln,
        "swapped_legacy_costs": lw,
        "candidate_mask": candidates.astype(bool),
    }
    return mask, diagnostics, detail
