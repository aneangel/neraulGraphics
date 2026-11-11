"""Computes the single-scattering directional albedo of the renderer's GGX specular lobe (F = 1) by Monte Carlo over visible-normal samples
(the expectation of G2 / G1) and fits the polynomial that single_scatter_albedo() in src/path_core.h evaluates.

    python tools/rendering/single_scatter_albedo.py

The albedo is below 1 for a rough surface; the shortfall is the light that scatters more than once between microfacets, which the renderer
puts back (Turquin 2019). Rerun this and paste the coefficients if the BRDF's visibility or distribution terms change.
"""

import numpy as np
rng = np.random.default_rng(0)
def ess_mc(mu, r, n=400000):
    """E[G2/G1] over visible-normal samples: the single-scattering directional albedo of the lobe with F = 1, as the renderer samples it."""
    a = r * r; a2 = a * a
    wo = np.array([np.sqrt(1 - mu * mu), 0.0, mu])
    u1, u2 = rng.random(n), rng.random(n)
    vh = np.array([a * wo[0], a * wo[1], wo[2]]); vh /= np.linalg.norm(vh)
    len2 = vh[0] ** 2 + vh[1] ** 2
    t1 = np.array([-vh[1], vh[0], 0.0]) / np.sqrt(len2) if len2 > 1e-12 else np.array([1.0, 0, 0])
    t2 = np.cross(vh, t1)
    rr, phi = np.sqrt(u1), 2 * np.pi * u2
    p1 = rr * np.cos(phi); p2 = rr * np.sin(phi)
    k = 0.5 * (1 + vh[2]); p2 = (1 - k) * np.sqrt(np.maximum(0, 1 - p1 * p1)) + k * p2
    nh = p1[:, None] * t1 + p2[:, None] * t2 + np.sqrt(np.maximum(0, 1 - p1 * p1 - p2 * p2))[:, None] * vh
    hl = np.stack([a * nh[:, 0], a * nh[:, 1], np.maximum(0, nh[:, 2])], -1); hl /= np.linalg.norm(hl, axis=1, keepdims=True)
    wi = 2 * (hl @ wo)[:, None] * hl - wo
    nol = wi[:, 2]; nov = mu
    g1 = 2 * nov / (nov + np.sqrt(a2 + (1 - a2) * nov * nov))
    g2 = 2 * nol * nov / (nol * np.sqrt(nov * nov * (1 - a2) + a2) + nov * np.sqrt(nol * nol * (1 - a2) + a2) + 1e-20)
    w = np.where(nol > 0, g2 / g1, 0.0)
    return float(w.mean())
rs = np.linspace(0.1, 1.0, 19); mus = np.linspace(0.05, 1.0, 20)
Z = np.array([[ess_mc(m, r) for m in mus] for r in rs])
print("E_ss(mu, r): rows roughness, cols mu .05 .3 .6 1.0")
for i in (0, 4, 9, 14, 18):
    print(f"  r={rs[i]:.2f}:", [round(Z[i, j], 3) for j in (0, 5, 11, 19)])
R, M = np.meshgrid(rs, mus, indexing="ij")
for dr, dm in ((3, 3), (4, 4), (5, 4), (5, 5)):
    A = np.stack([R.ravel() ** i * M.ravel() ** j for i in range(dr + 1) for j in range(dm + 1)], -1)
    c, *_ = np.linalg.lstsq(A, Z.ravel(), rcond=None)
    err = np.abs(A @ c - Z.ravel())
    print(f"polynomial degree {dr} x {dm} ({(dr+1)*(dm+1)} coefficients): max error {err.max():.4f}, mean {err.mean():.4f}")
    if (dr, dm) == (4, 4): coefficients = c.reshape(5, 5)
