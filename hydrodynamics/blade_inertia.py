"""Offline geometry for normal added inertia of the repository's YZ blade planes."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from shapely import LineString, Polygon, unary_union

from .models import ROOT, projected_blade


def load_parameters(path=ROOT/'data/blade_inertia.json'):
    return validate_parameters(json.loads(Path(path).read_text(encoding='utf-8')))


def validate_parameters(p):
    if p['schema_version'] != 1 or p['cells_per_strip'] != 4:
        raise ValueError('Blade inertia requires schema 1 and four cells per strip')
    for key in ('full_normal_mass_kg_per_blade', 'rho_reference_kg_m3', 'wetting_depth_m'):
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    count = p['strips_per_blade']
    if not isinstance(count, int) or not 2 <= count <= 128:
        raise ValueError('strips_per_blade must be an integer in [2,128]')
    return p


@dataclass
class BladeStrips:
    endpoints: np.ndarray  # (2 * strips, 5, 3), material cell endpoints
    full_mass: np.ndarray  # (2 * strips,), sums to nominal mass for each blade
    metadata: list


def make_strips(surfaces, p):
    """Midpoint integration in span; exact outline intersection in chord.

    Plane normal is local +X, span local Y, chord local Z, matching hydro.py.
    Four material cells per strip retain in-plane rotational added inertia.
    Non-single-interval cross-sections are rejected instead of bridging gaps.
    """
    if len(surfaces) != 2:
        raise ValueError('Exactly two projected blade surfaces are required')
    endpoints, masses, metadata = [], [], []
    ns = p['strips_per_blade']
    for triangles in surfaces:
        vertices = np.asarray(triangles).reshape(-1, 3)
        if np.ptp(vertices[:, 0]) > 1e-6:
            raise ValueError('Added-inertia surface must lie in a local YZ plane')
        outline = unary_union([Polygon(t[:, 1:]) for t in triangles])
        ymin, zmin, ymax, zmax = outline.bounds
        ds = (ymax-ymin)/ns
        side_points, weights = [], []
        for y in ymin+(np.arange(ns)+.5)*ds:
            chord = outline.intersection(LineString([(y, zmin-1), (y, zmax+1)]))
            if chord.geom_type != 'LineString' or chord.length <= 0:
                raise ValueError('Blade outline must have one nonempty chord per span station')
            low, high = chord.bounds[1], chord.bounds[3]
            side_points.append(np.column_stack((np.full(5, vertices[:, 0].mean()),
                                                 np.full(5, y), np.linspace(low, high, 5))))
            weights.append(ds*(high-low)**2)
        weights = np.asarray(weights)
        endpoints.extend(side_points)
        masses.extend(p['full_normal_mass_kg_per_blade']*weights/weights.sum())
        metadata.append(dict(projected_area_m2=float(outline.area), span_m=float(ymax-ymin),
                             equivalent_chord_m=float(outline.area/(ymax-ymin))))
    return BladeStrips(np.asarray(endpoints, np.float32), np.asarray(masses, np.float32), metadata)


def repository_strips(repository, p):
    repository = Path(repository)
    paths = [repository/'boat_meshes'/f'{side}_blade_3mm.stl' for side in ('port', 'starboard')]
    strips = make_strips([projected_blade(path)[0] for path in paths], p)
    for info, path in zip(strips.metadata, paths):
        info.update(mesh=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return strips
