"""Rebuild the existing fin as a 3 mm plate with rounded nose and tapered tail.

Boat-forward is +X. Preserve the main XZ silhouette and lateral centre (0.1 mm outline cleanup).
The original mesh is retained in data/fin_original_44mm.stl for reproducibility.
"""
from pathlib import Path
import shutil

import numpy as np
import trimesh
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / 'boat_meshes/fin.stl'
    source = ROOT / 'data/fin_original_44mm.stl'
    if not source.exists():
        shutil.copy2(target, source)
    original = trimesh.load_mesh(source, process=True)
    outline = unary_union([
        Polygon(t[:, [0, 2]]) for t in original.triangles
        if Polygon(t[:, [0, 2]]).area > 1e-14
    ]).simplify(0.0001, preserve_topology=True)
    xmin, zmin, xmax, zmax = outline.bounds
    centre_y = original.bounds[:, 1].mean()
    # Horizontal rings avoid sliver triangles at the original CAD micro-edges.
    levels = np.unique(np.r_[np.asarray(outline.exterior.coords)[:, 1],
                            np.linspace(zmin, zmax, 13)])
    levels = levels[levels > zmin + 1e-5]
    levels = np.r_[zmin + 1e-5, levels]
    fractions = np.array([0, .0003, .001, .003, .006, .009, .012,
                          .02, .05, .1, .2, .4, .6, .8, .9, .95, 1.])
    vertices = []
    for z in levels:
        section = outline.intersection(LineString([(xmin - 1, z), (xmax + 1, z)]))
        lo, _, hi, _ = section.bounds
        chord = hi - lo
        d = fractions * chord
        radius = min(.0015, chord * .1)
        nose = np.sqrt(np.maximum(0., radius**2 - np.maximum(radius-d, 0.)**2))
        nose = np.maximum(.00005, .0015 * nose / radius)
        tail = .00015 + .00135 * np.minimum((chord-d)/(.2*chord), 1.)
        halves = np.minimum(nose, tail)
        ring = [[hi-di, centre_y+h, z] for di,h in zip(d, halves)]
        ring += [[hi-di, centre_y-h, z] for di,h in zip(d[::-1], halves[::-1])]
        vertices.extend(ring)
    k = 2*len(fractions)
    faces = []
    for j in range(len(levels)-1):
        for i in range(k):
            a, b = j*k+i, j*k+(i+1)%k
            faces.extend([[a,b,b+k],[a,b+k,a+k]])
    # Cap the concave section using a polygon triangulation, with identical indices.
    from shapely import constrained_delaunay_triangles
    for j in [0, len(levels)-1]:
        ring = np.asarray(vertices[j*k:(j+1)*k])[:, :2]
        polygon = Polygon(ring)
        for tri in constrained_delaunay_triangles(polygon).geoms:
            ids = [int(np.argmin(np.linalg.norm(ring-np.asarray(v),axis=1)))+j*k
                   for v in list(tri.exterior.coords)[:3]]
            faces.append(ids)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    assert np.isclose(mesh.extents[1], .003, atol=1e-9)
    assert np.all(mesh.area_faces > 1e-16)
    mesh.export(target)
    saved = trimesh.load_mesh(target, process=True)
    assert saved.is_watertight and saved.is_winding_consistent
    assert np.isclose(saved.extents[1], .003, atol=1e-8)
    print(f'Fin: {len(saved.faces)} triangles; thickness {saved.extents[1]*1000:.6f} mm')
    print(f'Side area: {outline.area:.9f} m2; volume: {saved.volume:.9f} m3')
    print(f'Centre Y: {centre_y:.9f} m; watertight: {saved.is_watertight}')


if __name__ == '__main__':
    main()
