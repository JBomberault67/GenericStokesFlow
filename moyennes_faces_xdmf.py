"""Post-traitement PyVista d'un fichier résultat FEniCSx au format XDMF.

Ouvre un fichier .xdmf (écrit par dolfinx/XDMFFile, avec son .h5 associé)
et calcule, sur les faces situées en x = x0 et x = x1 :

  - la moyenne de la pression ;
  - la moyenne de la vitesse selon la direction x (u_x).

Les moyennes sont calculées de deux façons :
  - pondérée par l'aire des faces (physiquement significative) ;
  - simple (moyenne des noeuds de la face).

Utilisation
-----------
    python moyennes_faces_xdmf.py resultats/stokes_resultats.xdmf --x0 0.0 --x1 4.0

Si --x0 / --x1 sont omis, les bornes de la boîte englobante du maillage
sont utilisées. Les noms des champs se règlent via --pressure / --velocity
(détection automatique par défaut : "pressure", "velocity").
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# Types Xdmf -> identifiants de cellules VTK
VTK_CELL_TYPES = {
    "Tetrahedron": 10,
    "Hexahedron": 12,
    "Triangle": 5,
    "Quad": 9,
    "Wedge": 13,
    "Pyramid": 14,
}


# ---------------------------------------------------------------------------
# Lecture : meshio d'abord, puis parseur XDMF/HDF5 natif (fichiers dolfinx)
# ---------------------------------------------------------------------------
def _read_hdf_data(xdmf_path: Path, data_item) -> np.ndarray:
    """Lit un <DataItem Format="HDF">fichier.h5:/chemin/vers/dataset."""
    import h5py

    text = (data_item.text or "").strip()
    if ":" not in text:
        raise ValueError(f"DataItem HDF illisible : {text!r}")
    fname, dset_path = text.rsplit(":", 1)
    fname = fname.split("/")[-1] if "/" in fname else fname
    h5_file = str((xdmf_path.parent / fname).resolve())
    with h5py.File(h5_file, "r") as f:
        return np.asarray(f[dset_path])


def _first_uniform_grid(domain) -> ET.Element | None:
    for grid in domain.iter("Grid"):
        if grid.get("GridType", "Uniform") == "Uniform":
            return grid
    return None


def _resolve_topology_geometry(domain, grid: ET.Element):
    """Retrouve Topology/Geometry, en suivant les xi:include/xpointer éventuels."""
    topo = grid.find("Topology")
    geom = grid.find("Geometry")
    if topo is None or geom is None:
        ref = _first_uniform_grid(domain)
        if topo is None:
            topo = None if ref is None else ref.find("Topology")
        if geom is None:
            geom = None if ref is None else ref.find("Geometry")
    if topo is None or geom is None:
        raise ValueError("Topology/Geometry introuvables dans le fichier XDMF.")
    return topo, geom


def _attribute_data(grid: ET.Element, xdmf_path: Path) -> dict:
    """Renvoie {nom: (type, données)} pour chaque attribut d'une grille uniforme."""
    results = {}
    for attr in grid.findall("Attribute"):
        name = attr.get("Name")
        item = attr.find("DataItem")
        results[name] = (attr.get("AttributeType", "Scalar"), _read_hdf_data(xdmf_path, item))
    return results


def _parse_xdmf_native(filename: Path):
    """Parse un XDMF writ par dolfinx : maillage + attributs au dernier temps."""
    import pyvista as pv

    tree = ET.parse(filename)
    root = tree.getroot()
    domain = root.find("Domain")
    if domain is None:
        raise ValueError("Pas de <Domain> dans le fichier XDMF.")

    topo, geom = _resolve_topology_geometry(domain, _first_uniform_grid(domain))
    topo_item = topo.find("DataItem")
    geom_item = geom.find("DataItem")
    if topo_item is None or geom_item is None:
        raise ValueError("DataItem de Topology/Geometry introuvable.")

    cells = _read_hdf_data(filename, topo_item).astype(np.int64)
    points = _read_hdf_data(filename, geom_item).astype(np.float64)

    topo_type = topo.get("TopologyType", "Mixed")
    nvpe = int(topo.get("NodesPerElement", 0))
    if topo_type == "Mixed" or nvpe == 0:
        raise ValueError(f"Type de topologie non géré : {topo_type!r}")
    cell_type = VTK_CELL_TYPES.get(topo_type)
    if cell_type is None:
        raise ValueError(f"Type de cellule inconnu : {topo_type!r}")
    if cells.shape[1] not in (nvpe, 0) and nvpe:
        pass
    ncells = int(topo.get("NumberOfElements", cells.shape[0]))
    cells = cells[:ncells]

    n_per_cell = {"Tetrahedron": 4, "Hexahedron": 8, "Triangle": 3, "Quad": 4,
                  "Wedge": 6, "Pyramid": 5}[topo_type]
    if cells.shape[1] != n_per_cell:
        raise ValueError(
            f"Incohérence : {cells.shape[1]} noeuds/élément lus, {n_per_cell} attendus."
        )

    grid = pv.UnstructuredGrid(
        np.hstack([np.full((ncells, 1), n_per_cell, dtype=np.int64), cells]).ravel(),
        np.full(ncells, cell_type, dtype=np.uint8),
        points,
    )

    point_data: dict[str, tuple] = {}
    cell_data: dict[str, tuple] = {}
    for collection in domain.findall("Grid"):
        gtype = collection.get("GridType", "Uniform")
        if gtype == "Uniform":
            grids = [collection]
        elif gtype == "Collection":
            grids = collection.findall("Grid")
        else:
            continue
        for g in grids:
            attrs = _attribute_data(g, filename)
            for name, (atype, data) in attrs.items():
                if atype == "Vector":
                    data = data.reshape(-1, 3)
                else:
                    data = data.reshape(-1)
                if data.shape[0] == grid.n_points:
                    point_data[name] = data
                elif data.shape[0] == grid.n_cells:
                    cell_data[name] = data

    for name, data in point_data.items():
        grid.point_data[name] = data
    for name, data in cell_data.items():
        grid.cell_data[name] = data
    return grid


def read_xdmf(filename: str | Path):
    """Charge un XDMF vers un pyvista.UnstructuredGrid.

    Essaie meshio (fichiers génériques), puis un parseur natif pour les
    fichiers écrits par dolfinx (grille temporelle + xi:include).
    """
    import pyvista as pv

    filename = Path(filename)
    try:
        import meshio

        m = meshio.read(filename)
        cells, ctypes = [], []
        for block in m.cells:
            n = {"tetra": 4, "hexahedron": 8, "triangle": 3, "quad": 4,
                 "wedge": 6, "pyramid": 5}.get(block.type)
            if n is None:
                continue
            cells.append(np.hstack([np.full((len(block.data), 1), n), block.data]))
            ctypes.append(np.full(len(block.data), VTK_CELL_TYPES[block.type.capitalize()]))
        grid = pv.UnstructuredGrid(
            np.concatenate([c.ravel() for c in cells]),
            np.concatenate(ctypes).astype(np.uint8),
            np.asarray(m.points, dtype=np.float64),
        )
        for name, data in m.point_data.items():
            arr = np.asarray(data)
            grid.point_data[name] = arr.reshape(-1, 3) if arr.ndim == 2 else arr
        for name, data in m.cell_data.items():
            if len(data):
                arr = np.asarray(data[0])
                grid.cell_data[name] = arr.reshape(-1, 3) if arr.ndim == 2 else arr
        return grid
    except BaseException as err:  # meshio peut lever SystemExit sur certains fichiers
        print(f"  (meshio a échoué : {type(err).__name__} — lecture native XDMF/HDF5)", file=sys.stderr)
        return _parse_xdmf_native(filename)


# ---------------------------------------------------------------------------
# Extraction des faces et moyennes
# ---------------------------------------------------------------------------
def _find_field(data_dict, *keywords):
    for name in data_dict:
        low = name.lower()
        if all(k in low for k in keywords):
            return name
    return None


def _extract_surface(grid):
    """extract_surface avec l'argument `algorithm` si disponible (pyvista récent)."""
    import inspect

    if "algorithm" in inspect.signature(grid.extract_surface).parameters:
        return grid.extract_surface(algorithm="dataset_surface")
    return grid.extract_surface()


def face_means(grid, x: float, tol: float, pressure: str | None, velocity: str | None) -> dict:
    """Moyennes de p et u_x sur les faces situées en x (à tol près)."""
    surf = _extract_surface(grid)
    centers = surf.cell_centers().points[:, 0]
    idx = np.flatnonzero(np.abs(centers - x) <= tol)
    if idx.size == 0:
        raise ValueError(f"Aucune face trouvée en x = {x} (tolérance {tol:g}).")
    face = surf.extract_cells(idx)

    areas = face.compute_cell_sizes(length=False, area=True, volume=False)["Area"]

    p_name = pressure or _find_field(face.point_data, "pressure") or _find_field(face.cell_data, "pressure")
    v_name = velocity or _find_field(face.point_data, "velocity") or _find_field(face.cell_data, "velocity")
    if p_name is None and v_name is None:
        raise ValueError("Champs pression/vitesse introuvables dans le fichier.")

    out: dict = {"area": float(areas.sum()), "n_cells": face.n_cells, "n_points": face.n_points}

    # Intégrales surfaciques (VTK) : moyenne pondérée par l'aire = integrale / aire
    integ = face.integrate_data()
    area_int = float(integ["Area"][0])

    if p_name is not None:
        if p_name in face.point_data:
            p_nodal = np.asarray(face.point_data[p_name]).ravel()
            out["p_moyenne_simple"] = float(p_nodal.mean())
            p_integral = float(integ[p_name][0])
        else:
            p_cell = np.asarray(face.cell_data[p_name]).ravel()
            p_integral = float(np.sum(p_cell * areas))
        out["p_moyenne_aire"] = p_integral / area_int

    if v_name is not None:
        if v_name in face.point_data:
            v_nodal = np.asarray(face.point_data[v_name])
            ux_nodal = v_nodal[:, 0] if v_nodal.ndim == 2 else v_nodal
            out["ux_moyenne_simple"] = float(ux_nodal.mean())
            v_integral = np.asarray(integ[v_name][0])
            ux_integral = float(v_integral[0] if v_integral.ndim > 0 else v_integral)
        else:
            v_cell = np.asarray(face.cell_data[v_name])
            ux_cell = v_cell[:, 0] if v_cell.ndim == 2 else v_cell
            ux_integral = float(np.sum(ux_cell * areas))
        out["ux_moyenne_aire"] = ux_integral / area_int

    return out


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Moyennes de pression et de u_x sur les faces x = x0 et x = x1 (XDMF/PyVista)."
    )
    parser.add_argument("xdmf", help="Fichier résultat .xdmf (avec le .h5 associé)")
    parser.add_argument("--x0", type=float, default=None, help="Position de la première face (défaut : x_min)")
    parser.add_argument("--x1", type=float, default=None, help="Position de la seconde face (défaut : x_max)")
    parser.add_argument("--tol", type=float, default=None,
                        help="Tolérance sur x (défaut : 1e-6 fois la taille de la boîte)")
    parser.add_argument("--pressure", default=None, help="Nom du champ de pression")
    parser.add_argument("--velocity", default=None, help="Nom du champ de vitesse")
    args = parser.parse_args()

    grid = read_xdmf(args.xdmf)
    bounds = grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    x0 = bounds[0] if args.x0 is None else args.x0
    x1 = bounds[1] if args.x1 is None else args.x1
    tol = args.tol if args.tol is not None else 1e-6 * max(1.0, bounds[1] - bounds[0])

    print(f"Champs disponibles : {list(grid.point_data.keys()) + list(grid.cell_data.keys())}")
    print(f"Boîte englobante : x ∈ [{bounds[0]:g}, {bounds[1]:g}], "
          f"y ∈ [{bounds[2]:g}, {bounds[3]:g}], z ∈ [{bounds[4]:g}, {bounds[5]:g}]")

    for label, x in (("x0", x0), ("x1", x1)):
        res = face_means(grid, x, tol, args.pressure, args.velocity)
        print(f"\nFace {label} (x = {x:g}) : {res['n_cells']} faces, "
              f"aire = {res['area']:.6g}, {res['n_points']} noeuds")
        if "p_moyenne_aire" in res:
            print(f"  Pression  <p>  = {res['p_moyenne_aire']:+.8g} (pondérée par l'aire)"
                  + (f"   [moyenne simple : {res['p_moyenne_simple']:+.8g}]"
                     if "p_moyenne_simple" in res else ""))
        if "ux_moyenne_aire" in res:
            print(f"  Vitesse   <ux> = {res['ux_moyenne_aire']:+.8g} (pondérée par l'aire)"
                  + (f"   [moyenne simple : {res['ux_moyenne_simple']:+.8g}]"
                     if "ux_moyenne_simple" in res else ""))


if __name__ == "__main__":
    main()
