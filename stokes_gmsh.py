"""Résolution d'un écoulement de Stokes incompressible (viscosité constante)
dans un pavé droit maillé par Gmsh, avec FEniCSx (dolfinx).

Description du problème
-----------------------
- Écoulement selon la direction x (coordonnées cartésiennes).
- Conditions aux limites en pression sur les faces de normale x :
    * face x = x_min  (entrée)  : p = p_in
    * face x = x_max  (sortie)  : p = p_out
- Sur toutes les autres faces (normales y et z) :
    u = 0 (paroi imperméable, condition d'adhérence -> débit nul).
- Formulation mixte Taylor--Hood P2 (vitesse) / P1 (pression),
  stable (inf-sup), sur maillage tétraédrique ou hexaédrique.

Équation :  -mu * lap(u) + grad(p) = f   (f = 0 par défaut)
            div(u)             = 0

Imposition de la pression
-------------------------
Dans la formulation mixte, la condition de pression est imposée comme
contrainte normale sur les faces d'entrée/sortie :

    sigma . n = -p_bc . n,   sigma = -p I + mu grad(u)

soit le terme de charge  - int(p_bc n . v) ds  dans le second membre.
C'est la condition "naturelle" de la formulation : elle préserve TOUTES
les équations de continuité (div u = 0 est respecté discrètement, le bilan
de masse est exact aux erreurs d'arrondi près) et fixe le niveau absolu de
la pression (le système n'est pas singulier).

Attention : imposer p fortement (Dirichlet sur les ddl de pression des
faces) supprime les équations de continuité associées et détruit la
conservation de la masse : à éviter avec Taylor--Hood.

Le système mixte monolithique est résolu par factorisation LU directe
(MUMPS via PETSc si disponible, sinon SuperLU_DIST, intégré à dolfinx).

Utilisation
-----------
    python stokes_gmsh.py mon_maillage.msh --mu 1e-3 --p-in 10.0 --p-out 0.0

Le maillage Gmsh (.msh) doit être un volume 3D. Les faces d'entrée/sortie
sont détectées géométriquement (x = x_min / x = x_max) : aucun groupe
physique n'est requis. Si le .msh contient des groupes physiques de
surfaces, on peut les utiliser à la place via --use-physical-tags.

Pour générer un maillage de démonstration (pavé droit) :
    python stokes_gmsh.py --demo box.msh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI

from basix.ufl import element, mixed_element
from dolfinx import fem, io
from dolfinx import mesh as dmesh
try:
    from dolfinx.io.gmsh import read_from_msh  # dolfinx >= 0.12
except ImportError:
    from dolfinx.io.gmshio import read_from_msh  # dolfinx <= 0.10

# Tags des groupes de surfaces (utilisés pour l'export et le calcul de débit)
TAG_INLET = 1
TAG_OUTLET = 2


# ---------------------------------------------------------------------------
# Maillage
# ---------------------------------------------------------------------------
def read_gmsh_mesh(filename: str | Path, gdim: int = 3):
    """Lit un maillage Gmsh (.msh) et renvoie (mesh, cell_tags, facet_tags)."""
    data = read_from_msh(str(filename), MPI.COMM_WORLD, gdim=gdim)
    if hasattr(data, "mesh"):  # dolfinx >= 0.11 : MeshData
        return data.mesh, data.cell_tags, data.facet_tags
    mesh, cell_tags, facet_tags = data  # dolfinx <= 0.10
    return mesh, cell_tags, facet_tags


def locate_box_boundary_facets(mesh):
    """Détecte géométriquement les faces du pavé droit.

    Renvoie (facettes entrée, facettes sortie, facettes parois) où :
      - entrée  : faces situées en x = x_min,
      - sortie  : faces situées en x = x_max,
      - parois  : toutes les autres faces frontière (normales y et z).
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    x = mesh.geometry.x
    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    tol = 1e-6 * max(1.0, x_max - x_min)

    def on_xmin(xa):
        return np.isclose(xa[0], x_min, rtol=0.0, atol=tol)

    def on_xmax(xa):
        return np.isclose(xa[0], x_max, rtol=0.0, atol=tol)

    def on_boundary(xa):
        return np.ones(xa.shape[1], dtype=bool)

    inlet = dmesh.locate_entities_boundary(mesh, fdim, on_xmin)
    outlet = dmesh.locate_entities_boundary(mesh, fdim, on_xmax)
    boundary = dmesh.locate_entities_boundary(mesh, fdim, on_boundary)
    walls = np.setdiff1d(boundary, np.union1d(inlet, outlet))
    return inlet, outlet, walls


def facet_tags_from_physical_groups(mesh, facet_tags, inlet_tag, outlet_tag):
    """Extrait les facettes depuis des groupes physiques Gmsh existants."""
    inlet = facet_tags.find(inlet_tag)
    outlet = facet_tags.find(outlet_tag)
    if inlet.size == 0 or outlet.size == 0:
        raise ValueError(
            f"Groupes physiques introuvables (inlet={inlet_tag}, outlet={outlet_tag}). "
            "Utilisez la détection géométrique (défaut) ou vérifiez les tags."
        )
    boundary = dmesh.locate_entities_boundary(
        mesh,
        mesh.topology.dim - 1,
        lambda xa: np.ones(xa.shape[1], dtype=bool),
    )
    walls = np.setdiff1d(boundary, np.union1d(inlet, outlet))
    return inlet, outlet, walls


def _write_box_msh_ascii(filename: str | Path, Lx=4.0, Ly=1.0, Lz=1.0, h=0.08):
    """Écrit un pavé droit en .msh (format Gmsh 2.2 ASCII, tétraèdres).

    Repli utilisé quand la bibliothèque gmsh n'est pas disponible.
    Groupes physiques : inlet (1, x=x_min), outlet (2, x=x_max), walls (3).
    """
    nx, ny, nz = (max(1, int(round(L / h))) for L in (Lx, Ly, Lz))
    hx, hy, hz = Lx / nx, Ly / ny, Lz / nz
    npx, npy, npz = nx + 1, ny + 1, nz + 1

    def node(i, j, k):
        return i * npy * npz + j * npz + k + 1

    nodes = []
    for i in range(npx):
        for j in range(npy):
            for k in range(npz):
                nodes.append((node(i, j, k), i * hx, j * hy, k * hz))

    kuhn_tets = [
        (0, 1, 2, 6), (0, 1, 5, 6), (0, 4, 5, 6),
        (0, 4, 7, 6), (0, 3, 7, 6), (0, 3, 2, 6),
    ]

    def cube_corners(i, j, k):
        return [
            (i, j, k), (i + 1, j, k), (i + 1, j + 1, k), (i, j + 1, k),
            (i, j, k + 1), (i + 1, j, k + 1), (i + 1, j + 1, k + 1), (i, j + 1, k + 1),
        ]

    def tet_nodes(i, j, k):
        corners = [node(*c) for c in cube_corners(i, j, k)]
        tets = []
        for t in kuhn_tets:
            tn = [corners[c] for c in t]
            p = [np.array(nodes[n - 1][1:]) for n in tn]
            vol = np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p[3] - p[0])
            if vol < 0:
                tn[1], tn[2] = tn[2], tn[1]
            tets.append(tn)
        return tets

    hex_faces = {
        (0, 0): [(0, 3, 7), (0, 7, 4)], (0, 1): [(1, 5, 6), (1, 6, 2)],
        (1, 0): [(0, 4, 5), (0, 5, 1)], (1, 1): [(3, 2, 6), (3, 6, 7)],
        (2, 0): [(0, 1, 2), (0, 2, 3)], (2, 1): [(4, 7, 6), (4, 6, 5)],
    }

    tets, tris_in, tris_out, tris_wall = [], [], [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                tets.extend(tet_nodes(i, j, k))
                corners = [node(*c) for c in cube_corners(i, j, k)]
                for (axis, side), faces in hex_faces.items():
                    phys = {(0, 0): TAG_INLET, (0, 1): TAG_OUTLET}.get((axis, side), 3)
                    tri_list = {1: tris_in, 2: tris_out, 3: tris_wall}[phys]
                    tri_list.extend((phys, [corners[c] for c in f]) for f in faces)

    with open(filename, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$PhysicalNames\n4\n")
        f.write('2 1 "inlet"\n2 2 "outlet"\n2 3 "walls"\n3 1 "volume"\n$EndPhysicalNames\n')
        f.write(f"$Nodes\n{len(nodes)}\n")
        for nid, xx, yy, zz in nodes:
            f.write(f"{nid} {xx:.16g} {yy:.16g} {zz:.16g}\n")
        f.write("$EndNodes\n")
        n_elem = len(tets) + len(tris_in) + len(tris_out) + len(tris_wall)
        f.write(f"$Elements\n{n_elem}\n")
        eid = 0
        for tris in (tris_in, tris_out, tris_wall):
            for phys, nlist in tris:
                eid += 1
                f.write(f"{eid} 2 2 {phys} 1 " + " ".join(map(str, nlist)) + "\n")
        for nlist in tets:
            eid += 1
            f.write(f"{eid} 4 2 1 1 " + " ".join(map(str, nlist)) + "\n")
        f.write("$EndElements\n")
    print(f"Maillage de démonstration écrit : {filename}")



def generate_demo_box(filename: str | Path, Lx=4.0, Ly=1.0, Lz=1.0, h=0.08):
    """Génère un pavé droit maillé (tétraèdres, taille h).

    Utilise la bibliothèque gmsh si elle est disponible, sinon écrit
    directement un maillage structuré au format Gmsh 2.2.
    """
    try:
        import gmsh
    except (ImportError, OSError):
        _write_box_msh_ascii(filename, Lx, Ly, Lz, h)
        return

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("box")
    gmsh.model.occ.addBox(0.0, 0.0, 0.0, Lx, Ly, Lz)
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(2)
    inlet, outlet, walls = [], [], []
    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if np.isclose(com[0], 0.0):
            inlet.append(tag)
        elif np.isclose(com[0], Lx):
            outlet.append(tag)
        else:
            walls.append(tag)

    gmsh.model.addPhysicalGroup(2, inlet, TAG_INLET, "inlet")
    gmsh.model.addPhysicalGroup(2, outlet, TAG_OUTLET, "outlet")
    gmsh.model.addPhysicalGroup(2, walls, 3, "walls")
    gmsh.model.addPhysicalGroup(3, [1], 1, "volume")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(filename))
    gmsh.finalize()
    print(f"Maillage de démonstration écrit : {filename}")


# ---------------------------------------------------------------------------
# Espaces, formulaires, conditions aux limites
# ---------------------------------------------------------------------------
def build_function_space(mesh, velocity_degree: int = 2):
    """Espace mixte Taylor--Hood : vitesse P2 vectorielle, pression P1."""
    cell = mesh.ufl_cell()
    cell = cell if isinstance(cell, str) else cell.cellname
    V_el = element("Lagrange", cell, velocity_degree, shape=(mesh.geometry.dim,))
    Q_el = element("Lagrange", cell, velocity_degree - 1)
    W = fem.functionspace(mesh, mixed_element([V_el, Q_el]))
    return W


def build_bcs(mesh, W, wall_facets):
    """Conditions aux limites de vitesse.

    u = 0 sur les facettes de paroi (imperméable + adhérence -> débit
    nul). La vitesse reste libre sur les faces d'entrée/sortie : c'est la
    pression (imposée comme contrainte normale dans la forme faible) qui
    pilote l'écoulement.
    """
    fdim = mesh.topology.dim - 1

    # Espace vitesse seul (pour la condition u = 0)
    V, V_to_W = W.sub(0).collapse()
    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0
    dofs_wall = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    bc_wall = fem.dirichletbc(u_zero, dofs_wall, W.sub(0))
    return [bc_wall]


def make_facet_tags(mesh, inlet_facets, outlet_facets):
    """Meshtags des faces d'entrée (TAG_INLET) et de sortie (TAG_OUTLET)."""
    fdim = mesh.topology.dim - 1
    entities = np.concatenate((inlet_facets, outlet_facets)).astype(np.int32)
    values = np.concatenate((
        np.full(len(inlet_facets), TAG_INLET, dtype=np.int32),
        np.full(len(outlet_facets), TAG_OUTLET, dtype=np.int32),
    ))
    order = np.argsort(entities)
    return dmesh.meshtags(mesh, fdim, entities[order], values[order])


def build_forms(mesh, W, mu, p_in, p_out, inlet_facets, outlet_facets, body_force=None):
    """Formes variationnelles de Stokes (Taylor--Hood).

    La pression est imposée sur les faces d'entrée/sortie comme contrainte
    normale (condition naturelle de la formulation mixte) :

        sigma . n = -p_bc . n,   sigma = -p I + mu grad(u)

    ce qui ajoute au second membre le terme de charge  - int(p_bc n . v) ds.
    Cette approche préserve toutes les équations de continuité (div u = 0
    est respecté discrètement, le bilan de masse est exact) et fixe le
    niveau absolu de la pression (pas de nullspace à constante près).

    mu : viscosité dynamique (constante) ;
    body_force : Function/Constant vectorielle de volume, ou None.
    """
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    dx = ufl.Measure("dx", domain=mesh)
    ft = make_facet_tags(mesh, inlet_facets, outlet_facets)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    n = ufl.FacetNormal(mesh)

    mu_c = mu if isinstance(mu, (fem.Constant, ufl.Constant)) else fem.Constant(mesh, float(mu))
    f = body_force if body_force is not None else fem.Constant(mesh, np.zeros(mesh.geometry.dim))

    a = (
        mu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        - ufl.inner(p, ufl.div(v)) * dx
        - ufl.inner(ufl.div(u), q) * dx
    )
    L = (
        ufl.inner(f, v) * dx
        - p_in * ufl.inner(n, v) * ds(TAG_INLET)
        - p_out * ufl.inner(n, v) * ds(TAG_OUTLET)
    )
    return a, L


# ---------------------------------------------------------------------------
# Solveur monolithique
# ---------------------------------------------------------------------------
def solve_mixed_system(mesh, W, a, L, bcs):
    """Assemble et résout le système mixte monolithique par LU direct.

    Utilise PETSc (MUMPS/LU) si disponible (dolfinx <= 0.10), sinon le
    solveur SuperLU_DIST intégré à dolfinx >= 0.11.
    """
    a_form = fem.form(a)
    L_form = fem.form(L)

    try:
        from dolfinx.fem.petsc import apply_lifting as petsc_apply_lifting
        from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix
        from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector
        from dolfinx.fem.petsc import set_bc as petsc_set_bc

        import petsc4py
        from petsc4py import PETSc

        petsc4py.init(sys.argv)

        A = petsc_assemble_matrix(a_form, bcs=bcs)
        A.assemble()
        b = petsc_assemble_vector(L_form)
        petsc_apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        petsc_set_bc(b, bcs)

        ksp = PETSc.KSP().create(mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        try:
            pc.setFactorSolverType("mumps")
        except Exception:
            pass

        w = fem.Function(W)
        w_vec = getattr(w.x, "petsc_vec", None)
        if w_vec is None:
            w_vec = w.vector
        ksp.solve(b, w_vec)
        w.x.scatter_forward()
        ksp.destroy()
        b.destroy()
        A.destroy()
        return w

    except ImportError:
        A = fem.assemble_matrix(a_form, bcs=bcs)
        b = fem.assemble_vector(L_form)
        fem.apply_lifting(b.array, [a_form], bcs=[bcs])
        b.scatter_forward()
        for bc in bcs:
            bc.set(b.array)

        from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver

        A_slu = superlu_dist_matrix(A)
        solver = superlu_dist_solver(A_slu)
        x = fem.create_vector(W)
        info = solver.solve(b, x)
        if info != 0:
            raise RuntimeError(f"Échec de SuperLU_DIST (code {info}).")

        w = fem.Function(W)
        w.x.array[:] = x.array
        w.x.scatter_forward()
        return w


# ---------------------------------------------------------------------------
# Post-traitement
# ---------------------------------------------------------------------------
def compute_flux(mesh, uh_V, facets) -> float:
    """Débit volumique Q = int(u . n) ds à travers les facettes données."""
    ft = dmesh.meshtags(
        mesh, mesh.topology.dim - 1, np.sort(facets), np.full(len(facets), 1, dtype=np.int32)
    )
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    n = ufl.FacetNormal(mesh)
    flux = fem.assemble_scalar(fem.form(ufl.inner(uh_V, n) * ds(1)))
    return mesh.comm.allreduce(flux, op=MPI.SUM)


def split_solution(W, w):
    """Sépare la solution mixte en (u, p) sur leurs espaces propres."""
    uh = w.sub(0).collapse()
    ph = w.sub(1).collapse()
    uh.name = "velocity"
    ph.name = "pressure"
    return uh, ph


def save_results(mesh, uh, ph, outdir: Path):
    """Écrit la vitesse et la pression au format XDMF (ParaView).

    XDMF ne stocke que des fonctions continues d'ordre 1 : les champs
    P2 sont interpolés en P1 pour l'export (post-traitement).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    cell = mesh.ufl_cell()
    cell = cell if isinstance(cell, str) else cell.cellname
    gdim = mesh.geometry.dim
    V1 = fem.functionspace(mesh, element("Lagrange", cell, 1, shape=(gdim,)))
    Q1 = fem.functionspace(mesh, element("Lagrange", cell, 1))
    uh1 = fem.Function(V1)
    uh1.interpolate(uh)
    ph1 = fem.Function(Q1)
    ph1.interpolate(ph)
    uh1.name = "velocity"
    ph1.name = "pressure"
    xdmf = io.XDMFFile(mesh.comm, str(outdir / "stokes_resultats.xdmf"), "w")
    xdmf.write_mesh(mesh)
    xdmf.write_function(uh1)
    xdmf.write_function(ph1)
    xdmf.close()


# ---------------------------------------------------------------------------
# Routine principale
# ---------------------------------------------------------------------------
def solve_stokes(
    mesh,
    mu: float,
    p_in: float,
    p_out: float,
    inlet_facets=None,
    outlet_facets=None,
    wall_facets=None,
    velocity_degree: int = 2,
    body_force=None,
):
    """Résout le Stokes sur un maillage déjà lu.

    Si les facettes ne sont pas fournies, elles sont détectées
    géométriquement (pavé droit, écoulement selon x).
    Renvoie (uh, ph) sur les espaces P2 vectoriel et P1.
    """
    if inlet_facets is None or outlet_facets is None or wall_facets is None:
        inlet_facets, outlet_facets, wall_facets = locate_box_boundary_facets(mesh)

    W = build_function_space(mesh, velocity_degree)
    bcs = build_bcs(mesh, W, wall_facets)
    a, L = build_forms(mesh, W, mu, p_in, p_out, inlet_facets, outlet_facets, body_force)
    w = solve_mixed_system(mesh, W, a, L, bcs)
    uh, ph = split_solution(W, w)
    return uh, ph


def main():
    parser = argparse.ArgumentParser(
        description="Stokes incompressible (FEniCSx) sur un maillage Gmsh 3D (pavé droit)."
    )
    parser.add_argument("mesh", nargs="?", help="Fichier maillage Gmsh (.msh)")
    parser.add_argument("--mu", type=float, default=1.0, help="Viscosité dynamique")
    parser.add_argument("--p-in", type=float, default=10.0, help="Pression à l'entrée (x_min)")
    parser.add_argument("--p-out", type=float, default=0.0, help="Pression à la sortie (x_max)")
    parser.add_argument("--degree", type=int, default=2, help="Degré vitesse (Taylor-Hood: 2)")
    parser.add_argument("--use-physical-tags", action="store_true",
                        help="Utiliser les groupes physiques Gmsh au lieu de la détection géométrique")
    parser.add_argument("--inlet-tag", type=int, default=1)
    parser.add_argument("--outlet-tag", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("resultats"), help="Répertoire de sortie")
    parser.add_argument("--demo", type=Path, metavar="BOX.MSH",
                        help="Générer un maillage pavé droit de démonstration puis s'arrêter")
    args = parser.parse_args()

    if args.demo:
        generate_demo_box(args.demo)
        if args.mesh is None:
            return
        args.mesh = args.demo

    if args.mesh is None:
        parser.error("Fournir un fichier .msh (ou --demo pour en générer un).")

    mesh, _cell_tags, facet_tags = read_gmsh_mesh(args.mesh)

    if args.use_physical_tags:
        inlet, outlet, walls = facet_tags_from_physical_groups(
            mesh, facet_tags, args.inlet_tag, args.outlet_tag
        )
    else:
        inlet, outlet, walls = locate_box_boundary_facets(mesh)

    uh, ph = solve_stokes(
        mesh, mu=args.mu, p_in=args.p_in, p_out=args.p_out,
        inlet_facets=inlet, outlet_facets=outlet, wall_facets=walls,
        velocity_degree=args.degree,
    )

    q_in = compute_flux(mesh, uh, inlet)
    q_out = compute_flux(mesh, uh, outlet)
    print(f"Débit entrée  Q_in  = {q_in:+.6e}")
    print(f"Débit sortie  Q_out = {q_out:+.6e}")
    print(f"Bilan (Q_in + Q_out) = {q_in + q_out:+.3e} (conservation de la masse)")

    save_results(mesh, uh, ph, args.output)
    print(f"Résultats écrits dans {args.output / 'stokes_resultats.xdmf'}")


if __name__ == "__main__":
    main()
