#!/usr/bin/env python3
"""
Compare metal binding sites: minimization probes vs AlphaFold3 predictions.

For 22 P. putida proteins, compares:
- Good sites from energy minimization (all probes + filtered probes)
- Good sites from AF3 (folded with 1 LA ion)

Outputs per-protein summary, Venn diagrams, and spatial matching statistics.
"""

import csv
import os
import sys
import warnings
from pathlib import Path
from collections import defaultdict
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from Bio.PDB import MMCIFParser, PDBParser

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────────

DONOR_ATOM_NAMES = {'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH'}
DONOR_RADIUS = 4.0  # Angstroms
MIN_DONORS = 5
THRESHOLDS = [3.0, 5.0, 8.0]  # Angstroms, for spatial matching
DEDUP_DIST = 3.0  # Angstroms, merge probes closer than this into one site

REPO = Path(__file__).resolve().parent
DROPBOX = Path.home() / 'Dropbox' / 'BiominingLDRD' / 'datasets' / 'genome_mining' / 'putida'
APO_DIR = REPO / 'inputs'
PROBES_DIR = REPO / 'putida_probes'
AF3_DIR = DROPBOX / 'putida_1la_af3' / 'tmp_putida_cifs'
FINALS_DIR = DROPBOX / '260403_ped_cluster_centroid_probes' / 'finals'
GT_CSV = REPO / 'ground_truth_centroid_probes.csv'
CONFIG_FILE = REPO / 'metal_config_centroid_optimized.yaml'

# UniProt accession -> AF3 CIF filename
AF3_FILE_MAP = {
    'A0A140FW92': 'tra0a140fw92a0a140fw92_psepk_la_model.cif',
    'Q88JG6': 'trq88jg6q88jg6_psepk_la_model.cif',
    'Q88JG7': 'trq88jg7q88jg7_psepk_la_model.cif',
    'Q88JG8': 'spq88jg8pqqd2_psepk_la_model.cif',
    'Q88JG9': 'trq88jg9q88jg9_psepk_la_model.cif',
    'Q88JH0': 'spq88jh0pedh_psepk_la_model.cif',
    'Q88JH1': 'trq88jh1q88jh1_psepk_la_model.cif',
    'Q88JH2': 'trq88jh2q88jh2_psepk_la_model.cif',
    'Q88JH3': 'trq88jh3q88jh3_psepk_la_model.cif',
    'Q88JH4': 'trq88jh4q88jh4_psepk_la_model.cif',
    'Q88JH5': 'spq88jh5pede_psepk_la_model.cif',
    'Q88JH6': 'trq88jh6q88jh6_psepk_la_model.cif',
    'Q88JH7': 'trq88jh7q88jh7_psepk_la_model.cif',
    'Q88JH8': 'trq88jh8q88jh8_psepk_la_model.cif',
    'Q88JI0': 'trq88ji0q88ji0_psepk_la_model.cif',
    'Q88JI1': 'trq88ji1q88ji1_psepk_la_model.cif',
    'Q88JI2': 'trq88ji2q88ji2_psepk_la_model.cif',
    'Q88JI3': 'trq88ji3q88ji3_psepk_la_model.cif',
    'Q88JI4': 'trq88ji4q88ji4_psepk_la_model.cif',
    'Q88JI5': 'trq88ji5q88ji5_psepk_la_model.cif',
    'Q88JI6': 'trq88ji6q88ji6_psepk_la_model.cif',
    'Q88JI7': 'trq88ji7q88ji7_psepk_la_model.cif',
}

_cif_parser = MMCIFParser(QUIET=True)
_pdb_parser = PDBParser(QUIET=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def accession(pid):
    """'AF-Q88JH0-F1' -> 'Q88JH0'"""
    return pid.split('-')[1]


def load_ground_truth():
    """Returns {protein_id: [{probe_number, n_donors, is_good, failed}]}."""
    data = defaultdict(list)
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            data[row['protein_id']].append({
                'probe_number': int(row['probe_number']),
                'n_donors': int(row['n_donors']),
                'is_good': row['is_good_binder'] == 'True',
                'failed': row['failed_minimization'] == 'True',
            })
    return dict(data)


def get_ca_dict(structure, chain_id=None):
    """Get {resnum: CA_Atom} from a structure. If chain_id=None, search all chains."""
    ca = {}
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            for res in chain:
                if res.id[0] == ' ' and 'CA' in res:
                    ca[res.id[1]] = res['CA']
    return ca


def find_la(structure):
    """Find LA atom coordinates in any chain."""
    for model in structure:
        for chain in model:
            for res in chain:
                for atom in res:
                    if atom.name.strip() == 'LA':
                        return atom.get_vector().get_array().copy()
    return None


def count_donors_near(structure, point, chain_id='A'):
    """Count donor atoms (OD1/OD2/OE1/OE2/OG/OG1/OH) within DONOR_RADIUS of point."""
    n = 0
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            for res in chain:
                if res.id[0] != ' ':
                    continue
                for atom in res:
                    if atom.name in DONOR_ATOM_NAMES:
                        if np.linalg.norm(atom.get_vector().get_array() - point) <= DONOR_RADIUS:
                            n += 1
    return n


def kabsch_align(ref_ca, mob_ca):
    """
    Kabsch alignment of mobile Cα atoms onto reference Cα atoms.
    Inputs: dicts {resnum: Atom} from get_ca_dict.
    Returns (R, t, rmsd) where transformed = R @ original + t.
    """
    common = sorted(set(ref_ca) & set(mob_ca))
    if len(common) < 10:
        raise ValueError(f"Only {len(common)} common Cα atoms, need >= 10")

    rc = np.array([ref_ca[r].get_vector().get_array() for r in common])
    mc = np.array([mob_ca[r].get_vector().get_array() for r in common])

    rc_mean, mc_mean = rc.mean(0), mc.mean(0)
    H = (mc - mc_mean).T @ (rc - rc_mean)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = rc_mean - R @ mc_mean

    aligned = (R @ mc.T).T + t
    rmsd = np.sqrt(np.mean(np.sum((aligned - rc) ** 2, axis=1)))
    return R, t, rmsd


# ─── Metalfinder filtering ──────────────────────────────────────────────────

def get_filtered_probe_numbers(pid, apo_cif_path):
    """
    Run the metalfinder pipeline with the optimized config for one protein.
    Returns the set of probe numbers that survive the SphereDonorFilter.

    Matching is done by comparing filtered centroid positions to the known
    probe positions from putida_probes/ PDB files.
    """
    from pyKVFinder.metalfinder.cli import run_metalfinder, load_config

    # Load known probe positions from existing PDB files
    probe_positions = {}
    prefix = f"{pid}-model_v4_centroids_probe_"
    for f in sorted(PROBES_DIR.glob(f"{prefix}*.pdb")):
        num = int(f.stem.split('probe_')[1])
        with open(f) as fh:
            for line in fh:
                if line.startswith('HETATM') and 'PRB' in line:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    probe_positions[num] = np.array([x, y, z])
                    break

    if not probe_positions:
        return set()

    # Run metalfinder with optimized config (SphereDonorFilter enabled)
    config = load_config(str(CONFIG_FILE))

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_metalfinder(
            pdb_file=str(apo_cif_path),
            config=config,
            output_prefix=os.path.join(tmpdir, 'mf'),
            verbose=False,
        )

    filtered_positions = result['probes'].positions
    if len(filtered_positions) == 0:
        return set()

    # Match filtered centroids to known probe positions
    filtered_nums = set()
    for fp in filtered_positions:
        dists = {num: np.linalg.norm(fp - pos) for num, pos in probe_positions.items()}
        best_num = min(dists, key=dists.get)
        if dists[best_num] < 1.0:
            filtered_nums.add(best_num)

    return filtered_nums


# ─── Visualization ───────────────────────────────────────────────────────────

def draw_venn2(ax, a_only, both, b_only, a_label, b_label, title, annotation=''):
    """Draw a 2-set Venn diagram on the given axes."""
    c1 = Circle((-0.25, 0), 0.65, alpha=0.25, color='#4e79a7', lw=2, ec='#4e79a7')
    c2 = Circle((0.25, 0), 0.65, alpha=0.25, color='#e15759', lw=2, ec='#e15759')
    ax.add_patch(c1)
    ax.add_patch(c2)

    fs = 18
    ax.text(-0.50, 0, str(a_only), ha='center', va='center', fontsize=fs, fontweight='bold')
    ax.text(0.00, 0, str(both), ha='center', va='center', fontsize=fs, fontweight='bold')
    ax.text(0.50, 0, str(b_only), ha='center', va='center', fontsize=fs, fontweight='bold')

    ax.text(-0.50, 0.78, a_label, ha='center', fontsize=11, fontweight='bold', color='#4e79a7')
    ax.text(0.50, 0.78, b_label, ha='center', fontsize=11, fontweight='bold', color='#e15759')

    if annotation:
        ax.text(0, -0.85, annotation, ha='center', fontsize=9, style='italic', color='#555')

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.05, 0.95)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_aspect('equal')
    ax.axis('off')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MINIMIZATION vs AF3: METAL BINDING SITE COMPARISON")
    print("=" * 70)

    # ── 1. Load ground truth ──────────────────────────────────────────────
    gt = load_ground_truth()
    good_probes = {pid: {p['probe_number'] for p in probes if p['is_good']}
                   for pid, probes in gt.items()}
    n_good = sum(len(v) for v in good_probes.values())
    n_proteins_good = sum(1 for v in good_probes.values() if v)

    print(f"\n{len(gt)} proteins, {sum(len(v) for v in gt.values())} total probes")
    print(f"{n_good} good probes across {n_proteins_good} proteins\n")

    # ── 2. Run metalfinder to determine filtered probes ───────────────────
    print("Running metalfinder with optimized config...")
    filtered = {}
    for pid in sorted(gt):
        acc_id = accession(pid)
        apo_cif = APO_DIR / f"AF-{acc_id}-F1-model_v4.cif"
        print(f"  {pid}...", end=" ", flush=True)
        try:
            f = get_filtered_probe_numbers(pid, apo_cif)
            filtered[pid] = f
            n_gp = len(good_probes[pid] & f)
            print(f"{len(f)} pass filter, {n_gp} good")
        except Exception as e:
            print(f"ERROR: {e}")
            filtered[pid] = set()

    # ── 3. Process AF3 structures + alignment ─────────────────────────────
    print("\nProcessing AF3 structures and aligning...")

    results = {}  # pid -> {af3_good, af3_donors, af3_la_apo, min_las_apo, best_dist}

    for pid in sorted(gt):
        acc_id = accession(pid)
        af3_path = AF3_DIR / AF3_FILE_MAP[acc_id]
        apo_path = APO_DIR / f"AF-{acc_id}-F1-model_v4.cif"

        print(f"\n  {pid}:")

        # Parse apo (reference frame) and AF3
        apo_s = _cif_parser.get_structure('apo', str(apo_path))
        af3_s = _cif_parser.get_structure('af3', str(af3_path))

        apo_ca = get_ca_dict(apo_s, chain_id='A')
        af3_ca = get_ca_dict(af3_s, chain_id='A')

        # Sequence verification
        if len(apo_ca) != len(af3_ca):
            print(f"    WARNING: Cα count mismatch apo={len(apo_ca)} af3={len(af3_ca)}")

        # AF3: LA position + donor count
        af3_la = find_la(af3_s)
        if af3_la is None:
            print("    No LA atom in AF3 structure!")
            results[pid] = {'af3_good': False, 'af3_donors': 0,
                            'af3_la_apo': None, 'min_las_apo': {}, 'best_dist': float('inf')}
            continue

        af3_donors = count_donors_near(af3_s, af3_la, chain_id='A')
        af3_good = af3_donors >= MIN_DONORS
        print(f"    AF3: {af3_donors} donors -> {'GOOD' if af3_good else 'not good'}")

        # Align AF3 -> apo frame
        R_af3, t_af3, rmsd_af3 = kabsch_align(apo_ca, af3_ca)
        af3_la_apo = R_af3 @ af3_la + t_af3
        print(f"    AF3->apo RMSD: {rmsd_af3:.3f} A")

        # Process each good minimization probe
        min_las_apo = {}
        for pn in sorted(good_probes[pid]):
            probe_info = [p for p in gt[pid] if p['probe_number'] == pn][0]
            if probe_info['failed']:
                continue

            final_pdb = FINALS_DIR / f"AF-{acc_id}-F1-model_v4_centroids_probe_{pn:03d}_final.pdb"
            if not final_pdb.exists():
                print(f"    Probe {pn}: final PDB not found")
                continue

            fin_s = _pdb_parser.get_structure('fin', str(final_pdb))
            fin_ca = get_ca_dict(fin_s)  # chain may be ' '
            fin_la = find_la(fin_s)

            if fin_la is None:
                print(f"    Probe {pn}: no LA in final PDB")
                continue

            R_fin, t_fin, rmsd_fin = kabsch_align(apo_ca, fin_ca)
            la_apo = R_fin @ fin_la + t_fin
            min_las_apo[pn] = la_apo

            dist = np.linalg.norm(la_apo - af3_la_apo)
            print(f"    Probe {pn}: align RMSD={rmsd_fin:.3f} A, dist to AF3 LA={dist:.1f} A")

        best_dist = min(
            (np.linalg.norm(la - af3_la_apo) for la in min_las_apo.values()),
            default=float('inf')
        )

        results[pid] = {
            'af3_good': af3_good,
            'af3_donors': af3_donors,
            'af3_la_apo': af3_la_apo,
            'min_las_apo': min_las_apo,
            'best_dist': best_dist,
        }

    # ── 4. Deduplicate good min sites & build site-level data ───────────
    # Group good probes by spatial proximity (within DEDUP_DIST) per protein.
    # Each cluster becomes one "site" represented by its centroid.
    def dedup_sites(las_apo, probe_nums):
        """Cluster positions within DEDUP_DIST, return list of (centroid, {probe_nums})."""
        if not las_apo:
            return []
        sites = []  # [(centroid_pos, {probe_numbers})]
        for pn in sorted(probe_nums):
            if pn not in las_apo:
                continue
            pos = las_apo[pn]
            merged = False
            for site in sites:
                if np.linalg.norm(site[0] - pos) <= DEDUP_DIST:
                    site[1].add(pn)
                    # Update centroid
                    all_pos = [las_apo[p] for p in site[1]]
                    site[0] = np.mean(all_pos, axis=0)
                    merged = True
                    break
            if not merged:
                sites.append([pos.copy(), {pn}])
        return sites

    # Build site-level structures
    # min_sites[pid] = list of (position, {probe_nums}) for ALL good probes
    # min_sites_filt[pid] = same but only probes that pass filter
    min_sites = {}
    min_sites_filt = {}
    af3_sites = {}  # pid -> (position,) or empty

    for pid in sorted(gt):
        r = results[pid]
        gp_all = good_probes[pid]
        gp_filt = gp_all & filtered.get(pid, set())

        min_sites[pid] = dedup_sites(r['min_las_apo'], gp_all)
        min_sites_filt[pid] = dedup_sites(r['min_las_apo'], gp_filt)

        if r['af3_good'] and r['af3_la_apo'] is not None:
            af3_sites[pid] = [r['af3_la_apo']]
        else:
            af3_sites[pid] = []

    n_min_all = sum(len(s) for s in min_sites.values())
    n_min_filt = sum(len(s) for s in min_sites_filt.values())
    n_af3 = sum(len(s) for s in af3_sites.values())

    # ── 5. Per-protein summary table ──────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"PER-PROTEIN SUMMARY (dedup dist = {DEDUP_DIST} A)")
    print("=" * 70)
    header = f"{'Protein':<20} {'#Sites':>6} {'#Filt':>5} {'AF3don':>6} {'AF3ok':>5} {'BestDist':>8}"
    print(header)
    print("-" * len(header))

    for pid in sorted(gt):
        r = results[pid]
        ns_all = len(min_sites[pid])
        ns_filt = len(min_sites_filt[pid])
        af3_d = r['af3_donors']
        af3_ok = 'YES' if r['af3_good'] else 'no'
        bd = f"{r['best_dist']:.1f}" if r['best_dist'] < 1e6 else '-'
        print(f"{pid:<20} {ns_all:>6} {ns_filt:>5} {af3_d:>6} {af3_ok:>5} {bd:>8}")

    print(f"\nTotal min sites (all):      {n_min_all}")
    print(f"Total min sites (filtered): {n_min_filt}")
    print(f"Total AF3 good sites:       {n_af3}")

    # ── 6. Venn analysis (site level) ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("VENN DIAGRAM ANALYSIS (site level)")
    print("=" * 70)

    venn_data = {}

    for label, sites_dict in [("ALL PROBES", min_sites), ("FILTERED PROBES", min_sites_filt)]:
        print(f"\n{'─' * 40}")
        print(f"  {label}")
        print(f"{'─' * 40}")

        total_min = sum(len(s) for s in sites_dict.values())
        total_af3 = n_af3
        print(f"  Total min sites:  {total_min}")
        print(f"  Total AF3 sites:  {total_af3}")

        for thresh in THRESHOLDS:
            # For each threshold, count:
            # - min sites that match an AF3 site (overlap)
            # - min sites with no AF3 match (min-only)
            # - AF3 sites with no min match (af3-only)
            n_min_matched = 0
            n_af3_matched = 0
            details = []

            for pid in sorted(gt):
                r = results[pid]
                msites = sites_dict[pid]
                asites = af3_sites[pid]
                if not msites and not asites:
                    continue

                af3_matched_this = False
                for site_pos, site_pns in msites:
                    matched = False
                    for af3_pos in asites:
                        if np.linalg.norm(site_pos - af3_pos) <= thresh:
                            matched = True
                            af3_matched_this = True
                            break
                    if matched:
                        n_min_matched += 1
                        details.append(f"{accession(pid)}:p{sorted(site_pns)}")

                if af3_matched_this:
                    n_af3_matched += 1

            min_only = total_min - n_min_matched
            af3_only = total_af3 - n_af3_matched
            overlap = n_min_matched  # each matched min site counts once

            # But we also need to count AF3 sites that matched as overlap
            # The overlap in the Venn = min sites matched + af3 sites matched
            # that aren't already counted. Since each AF3 site can match
            # multiple min sites, the Venn "both" = matched pairs.
            # Simplest: Venn sets are min_sites and af3_sites.
            # |min only| = total_min - n_min_matched
            # |af3 only| = total_af3 - n_af3_matched
            # |both| = n_min_matched + n_af3_matched (double counts shared)
            # Actually for Venn: sites in intersection are those that match.
            # Let's count unique sites: total = min_only + overlap + af3_only
            # where overlap = n_min_matched (min sites near AF3) + n_af3_matched - overlap_pairs
            # Simpler: just report the three regions directly.

            print(f"\n  Threshold {thresh:.0f} A:")
            print(f"    Min sites matched by AF3:  {n_min_matched}")
            print(f"    AF3 sites matched by Min:  {n_af3_matched}")
            print(f"    Min-only sites:            {min_only}")
            print(f"    AF3-only sites:            {af3_only}")
            if details:
                print(f"    Matches: {', '.join(details)}")

            venn_data[(label, thresh)] = (min_only, n_min_matched, n_af3_matched, af3_only)

    # ── 7. Generate figures ───────────────────────────────────────────────
    n_thresh = len(THRESHOLDS)
    fig, axes = plt.subplots(2, n_thresh, figsize=(5 * n_thresh, 10))
    fig.suptitle(
        f"Minimization vs AF3: Good Metal Binding Sites (site level, dedup {DEDUP_DIST:.0f} Å)\n"
        f"Good = ≥{MIN_DONORS} donors ({', '.join(DONOR_ATOM_NAMES)}) within {DONOR_RADIUS} Å of LA",
        fontsize=13, fontweight='bold', y=0.99,
    )

    for row, label in enumerate(["ALL PROBES", "FILTERED PROBES"]):
        for col, thresh in enumerate(THRESHOLDS):
            min_only, n_min_match, n_af3_match, af3_only = venn_data[(label, thresh)]
            # For the Venn circles: left = min sites, right = AF3 sites
            # overlap region shows matched counts from both sides
            overlap_label = f"{n_min_match}m/{n_af3_match}a" if n_min_match != n_af3_match else str(n_min_match)
            draw_venn2(
                axes[row, col], min_only, overlap_label, af3_only,
                f"Min ({min_only + n_min_match})", f"AF3 ({af3_only + n_af3_match})",
                f"{label}\n(match ≤ {thresh:.0f} Å)",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = REPO / 'venn_min_vs_af3.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure: {out_path}")
    plt.close()

    print("\nDone.")


if __name__ == '__main__':
    main()
