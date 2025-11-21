#!/usr/bin/env python3
"""
MetalFinder CLI

Command-line interface for metal binding site identification using pyKVFinder.
Runs the complete pipeline from cavity detection to metal site prediction.
"""

import argparse
import sys
import yaml
from pathlib import Path
import numpy as np

# Import pyKVFinder
import pyKVFinder

# Import metalfinder components
from pyKVFinder.metalfinder import (
    ProbeConverter,
    ProbeSet,
    DistanceFilter,
    CoordinationFilter,
    HardCoordinationFilter,
    SignatureDeduplicator,
    run_filter_pipeline
)
from pyKVFinder.metalfinder.pdb_parser import parse_pdb


def load_config(config_file: str) -> dict:
    """Load YAML configuration file.
    
    Parameters
    ----------
    config_file : str
        Path to YAML configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use 'default' section if present, otherwise use entire config
    if 'default' in config:
        return config['default']
    return config


def run_metalfinder(
    pdb_file: str,
    config: dict,
    output_prefix: str = "metalfinder",
    save_intermediates: bool = False,
    verbose: bool = True
):
    """Run complete metalfinder pipeline.
    
    Parameters
    ----------
    pdb_file : str
        Path to input PDB file
    config : dict
        Configuration dictionary from YAML
    output_prefix : str
        Prefix for output files
    save_intermediates : bool
        Save intermediate PDB files at each filter stage
    verbose : bool
        Print detailed progress information
        
    Returns
    -------
    dict
        Results dictionary with probes and metadata
    """
    if verbose:
        print("="*70)
        print("METALFINDER PIPELINE")
        print("="*70)
        print()
    
    # Step 1: Run pyKVFinder cavity detection
    if verbose:
        print("STEP 1: Running pyKVFinder cavity detection")
        print("-"*70)
    
    # Get KVFinder parameters (use defaults if not specified)
    kvfinder_params = config.get('kvfinder', {})
    
    results = pyKVFinder.run_workflow(
        pdb_file,
        step=kvfinder_params.get('step', 0.6),
        probe_in=kvfinder_params.get('probe_in', 1.4),
        probe_out=kvfinder_params.get('probe_out', 4.0),
        removal_distance=kvfinder_params.get('removal_distance', 2.4),
        volume_cutoff=kvfinder_params.get('volume_cutoff', 5.0)
    )
    
    if verbose:
        print(f"✓ Cavity detection complete")
        print(f"  Cavities grid shape: {results.cavities.shape}")
        print(f"  Number of cavities: {results.ncav}")
        print(f"  Grid step: {results._step} Å")
        print()
    
    # Step 2: Parse protein structure
    if verbose:
        print("STEP 2: Parsing protein structure")
        print("-"*70)
    
    protein_atoms, atom_names, atom_types, residue_names, is_backbone = parse_pdb(pdb_file)
    
    if verbose:
        print(f"✓ Protein structure parsed")
        print(f"  Total atoms: {len(protein_atoms)}")
        print(f"  Backbone atoms: {np.sum(is_backbone)}")
        print(f"  Unique elements: {set(atom_types)}")
        print()
    
    # Step 3: Extract probes from cavity grid
    if verbose:
        print("STEP 3: Extracting probes from cavity grid")
        print("-"*70)
    
    io_config = config.get('io', {})
    converter = ProbeConverter()
    probes = converter.extract_all_probes(
        results,
        include_cavity_interior=io_config.get('include_cavities', True),
        include_cavity_surface=io_config.get('include_cavity_surface', False),
        include_protein_surface=io_config.get('include_protein_surface', False),
        protein_surface_max_distance=io_config.get('protein_surface_max_distance', 5.0)
    )
    
    if verbose:
        print(f"✓ Extracted {len(probes)} total probes")
        print(f"  Cavity probes: {np.sum(probes.sources == 'cavity_interior')}")
        print(f"  Cavity surface probes: {np.sum(probes.sources == 'cavity_surface')}")
        print(f"  Protein surface probes: {np.sum(probes.sources == 'protein_surface')}")
        print()
    
    # Check for centroid mode
    use_cavity_centroids = io_config.get('use_cavity_centroids', False)
    
    if use_cavity_centroids:
        # Validate compatibility
        if io_config.get('include_protein_surface', False):
            raise ValueError(
                "use_cavity_centroids=true is incompatible with include_protein_surface=true. "
                "Protein surface points have no cavity IDs and cannot be used for centroid computation."
            )
        
        if verbose:
            print("STEP 4: Computing cavity centroids (centroid mode enabled)")
            print("-"*70)
        
        # Compute centroid for each cavity
        unique_cavity_ids = np.unique(probes.cavity_ids[probes.cavity_ids > 0])
        
        if len(unique_cavity_ids) == 0:
            if verbose:
                print("⚠ No cavities found, returning empty result")
            final_probes = ProbeSet(
                positions=np.array([]).reshape(0, 3),
                sources=np.array([]),
                cavity_ids=np.array([]),
                grid_indices=np.array([]).reshape(0, 3)
            )
        else:
            centroids = []
            centroid_cavity_ids = []
            centroid_sources = []
            centroid_grid_indices = []
            
            for cav_id in unique_cavity_ids:
                # Get all probes for this cavity
                mask = probes.cavity_ids == cav_id
                cavity_probes = probes.positions[mask]
                
                # Compute centroid
                centroid = cavity_probes.mean(axis=0)
                centroids.append(centroid)
                centroid_cavity_ids.append(cav_id)
                centroid_sources.append('cavity_interior')
                # Use (0, 0, 0) as placeholder grid indices for centroids
                centroid_grid_indices.append([0, 0, 0])
                
                if verbose:
                    print(f"  Cavity {cav_id}: {len(cavity_probes)} probes → centroid at ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
            
            final_probes = ProbeSet(
                positions=np.array(centroids),
                sources=np.array(centroid_sources),
                cavity_ids=np.array(centroid_cavity_ids),
                grid_indices=np.array(centroid_grid_indices)
            )
            
            if verbose:
                print(f"\n✓ Generated {len(final_probes)} cavity centroids")
                print()
        
        all_results = []  # No filter results in centroid mode
    else:
        # Step 4: Configure and run filter pipeline
        if verbose:
            print("STEP 4: Running filter pipeline")
            print("-"*70)
            if save_intermediates:
                print("(Intermediate PDB files will be saved)")
            print()
        
        # Configure filters from YAML
        dist_config = config.get('distance_filter', {})
        coord_config = config.get('coordination_filter', {})
        hsab_config = config.get('hsab_filter', {})
        cluster_config = config.get('clustering', {})
        perf_config = config.get('performance', {})
        
        # Create filter instances
        distance_filter = DistanceFilter(
            min_distance=dist_config.get('min_coordination_distance', 1.8),
            max_distance=dist_config.get('max_coordination_distance', 3.5),
            use_kdtree=perf_config.get('use_kdtree', True),
            use_gpu=perf_config.get('use_gpu', False),
            batch_size=perf_config.get('batch_size', 10000)
        )
        
        coordination_filter = CoordinationFilter(
            coordination_radius=coord_config.get('coordination_radius', 2.5),
            min_coordination=coord_config.get('min_coordination_number', 3),
            max_coordination=coord_config.get('max_coordination_number', 6),
            allowed_donor_atoms=dist_config.get('allowed_donor_atoms', None),
            use_kdtree=perf_config.get('use_kdtree', True),
            check_occlusion=coord_config.get('check_occlusion', True),
            occlusion_cone_angle=coord_config.get('occlusion_cone_angle', 30.0),
            occlusion_vdw_scale=coord_config.get('occlusion_vdw_scale', 1.0)
        )
        
        # HSAB filter (only if any criteria specified)
        hsab_filter = None
        if (hsab_config.get('min_hard_donors') is not None or
            hsab_config.get('max_soft_donors') is not None or
            hsab_config.get('min_borderline_donors') is not None):
            hsab_filter = HardCoordinationFilter(
                min_hard_donors=hsab_config.get('min_hard_donors'),
                max_soft_donors=hsab_config.get('max_soft_donors'),
                min_borderline_donors=hsab_config.get('min_borderline_donors')
            )
        
        # Signature deduplicator
        deduplicator = SignatureDeduplicator(
            selection_method=cluster_config.get('selection_method', 'centroid'),
            distance_threshold=cluster_config.get('distance_threshold', 0.3),
            min_cluster_size=cluster_config.get('min_cluster_size', 1)
        )
        
        # Run the pipeline
        final_probes, all_results = run_filter_pipeline(
            probes=probes,
            protein_atoms=protein_atoms,
            atom_names=atom_names,
            atom_types=atom_types,
            residue_names=residue_names,
            is_backbone=is_backbone,
            distance_filter=distance_filter,
            coordination_filter=coordination_filter,
            hsab_filter=hsab_filter,
            deduplicator=deduplicator,
            verbose=verbose,
            save_intermediates=save_intermediates,
            output_prefix=output_prefix,
            protein_pdb=pdb_file if save_intermediates else None
        )
    
    # Step 5: Save final output
    if verbose:
        print()
        print("="*70)
        print("SAVING RESULTS")
        print("="*70)
    
    output_config = config.get('output', {})
    
    # Save combined PDB (probes + protein)
    if output_config.get('export_pdb', True):
        output_file = f"{output_prefix}_final.pdb"
        final_probes.to_pdb_with_protein(output_file, pdb_file)
        if verbose:
            print(f"✓ Saved combined PDB: {output_file}")
    
    # Save probes-only PDB
    probes_only_file = f"{output_prefix}_probes.pdb"
    final_probes.to_pdb(probes_only_file, atom_name=output_config.get('metal_symbol', 'M'))
    if verbose:
        print(f"✓ Saved probes PDB: {probes_only_file}")
    
    # Summary
    if verbose:
        print()
        print("="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        
        if use_cavity_centroids:
            print(f"Initial probes:              {len(probes)}")
            print(f"Cavity centroids generated:  {len(final_probes)}")
        else:
            print(f"Initial probes:              {len(probes)}")
            
            for i, result in enumerate(all_results, 1):
                filter_name = ["Distance", "Coordination", "HSAB", "Deduplicator"][i-1] if i <= 4 else f"Filter {i}"
                print(f"After {filter_name:20s}: {result.metadata['n_output']:6d} "
                      f"(rejected {result.metadata['n_rejected']:6d}, {result.metadata['rejection_rate']*100:5.1f}%)")
            
            print(f"\nFinal metal binding sites:   {len(final_probes)}")
            
            if len(probes) > 0:
                print(f"Overall retention:           {len(final_probes)/len(probes)*100:.2f}%")
    
    return {
        'probes': final_probes,
        'results': all_results,
        'n_sites': len(final_probes)
    }


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="MetalFinder: Metal binding site prediction using pyKVFinder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  %(prog)s protein.pdb -c metal_config.yaml
  
  # Save intermediate filter stages
  %(prog)s protein.pdb -c metal_config.yaml --save-intermediates
  
  # Specify custom output prefix
  %(prog)s protein.pdb -c metal_config.yaml -o my_protein_metal
  
  # Quiet mode (minimal output)
  %(prog)s protein.pdb -c metal_config.yaml --quiet
"""
    )
    
    parser.add_argument(
        'pdb',
        help='Input PDB file'
    )
    
    parser.add_argument(
        '-c', '--config',
        required=True,
        help='YAML configuration file (e.g., metal_config.yaml)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='metalfinder',
        help='Output prefix for result files (default: metalfinder)'
    )
    
    parser.add_argument(
        '--save-intermediates',
        action='store_true',
        help='Save PDB files at each filter stage for debugging'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='MetalFinder 1.0.0 (pyKVFinder plugin)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.pdb).exists():
        print(f"Error: PDB file not found: {args.pdb}", file=sys.stderr)
        return 1
    
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config file: {e}", file=sys.stderr)
        return 1
    
    # Run pipeline
    try:
        results = run_metalfinder(
            pdb_file=args.pdb,
            config=config,
            output_prefix=args.output,
            save_intermediates=args.save_intermediates,
            verbose=not args.quiet
        )
        
        if not args.quiet:
            print()
            print("✓ MetalFinder completed successfully!")
            print(f"  Found {results['n_sites']} potential metal binding sites")
        
        return 0
        
    except Exception as e:
        print(f"Error running MetalFinder: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
