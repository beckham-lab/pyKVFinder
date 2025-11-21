#!/usr/bin/env python3
"""
Optuna optimization for MetalFinder parameters.

Objective: Minimize total probe count while maintaining at least one probe
within 4 Å of TYR 137 CA atom.
"""

import optuna
import yaml
import numpy as np
from pathlib import Path
import tempfile
import os

from pyKVFinder.metalfinder.cli import run_metalfinder
from pyKVFinder.metalfinder.pdb_parser import parse_pdb


# Target: TYR 137 OH (hydroxyl oxygen) coordinates from inputs/PedS1_full.pdb
TYR137_OH = np.array([46.974, -5.741, -85.256])
TARGET_DISTANCE = 4.0  # Angstroms


def load_base_config():
    """Load base configuration from metal_config.yaml."""
    with open('metal_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['default']


def objective(trial):
    """Optuna objective function.
    
    Score = total_probes + penalty
    where penalty = 10000 if no probe within 4Å of TYR137
    
    Lower score is better.
    """
    
    # Suggest hyperparameters
    coord_radius = trial.suggest_float('coordination_radius', 3.0, 5.0)
    min_coord = trial.suggest_int('min_coordination_number', 2, 6)
    occlusion_angle = trial.suggest_float('occlusion_cone_angle', 15.0, 45.0)
    min_hard = trial.suggest_int('min_hard_donors', 0, 4)
    dist_threshold = trial.suggest_float('distance_threshold', 0.5, 0.95)
    min_cluster_size = trial.suggest_int('min_cluster_size', 1, 3)
    
    # Build config
    config = load_base_config()
    config['coordination_filter']['coordination_radius'] = coord_radius
    config['coordination_filter']['min_coordination_number'] = min_coord
    config['coordination_filter']['occlusion_cone_angle'] = occlusion_angle
    config['hsab_filter']['min_hard_donors'] = min_hard if min_hard > 0 else None
    config['clustering']['distance_threshold'] = dist_threshold
    config['clustering']['min_cluster_size'] = min_cluster_size
    
    # Run metalfinder with temporary output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_prefix = os.path.join(tmpdir, "optuna_run")
        
        try:
            run_metalfinder(
                pdb_file='inputs/PedS1_full.pdb',
                config=config,
                output_prefix=output_prefix,
                save_intermediates=False,
                verbose=False
            )
            
            # Read final probes
            probes_file = f"{output_prefix}_probes.pdb"
            if not os.path.exists(probes_file):
                print(f"  [Trial {trial.number}] FAIL: Probes file not created")
                trial.set_user_attr('failure_reason', 'no_probes_file')
                return 10000  # Failed run penalty
            
            # Parse probe coordinates
            probe_coords = []
            with open(probes_file, 'r') as f:
                for line in f:
                    if line.startswith('HETATM') or line.startswith('ATOM'):
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        probe_coords.append([x, y, z])
            
            if len(probe_coords) == 0:
                print(f"  [Trial {trial.number}] FAIL: No probes survived filtering (too strict)")
                trial.set_user_attr('failure_reason', 'zero_probes_after_filtering')
                return 10000  # No probes found
            
            probe_coords = np.array(probe_coords)
            
            # Calculate distances to TYR137
            distances = np.linalg.norm(probe_coords - TYR137_OH, axis=1)
            min_dist = np.min(distances)
            n_within_target = np.sum(distances <= TARGET_DISTANCE)
            total_probes = len(probe_coords)
            
            # Scoring
            if n_within_target == 0:
                # No probe near target - very bad
                print(f"  [Trial {trial.number}] FAIL: No probe within {TARGET_DISTANCE}Å of TYR137 (min dist={min_dist:.2f}Å)")
                trial.set_user_attr('failure_reason', 'no_probe_near_target')
                penalty = 10000
            elif n_within_target > 1:
                # Multiple probes near target - penalize extras
                print(f"  [Trial {trial.number}] OK: {n_within_target} probes near target, {total_probes} total")
                penalty = (n_within_target - 1) * 50
            else:
                # Exactly 1 probe near target - ideal
                print(f"  [Trial {trial.number}] EXCELLENT: 1 probe near target, {total_probes} total")
                penalty = 0
            
            score = total_probes + penalty
            
            # Log additional metrics
            trial.set_user_attr('total_probes', total_probes)
            trial.set_user_attr('min_distance_to_tyr137', float(min_dist))
            trial.set_user_attr('n_near_tyr137', int(n_within_target))
            
            return score
            
        except Exception as e:
            print(f"  [Trial {trial.number}] ERROR: {e}")
            trial.set_user_attr('failure_reason', f'exception: {str(e)[:100]}')
            return 10000


def main():
    """Run optimization."""
    import argparse
    parser = argparse.ArgumentParser(description='Optimize MetalFinder parameters')
    parser.add_argument('--resume', action='store_true', help='Resume existing study')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials')
    args = parser.parse_args()
    
    print("="*70)
    print("METALFINDER PARAMETER OPTIMIZATION")
    print("="*70)
    print(f"Target: TYR 137 OH at {TYR137_OH}")
    print(f"Distance threshold: {TARGET_DISTANCE} Å")
    print()
    
    # Create or load study
    storage_name = "sqlite:///metalfinder_optuna.db"
    
    if args.resume:
        print(f"Resuming existing study from {storage_name}")
        study = optuna.load_study(
            study_name='metalfinder_opt',
            storage=storage_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
    else:
        print(f"Creating new study (overwriting any existing)")
        try:
            optuna.delete_study(study_name='metalfinder_opt', storage=storage_name)
        except:
            pass  # Study doesn't exist yet
        study = optuna.create_study(
            direction='minimize',
            study_name='metalfinder_opt',
            storage=storage_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
    
    print()
    
    # Optimize
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    # Results
    print()
    print("="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.1f}")
    print()
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key:30s}: {value}")
    print()
    print("Best trial metrics:")
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key:30s}: {value}")
    
    # Save best config
    best_config = load_base_config()
    best_config['coordination_filter']['coordination_radius'] = study.best_params['coordination_radius']
    best_config['coordination_filter']['min_coordination_number'] = study.best_params['min_coordination_number']
    best_config['coordination_filter']['occlusion_cone_angle'] = study.best_params['occlusion_cone_angle']
    min_hard = study.best_params['min_hard_donors']
    best_config['hsab_filter']['min_hard_donors'] = min_hard if min_hard > 0 else None
    best_config['clustering']['distance_threshold'] = study.best_params['distance_threshold']
    best_config['clustering']['min_cluster_size'] = study.best_params['min_cluster_size']
    
    output_file = 'metal_config_optimized.yaml'
    with open(output_file, 'w') as f:
        yaml.dump({'default': best_config}, f, default_flow_style=False, sort_keys=False)
    
    print()
    print(f"✓ Saved optimized config to: {output_file}")


if __name__ == '__main__':
    main()
