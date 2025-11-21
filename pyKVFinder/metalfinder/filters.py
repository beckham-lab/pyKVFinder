"""
Filter Module

Provides progressive filtering for metal binding site identification:
1. DistanceFilter - Filter by distance to protein atoms
2. CoordinationFilter - Filter by coordination number and donor atom types
3. HardCoordinationFilter - Filter by HSAB hardness/softness
4. SignatureDeduplicator - Remove duplicate sites with same coordination signature

All filters inherit from BaseFilter abstract class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Set, FrozenSet
import numpy as np
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import fclusterdata

try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        TORCH_DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        TORCH_DEVICE = torch.device('mps')
    else:
        TORCH_DEVICE = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_DEVICE = None

from .probe_converter import ProbeSet


@dataclass
class FilterResult:
    """Result of a filter operation.
    
    Attributes
    ----------
    probes : ProbeSet
        Filtered probe set
    mask : np.ndarray
        Boolean mask indicating which probes passed (length = input probe count)
    metadata : dict
        Filter-specific metadata
    """
    probes: ProbeSet
    mask: np.ndarray
    metadata: dict


class BaseFilter(ABC):
    """Abstract base class for all filters.
    
    All filters must implement the filter() method which takes a ProbeSet
    and returns a FilterResult.
    """
    
    @abstractmethod
    def filter(self, probes: ProbeSet, **kwargs) -> FilterResult:
        """Apply filter to probe set.
        
        Parameters
        ----------
        probes : ProbeSet
            Input probe set
        **kwargs
            Filter-specific parameters
            
        Returns
        -------
        FilterResult
            Filtered probes with mask and metadata
        """
        pass


class DistanceFilter(BaseFilter):
    """Filter probes by distance to protein atoms.
    
    Applies dual distance criteria:
    1. No atom too close (steric clash prevention)
    2. At least one atom within coordination range
    
    Parameters
    ----------
    min_distance : float
        Minimum allowed distance to any protein atom (Ångströms)
    max_distance : float
        Maximum distance for coordination (Ångströms)
    use_kdtree : bool
        Use scipy KDTree for acceleration (recommended)
    use_gpu : bool
        Use GPU acceleration (requires PyTorch)
    batch_size : int
        Batch size for GPU processing
    """
    
    def __init__(
        self,
        min_distance: float = 1.8,
        max_distance: float = 3.5,
        use_kdtree: bool = True,
        use_gpu: bool = False,
        batch_size: int = 10000
    ):
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.use_kdtree = use_kdtree
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.batch_size = batch_size
        
        if use_gpu and not TORCH_AVAILABLE:
            print("Warning: GPU requested but PyTorch not available, falling back to CPU")
            self.use_gpu = False
    
    def filter(
        self,
        probes: ProbeSet,
        protein_atoms: np.ndarray,
        **kwargs
    ) -> FilterResult:
        """Filter probes by distance to protein atoms.
        
        Parameters
        ----------
        probes : ProbeSet
            Input probe set
        protein_atoms : np.ndarray
            (M, 3) protein atom coordinates in Ångströms
            
        Returns
        -------
        FilterResult
            Filtered probes with distance metadata
        """
        print(f"  Distance filter: processing {len(probes)} probes against {len(protein_atoms)} atoms")
        
        if self.use_gpu:
            print("  Using GPU acceleration...")
            mask, nearest_distances = self._filter_gpu(probes.positions, protein_atoms)
        elif self.use_kdtree:
            print("  Using KDTree acceleration...")
            mask, nearest_distances = self._filter_kdtree(probes.positions, protein_atoms)
        else:
            print("  Using naive distance computation...")
            mask, nearest_distances = self._filter_naive(probes.positions, protein_atoms)
        
        filtered_probes = probes.filter_by_mask(mask)
        
        metadata = {
            'n_input': len(probes),
            'n_output': len(filtered_probes),
            'n_rejected': len(probes) - len(filtered_probes),
            'rejection_rate': 1.0 - (len(filtered_probes) / len(probes)) if len(probes) > 0 else 0.0,
            'nearest_distances': nearest_distances[mask],
            'min_distance_threshold': self.min_distance,
            'max_distance_threshold': self.max_distance
        }
        
        return FilterResult(
            probes=filtered_probes,
            mask=mask,
            metadata=metadata
        )
    
    def _filter_kdtree(
        self,
        probe_positions: np.ndarray,
        protein_atoms: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter using KDTree for efficient nearest neighbor search."""
        print(f"    Building KDTree for {len(protein_atoms)} protein atoms...")
        tree = cKDTree(protein_atoms)
        
        print(f"    Querying nearest neighbors for {len(probe_positions)} probes...")
        nearest_distances, _ = tree.query(probe_positions, k=1)
        
        print(f"    Applying distance criteria: {self.min_distance} Å ≤ d ≤ {self.max_distance} Å")
        # Apply dual distance criteria
        mask = (nearest_distances >= self.min_distance) & (nearest_distances <= self.max_distance)
        
        n_pass = np.sum(mask)
        print(f"    Distance filter result: {n_pass}/{len(probe_positions)} probes passed")
        
        return mask, nearest_distances
    
    def _filter_naive(
        self,
        probe_positions: np.ndarray,
        protein_atoms: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter using naive pairwise distance computation."""
        n_probes = len(probe_positions)
        nearest_distances = np.zeros(n_probes)
        
        for i in range(n_probes):
            distances = np.linalg.norm(protein_atoms - probe_positions[i], axis=1)
            nearest_distances[i] = distances.min()
        
        mask = (nearest_distances >= self.min_distance) & (nearest_distances <= self.max_distance)
        
        return mask, nearest_distances
    
    def _filter_gpu(
        self,
        probe_positions: np.ndarray,
        protein_atoms: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter using GPU-accelerated distance computation."""
        n_probes = len(probe_positions)
        nearest_distances = np.zeros(n_probes)
        
        # Convert to torch tensors
        atoms_gpu = torch.from_numpy(protein_atoms).float().to(TORCH_DEVICE)
        
        # Process in batches
        for i in range(0, n_probes, self.batch_size):
            end_idx = min(i + self.batch_size, n_probes)
            batch = probe_positions[i:end_idx]
            batch_gpu = torch.from_numpy(batch).float().to(TORCH_DEVICE)
            
            # Compute pairwise distances
            distances = torch.cdist(batch_gpu, atoms_gpu, p=2)
            min_distances = distances.min(dim=1).values
            
            nearest_distances[i:end_idx] = min_distances.cpu().numpy()
        
        mask = (nearest_distances >= self.min_distance) & (nearest_distances <= self.max_distance)
        
        return mask, nearest_distances


@dataclass
class CoordinationInfo:
    """Coordination information for a probe.
    
    Attributes
    ----------
    probe_index : int
        Index in original ProbeSet
    coordination_number : int
        Number of coordinating atoms
    coordinating_atom_indices : np.ndarray
        Indices of coordinating atoms in protein
    coordinating_types : np.ndarray
        Element symbols of coordinating atoms
    distances : np.ndarray
        Distances to coordinating atoms
    """
    probe_index: int
    coordination_number: int
    coordinating_atom_indices: np.ndarray
    coordinating_types: np.ndarray
    distances: np.ndarray


class CoordinationFilter(BaseFilter):
    """Filter probes by coordination number and donor atom types.
    
    Uses chemical knowledge to identify valid metal-coordinating donors:
    - Backbone O: YES (carbonyl oxygen is a common donor)
    - Backbone N: NO (amide nitrogen rarely coordinates metals)
    - Sidechain donors: context-dependent (ASP/GLU O, HIS N, CYS S, etc.)
    
    Optionally performs geometric occlusion testing to exclude donors where
    another atom blocks the line-of-sight between probe and donor.
    
    Parameters
    ----------
    coordination_radius : float
        Radius for coordination sphere (Ångströms). Default: 2.5
    min_coordination : int
        Minimum coordination number. Default: 3
    max_coordination : int
        Maximum coordination number. Default: 6
    allowed_donor_atoms : Optional[List[str]]
        Allowed donor atom types (e.g., ['N', 'O', 'S']). None = all allowed.
        This is applied AFTER context-based filtering. Default: None
    use_kdtree : bool
        Use KDTree for ball query acceleration. Default: True
    check_occlusion : bool
        Enable geometric occlusion testing. When True, donors are rejected if
        another atom blocks the line-of-sight. Default: False
    occlusion_cone_angle : float
        Half-angle of occlusion cone in degrees. Atoms within this cone from
        the donor's perspective block coordination. Typical range: 20-45°.
        Default: 30.0
    occlusion_vdw_scale : float
        Scaling factor for Van der Waals radii in occlusion test. Values > 1.0
        make occlusion more strict, < 1.0 more lenient. Default: 1.0
    
    Examples
    --------
    >>> # Basic coordination filter
    >>> filter1 = CoordinationFilter(min_coordination=4, max_coordination=6)
    
    >>> # With occlusion checking enabled
    >>> filter2 = CoordinationFilter(
    ...     min_coordination=3,
    ...     check_occlusion=True,
    ...     occlusion_cone_angle=30.0,  # 30° cone
    ...     occlusion_vdw_scale=1.0     # Use standard VdW radii
    ... )
    
    >>> # Stricter occlusion (larger VdW radii, narrower cone)
    >>> filter3 = CoordinationFilter(
    ...     check_occlusion=True,
    ...     occlusion_cone_angle=20.0,   # Narrower cone
    ...     occlusion_vdw_scale=1.2      # 20% larger VdW radii
    ... )
    """
    
    # Valid metal-coordinating donors by (atom_name, residue_name) pairs
    # This provides precise control over which specific atoms can coordinate metals.
    # For example, this excludes Arg NE (which is not a good donor) while including
    # other nitrogen atoms that do coordinate.
    VALID_DONORS = {
        # Backbone carbonyl oxygen (NOT backbone nitrogen)
        ('O', 'backbone'),
        
        # Aspartate - carboxylate oxygens
        ('OD1', 'ASP'), ('OD2', 'ASP'),
        
        # Glutamate - carboxylate oxygens
        ('OE1', 'GLU'), ('OE2', 'GLU'),
        
        # Serine - hydroxyl oxygen
        ('OG', 'SER'),
        
        # Threonine - hydroxyl oxygen
        ('OG1', 'THR'),
        
        # Tyrosine - hydroxyl oxygen
        ('OH', 'TYR'),
        
        # Asparagine - amide oxygen and nitrogen
        ('OD1', 'ASN'), ('ND2', 'ASN'),
        
        # Glutamine - amide oxygen and nitrogen
        ('OE1', 'GLN'), ('NE2', 'GLN'),
        
        # Histidine - imidazole nitrogens (both tautomers)
        ('ND1', 'HIS'), ('NE2', 'HIS'),
        
        # Lysine - terminal amine
        ('NZ', 'LYS'),
        
        # Arginine - terminal guanidinium nitrogens (NH1, NH2)
        # NOTE: NE is excluded as it's not a good metal coordinator
        ('NH1', 'ARG'), ('NH2', 'ARG'),
        
        # Tryptophan - indole nitrogen
        ('NE1', 'TRP'),
        
        # Cysteine - thiol sulfur
        ('SG', 'CYS'),
        
        # Methionine - thioether sulfur
        ('SD', 'MET'),
        
        # Selenocysteine - selenium
        ('SE', 'SEC'),
        
        # Water - oxygen
        ('O', 'HOH'), ('O', 'WAT'), ('O', 'H2O'),
    }
    
    def __init__(
        self,
        coordination_radius: float = 2.5,
        min_coordination: int = 3,
        max_coordination: int = 6,
        allowed_donor_atoms: Optional[List[str]] = None,
        use_kdtree: bool = True,
        check_occlusion: bool = True,
        occlusion_cone_angle: float = 30.0,
        occlusion_vdw_scale: float = 1.0
    ):
        self.coordination_radius = coordination_radius
        self.min_coordination = min_coordination
        self.max_coordination = max_coordination
        self.allowed_donor_atoms = set(allowed_donor_atoms) if allowed_donor_atoms else None
        self.use_kdtree = use_kdtree
        self.check_occlusion = check_occlusion
        self.occlusion_cone_angle = np.deg2rad(occlusion_cone_angle)  # Convert to radians
        self.occlusion_vdw_scale = occlusion_vdw_scale
        
        # Van der Waals radii for common elements (Ångströms)
        self.vdw_radii = {
            'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
            'S': 1.80, 'P': 1.80, 'F': 1.47, 'Cl': 1.75,
            'Br': 1.85, 'I': 1.98, 'Se': 1.90, 'B': 1.92,
            'Si': 2.10, 'As': 1.85, 'Te': 2.06, 'default': 1.70
        }
    
    def is_valid_donor(
        self,
        atom_name: str,
        residue: str,
        is_backbone: bool
    ) -> bool:
        """Check if an atom is a valid metal-coordinating donor.
        
        Parameters
        ----------
        atom_name : str
            PDB atom name (e.g., 'OD1', 'NE2', 'SG')
        residue : str
            Residue name (3-letter code)
        is_backbone : bool
            Whether atom is part of backbone
            
        Returns
        -------
        bool
            True if atom can coordinate metals
        """
        if is_backbone:
            # Backbone context: only carbonyl O is a valid donor
            return (atom_name, 'backbone') in self.VALID_DONORS
        else:
            # Sidechain context: check atom name + residue specific donors
            return (atom_name, residue) in self.VALID_DONORS
    
    def _get_vdw_radius(self, element: str) -> float:
        """Get Van der Waals radius for an element.
        
        Parameters
        ----------
        element : str
            Element symbol
            
        Returns
        -------
        float
            VdW radius in Ångströms
        """
        # Handle two-letter elements (e.g., 'Se', 'Cl')
        if len(element) > 1:
            element = element[0].upper() + element[1].lower()
        else:
            element = element.upper()
        
        return self.vdw_radii.get(element, self.vdw_radii['default'])
    
    def is_occluded(
        self,
        probe_pos: np.ndarray,
        donor_pos: np.ndarray,
        donor_idx: int,
        all_atoms: np.ndarray,
        atom_types: np.ndarray,
        kdtree: Optional[cKDTree] = None
    ) -> bool:
        """Check if line-of-sight from probe to donor is occluded by other atoms.
        
        Uses a geometric occlusion test:
        1. Find atoms near the probe-donor line segment
        2. For each atom, check if it:
           - Is between probe and donor (not the donor itself)
           - Is close enough to the line segment (within VdW radius)
           - Falls within an angular cone from the donor's perspective
        
        Parameters
        ----------
        probe_pos : np.ndarray
            (3,) probe position
        donor_pos : np.ndarray
            (3,) donor atom position
        donor_idx : int
            Index of donor atom (to exclude it from occlusion check)
        all_atoms : np.ndarray
            (N, 3) all protein atom positions
        atom_types : np.ndarray
            (N,) element symbols for each atom
        kdtree : cKDTree, optional
            Precomputed KDTree for efficiency
            
        Returns
        -------
        bool
            True if occluded, False if clear line-of-sight
        """
        # Vector from probe to donor
        probe_to_donor = donor_pos - probe_pos
        distance = np.linalg.norm(probe_to_donor)
        
        if distance < 1e-6:
            return False  # Same position, no occlusion
        
        direction = probe_to_donor / distance
        
        # Find atoms near the line segment using KDTree
        # Query a cylinder around the line with radius = max expected VdW + buffer
        if kdtree is not None:
            # Find atoms within a sphere centered at midpoint
            midpoint = (probe_pos + donor_pos) / 2
            search_radius = distance / 2 + 3.0  # Half length + buffer
            nearby_indices = kdtree.query_ball_point(midpoint, r=search_radius)
        else:
            nearby_indices = np.arange(len(all_atoms))
        
        # Check each nearby atom for occlusion
        for idx in nearby_indices:
            if idx == donor_idx:
                continue  # Skip the donor atom itself
            
            atom_pos = all_atoms[idx]
            atom_element = atom_types[idx]
            
            # Vector from probe to this atom
            probe_to_atom = atom_pos - probe_pos
            
            # Project atom onto probe-donor line
            # t = 0 at probe, t = 1 at donor
            t = np.dot(probe_to_atom, direction) / distance
            
            # Skip if atom is not between probe and donor
            # Allow small margin (0.1 Å before probe, 0.1 Å past donor)
            if t < -0.05 or t > 1.05:
                continue
            
            # Compute closest point on line segment to atom
            closest_point = probe_pos + direction * (t * distance)
            
            # Distance from atom to line
            dist_to_line = np.linalg.norm(atom_pos - closest_point)
            
            # Get VdW radius for this atom
            vdw_radius = self._get_vdw_radius(atom_element) * self.occlusion_vdw_scale
            
            # Check if atom intersects the line (within VdW radius)
            if dist_to_line > vdw_radius:
                continue
            
            # Angular cone test: check if atom blocks the view from donor
            # Compute angle between donor→probe and donor→atom vectors
            donor_to_probe = -probe_to_donor  # Reverse direction
            donor_to_atom = atom_pos - donor_pos
            
            # Normalize vectors
            norm_dtp = np.linalg.norm(donor_to_probe)
            norm_dta = np.linalg.norm(donor_to_atom)
            
            if norm_dtp < 1e-6 or norm_dta < 1e-6:
                continue
            
            cos_angle = np.dot(donor_to_probe, donor_to_atom) / (norm_dtp * norm_dta)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # If atom is within the cone angle, it's blocking
            if angle < self.occlusion_cone_angle:
                return True
        
        return False
    
    def filter(
        self,
        probes: ProbeSet,
        protein_atoms: np.ndarray,
        atom_names: np.ndarray,
        atom_types: np.ndarray,
        residue_names: np.ndarray,
        is_backbone: np.ndarray,
        **kwargs
    ) -> FilterResult:
        """Filter probes by coordination number and donor validity.
        
        Parameters
        ----------
        probes : ProbeSet
            Input probe set
        protein_atoms : np.ndarray
            (M, 3) protein atom coordinates
        atom_names : np.ndarray
            (M,) PDB atom names for each atom (e.g., 'OD1', 'NE2')
        atom_types : np.ndarray
            (M,) element symbols for each atom
        residue_names : np.ndarray
            (M,) residue names for each atom
        is_backbone : np.ndarray
            (M,) whether each atom is backbone
            
        Returns
        -------
        FilterResult
            Filtered probes with coordination metadata
        """
        print(f"  Coordination filter: analyzing {len(probes)} probes")
        print(f"    Coordination sphere radius: {self.coordination_radius} Å")
        print(f"    Required coordination number: {self.min_coordination}-{self.max_coordination}")
        if self.allowed_donor_atoms:
            print(f"    Allowed donor atoms: {self.allowed_donor_atoms}")
        print(f"    Using atom-name-based donor validation (e.g., Arg NH1/NH2: YES, Arg NE: NO)")
        if self.check_occlusion:
            print(f"    Occlusion checking ENABLED:")
            print(f"      Cone angle: {np.rad2deg(self.occlusion_cone_angle):.1f}°")
            print(f"      VdW scale: {self.occlusion_vdw_scale:.2f}")
        
        coordination_infos = []
        valid_indices = []
        n_occluded_donors = 0
        n_rejected_pre_occlusion = 0  # Probes rejected before occlusion check
        n_rejected_post_occlusion = 0  # Probes rejected after occlusion check
        
        if self.use_kdtree:
            print("    Building KDTree for coordination queries...")
            tree = cKDTree(protein_atoms)
        else:
            tree = None
        
        print("    Calculating coordination for each probe...")
        for i, probe_pos in enumerate(probes.positions):
            if (i + 1) % 1000 == 0 or i == 0:
                print(f"      Processing probe {i+1}/{len(probes.positions)}...")
            
            if self.use_kdtree:
                # Ball query for atoms within coordination sphere
                indices = tree.query_ball_point(probe_pos, r=self.coordination_radius)
            else:
                # Naive distance computation
                distances = np.linalg.norm(protein_atoms - probe_pos, axis=1)
                indices = np.where(distances <= self.coordination_radius)[0]
            
            if len(indices) == 0:
                continue
            
            # Ensure indices is a numpy array of integers
            indices = np.array(indices, dtype=np.int64)
            
            # Get coordinating atom information
            coord_atoms = protein_atoms[indices]
            coord_names = atom_names[indices]
            coord_types = atom_types[indices]
            coord_residues = residue_names[indices]
            coord_backbone = is_backbone[indices]
            coord_distances = np.linalg.norm(coord_atoms - probe_pos, axis=1)
            
            # Filter by atom-name-based donor validation
            valid_donor_mask = np.array([
                self.is_valid_donor(name, res, bb)
                for name, res, bb in zip(coord_names, coord_residues, coord_backbone)
            ], dtype=bool)
            
            indices = indices[valid_donor_mask]
            coord_types = coord_types[valid_donor_mask]
            coord_distances = coord_distances[valid_donor_mask]
            
            # Additional filter by allowed donor atoms (element-based)
            if self.allowed_donor_atoms is not None:
                allowed_mask = np.array([t in self.allowed_donor_atoms for t in coord_types], dtype=bool)
                indices = indices[allowed_mask]
                coord_types = coord_types[allowed_mask]
                coord_distances = coord_distances[allowed_mask]
            
            # Check coordination number BEFORE occlusion
            coordination_number_pre_occlusion = len(indices)
            
            # Check if probe passes coordination constraints BEFORE occlusion
            if not (self.min_coordination <= coordination_number_pre_occlusion <= self.max_coordination):
                n_rejected_pre_occlusion += 1
                continue
            
            # Occlusion check: filter out donors with blocked line-of-sight
            if self.check_occlusion and len(indices) > 0:
                non_occluded_mask = []
                for idx, donor_pos in zip(indices, protein_atoms[indices]):
                    is_blocked = self.is_occluded(
                        probe_pos, donor_pos, idx,
                        protein_atoms, atom_types, tree
                    )
                    non_occluded_mask.append(not is_blocked)
                    if is_blocked:
                        n_occluded_donors += 1
                
                non_occluded_mask = np.array(non_occluded_mask, dtype=bool)
                indices = indices[non_occluded_mask]
                coord_types = coord_types[non_occluded_mask]
                coord_distances = coord_distances[non_occluded_mask]
            
            coordination_number = len(indices)
            
            # Check coordination number constraints AFTER occlusion
            if self.min_coordination <= coordination_number <= self.max_coordination:
                valid_indices.append(i)
                coordination_infos.append(CoordinationInfo(
                    probe_index=i,
                    coordination_number=coordination_number,
                    coordinating_atom_indices=indices,
                    coordinating_types=coord_types,
                    distances=coord_distances
                ))
            else:
                # Probe had enough donors pre-occlusion but not post-occlusion
                if self.check_occlusion:
                    n_rejected_post_occlusion += 1
        
        print(f"    Found {len(valid_indices)} probes with valid coordination")
        if self.check_occlusion:
            print(f"    Rejected {n_rejected_pre_occlusion} probes: insufficient donors BEFORE occlusion check")
            print(f"    Rejected {n_rejected_post_occlusion} probes: insufficient donors AFTER occlusion check")
            print(f"    Rejected {n_occluded_donors} individual donor atoms due to occlusion")
        else:
            print(f"    Rejected {n_rejected_pre_occlusion} probes: insufficient donors")
        
        # Create mask and filter probes
        mask = np.zeros(len(probes), dtype=bool)
        mask[valid_indices] = True
        filtered_probes = probes.filter_by_mask(mask)
        
        # Compute statistics
        if coordination_infos:
            coord_numbers = np.array([info.coordination_number for info in coordination_infos])
            print(f"    Coordination number range: {coord_numbers.min()}-{coord_numbers.max()} (mean: {coord_numbers.mean():.1f})")
        else:
            coord_numbers = np.array([])
        
        metadata = {
            'n_input': len(probes),
            'n_output': len(filtered_probes),
            'n_rejected': len(probes) - len(filtered_probes),
            'rejection_rate': 1.0 - (len(filtered_probes) / len(probes)) if len(probes) > 0 else 0.0,
            'coordination_infos': coordination_infos,
            'coordination_numbers': coord_numbers,
            'mean_coordination': coord_numbers.mean() if len(coord_numbers) > 0 else 0.0,
            'coordination_radius': self.coordination_radius,
            'min_coordination': self.min_coordination,
            'max_coordination': self.max_coordination,
            'occlusion_check_enabled': self.check_occlusion,
            'n_occluded_donors': n_occluded_donors if self.check_occlusion else 0
        }
        
        return FilterResult(
            probes=filtered_probes,
            mask=mask,
            metadata=metadata
        )


class HardCoordinationFilter(BaseFilter):
    """Filter probes by HSAB (Hard-Soft Acid-Base) coordination properties.
    
    Classifies coordinating atoms as hard, soft, or borderline donors based on
    element type and residue context, then filters by these properties.
    
    Parameters
    ----------
    min_hard_donors : Optional[int]
        Minimum number of hard donors required
    max_soft_donors : Optional[int]
        Maximum number of soft donors allowed
    min_borderline_donors : Optional[int]
        Minimum number of borderline donors required
    coordination_infos : List[CoordinationInfo]
        Coordination information from CoordinationFilter
    residue_names : np.ndarray
        Residue names for each atom
    is_backbone : np.ndarray
        Whether each atom is part of backbone
    """
    
    # HSAB classifications by (atom_name, residue)
    HARD_DONORS = {
        ('OD1', 'ASP'), ('OD2', 'ASP'),  # Aspartate carboxylate
        ('OE1', 'GLU'), ('OE2', 'GLU'),  # Glutamate carboxylate
        ('OG', 'SER'),  # Serine hydroxyl
        ('OG1', 'THR'),  # Threonine hydroxyl
        ('OH', 'TYR'),  # Tyrosine hydroxyl
        ('O', 'HOH'), ('O', 'WAT'), ('O', 'H2O'),  # Water
        ('O', 'backbone'),  # Backbone carbonyl
        ('NZ', 'LYS'),  # Lysine amine
    }
    
    SOFT_DONORS = {
        ('SG', 'CYS'),  # Cysteine thiol
        ('SD', 'MET'),  # Methionine thioether
        ('SE', 'SEC'),  # Selenocysteine
    }
    
    BORDERLINE_DONORS = {
        ('ND1', 'HIS'), ('NE2', 'HIS'),  # Histidine imidazole
        ('ND2', 'ASN'),  # Asparagine amide
        ('NE2', 'GLN'),  # Glutamine amide
        ('NE1', 'TRP'),  # Tryptophan indole
        ('NH1', 'ARG'), ('NH2', 'ARG'),  # Arginine guanidinium (NOT NE)
    }
    
    def __init__(
        self,
        min_hard_donors: Optional[int] = None,
        max_soft_donors: Optional[int] = None,
        min_borderline_donors: Optional[int] = None
    ):
        self.min_hard_donors = min_hard_donors
        self.max_soft_donors = max_soft_donors
        self.min_borderline_donors = min_borderline_donors
    
    def classify_atom(
        self,
        atom_name: str,
        residue: str,
        is_backbone: bool = False
    ) -> str:
        """Classify atom as hard, soft, or borderline.
        
        Parameters
        ----------
        atom_name : str
            PDB atom name (e.g., 'OD1', 'NE2')
        residue : str
            Residue name (3-letter code)
        is_backbone : bool
            Whether atom is backbone
            
        Returns
        -------
        str
            'hard', 'soft', or 'borderline'
        """
        if is_backbone:
            key = (atom_name, 'backbone')
        else:
            key = (atom_name, residue)
        
        if key in self.HARD_DONORS:
            return 'hard'
        elif key in self.SOFT_DONORS:
            return 'soft'
        elif key in self.BORDERLINE_DONORS:
            return 'borderline'
        else:
            # Default: classify by element (first character of atom name)
            # Extract element from atom name
            element = ''.join(c for c in atom_name if c.isalpha())
            if element and element[0] == 'O':
                return 'hard'
            elif element and element[0] in ('S', 'Se'):
                return 'soft'
            elif element and element[0] == 'N':
                return 'borderline'
            else:
                return 'borderline'
    
    def filter(
        self,
        probes: ProbeSet,
        coordination_infos: List[CoordinationInfo],
        atom_names: np.ndarray,
        residue_names: np.ndarray,
        is_backbone: np.ndarray,
        **kwargs
    ) -> FilterResult:
        """Filter probes by HSAB coordination properties.
        
        Parameters
        ----------
        probes : ProbeSet
            Input probe set (should match coordination_infos)
        coordination_infos : List[CoordinationInfo]
            Coordination information from CoordinationFilter
        atom_names : np.ndarray
            (M,) PDB atom names for each protein atom
        residue_names : np.ndarray
            (M,) residue names for each protein atom
        is_backbone : np.ndarray
            (M,) whether each atom is backbone
            
        Returns
        -------
        FilterResult
            Filtered probes with HSAB metadata
        """
        print(f"  HSAB filter: classifying {len(coordination_infos)} coordination spheres")
        if self.min_hard_donors is not None:
            print(f"    Min hard donors: {self.min_hard_donors}")
        if self.max_soft_donors is not None:
            print(f"    Max soft donors: {self.max_soft_donors}")
        if self.min_borderline_donors is not None:
            print(f"    Min borderline donors: {self.min_borderline_donors}")
        
        valid_indices = []
        hsab_infos = []
        
        print("    Classifying coordinating atoms by HSAB theory...")
        for idx, info in enumerate(coordination_infos):
            if (idx + 1) % 500 == 0 or idx == 0:
                print(f"      Processing coordination sphere {idx+1}/{len(coordination_infos)}...")
            
            # Classify each coordinating atom
            hardness_classes = []
            for atom_idx in info.coordinating_atom_indices:
                atom_name = atom_names[atom_idx]
                residue = residue_names[atom_idx]
                backbone = is_backbone[atom_idx]
                classification = self.classify_atom(atom_name, residue, backbone)
                hardness_classes.append(classification)
            
            # Count donor types
            hardness_array = np.array(hardness_classes)
            n_hard = np.sum(hardness_array == 'hard')
            n_soft = np.sum(hardness_array == 'soft')
            n_borderline = np.sum(hardness_array == 'borderline')
            
            # Apply filters
            passes = True
            if self.min_hard_donors is not None and n_hard < self.min_hard_donors:
                passes = False
            if self.max_soft_donors is not None and n_soft > self.max_soft_donors:
                passes = False
            if self.min_borderline_donors is not None and n_borderline < self.min_borderline_donors:
                passes = False
            
            if passes:
                # Store the index in the current probe list (idx), not info.probe_index
                valid_indices.append(idx)
                hsab_infos.append({
                    'probe_index': idx,  # Index in current filtered list
                    'n_hard': n_hard,
                    'n_soft': n_soft,
                    'n_borderline': n_borderline,
                    'hardness_classes': hardness_array
                })
        
        print(f"    HSAB filter passed: {len(valid_indices)}/{len(coordination_infos)} probes")
        if hsab_infos:
            n_hard_avg = np.mean([h['n_hard'] for h in hsab_infos])
            n_soft_avg = np.mean([h['n_soft'] for h in hsab_infos])
            n_border_avg = np.mean([h['n_borderline'] for h in hsab_infos])
            print(f"    Average donors - Hard: {n_hard_avg:.1f}, Soft: {n_soft_avg:.1f}, Borderline: {n_border_avg:.1f}")
        
        # Create mask and filter probes
        # valid_indices are 0-based positions in the coordination_infos list
        mask = np.zeros(len(probes), dtype=bool)
        if len(valid_indices) > 0:
            mask[valid_indices] = True
        filtered_probes = probes.filter_by_mask(mask)
        
        metadata = {
            'n_input': len(probes),
            'n_output': len(filtered_probes),
            'n_rejected': len(probes) - len(filtered_probes),
            'rejection_rate': 1.0 - (len(filtered_probes) / len(probes)) if len(probes) > 0 else 0.0,
            'hsab_infos': hsab_infos,
            'min_hard_donors': self.min_hard_donors,
            'max_soft_donors': self.max_soft_donors,
            'min_borderline_donors': self.min_borderline_donors
        }
        
        return FilterResult(
            probes=filtered_probes,
            mask=mask,
            metadata=metadata
        )


@dataclass
class CoordinationSignature:
    """Immutable coordination signature for deduplication.
    
    Attributes
    ----------
    atom_indices : frozenset
        Frozenset of coordinating atom indices
    coordination_number : int
        Number of coordinating atoms
    donor_elements : tuple
        Sorted tuple of donor element symbols
    """
    atom_indices: FrozenSet[int]
    coordination_number: int
    donor_elements: Tuple[str, ...]
    
    def __hash__(self) -> int:
        return hash((self.atom_indices, self.coordination_number, self.donor_elements))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, CoordinationSignature):
            return False
        return (self.atom_indices == other.atom_indices and
                self.coordination_number == other.coordination_number and
                self.donor_elements == other.donor_elements)


class SignatureDeduplicator(BaseFilter):
    """Remove duplicate probes with similar coordination signatures.
    
    Uses hierarchical clustering with complete linkage on coordination signatures.
    Complete linkage ensures every pair in a cluster has similarity ≥ threshold,
    preventing cluster drift where dissimilar probes join through intermediates.
    
    Parameters
    ----------
    selection_method : str
        How to select representative: 'centroid', 'first', or 'random'
        Default: 'centroid'
    distance_threshold : float
        Maximum dissimilarity distance [0, 1] to group probes together.
        distance = 1 - (Jaccard similarity of coordinating atom sets)
        Lower values = stricter clustering (more clusters)
        Default: 0.3 (requires 70% Jaccard similarity)
    min_cluster_size : int
        Minimum cluster size to keep (default: 1)
    """
    
    def __init__(
        self,
        selection_method: str = 'centroid',
        distance_threshold: float = 0.3,
        min_cluster_size: int = 1
    ):
        if selection_method not in ['centroid', 'first', 'random']:
            raise ValueError(f"selection_method must be 'centroid', 'first', or 'random', got {selection_method}")
        if not 0 <= distance_threshold <= 1:
            raise ValueError(f"distance_threshold must be in [0, 1], got {distance_threshold}")
        
        self.selection_method = selection_method
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
    
    def _compute_jaccard_distance(
        self,
        info1: CoordinationInfo,
        info2: CoordinationInfo
    ) -> float:
        """Compute Jaccard distance between two coordination spheres.
        
        Jaccard distance = 1 - Jaccard similarity = 1 - |A ∩ B| / |A ∪ B|
        
        This is a proper metric satisfying triangle inequality, unlike raw similarity.
        Range: [0, 1] where 0 = identical sets, 1 = completely disjoint sets
        
        Parameters
        ----------
        info1, info2 : CoordinationInfo
            Coordination information to compare
            
        Returns
        -------
        float
            Jaccard distance in [0, 1]
        """
        set1 = set(info1.coordinating_atom_indices)
        set2 = set(info2.coordinating_atom_indices)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0  # Both empty sets
        
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity
    
    def _compute_pairwise_distance_matrix(
        self,
        coordination_infos: List[CoordinationInfo]
    ) -> np.ndarray:
        """Compute pairwise Jaccard distance matrix for hierarchical clustering.
        
        Returns
        -------
        np.ndarray
            (N, N) symmetric distance matrix where distance is in [0, 1]
            Lower values = more similar coordination
        """
        n_probes = len(coordination_infos)
        distance_matrix = np.zeros((n_probes, n_probes))
        
        # Compute all pairwise Jaccard distances
        for i in range(n_probes):
            for j in range(i + 1, n_probes):
                distance = self._compute_jaccard_distance(
                    coordination_infos[i],
                    coordination_infos[j]
                )
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix
    
    def _hierarchical_cluster(
        self,
        coordination_infos: List[CoordinationInfo]
    ) -> Dict[int, List[int]]:
        """Cluster probes using complete-linkage hierarchical clustering.
        
        Complete linkage ensures every pair in a cluster has distance ≤ threshold,
        preventing chaining where dissimilar probes join through intermediates.
        
        Returns
        -------
        Dict[int, List[int]]
            Mapping from cluster representative index to list of member indices
        """
        n_probes = len(coordination_infos)
        
        # Handle edge cases
        if n_probes == 0:
            return {}
        if n_probes == 1:
            return {0: [0]}
        
        # Compute pairwise Jaccard distance matrix
        print("    Computing pairwise Jaccard distance matrix...")
        distance_matrix = self._compute_pairwise_distance_matrix(coordination_infos)
        
        # Debug: show distance statistics
        n_probes = len(coordination_infos)
        if n_probes > 1:
            triu_indices = np.triu_indices(n_probes, k=1)
            pairwise_distances = distance_matrix[triu_indices]
            print(f"    Distance statistics: min={pairwise_distances.min():.3f}, "
                  f"max={pairwise_distances.max():.3f}, mean={pairwise_distances.mean():.3f}, "
                  f"median={np.median(pairwise_distances):.3f}")
            n_within_threshold = np.sum(pairwise_distances <= self.distance_threshold)
            n_total = len(pairwise_distances)
            print(f"    Pairs within threshold: {n_within_threshold}/{n_total} ({100*n_within_threshold/n_total:.1f}%)")
        
        print(f"    Running complete-linkage clustering (threshold={self.distance_threshold:.3f})...")
        
        from scipy.spatial.distance import squareform
        from scipy.cluster.hierarchy import linkage, fcluster
        
        # Convert to condensed distance matrix
        condensed_dist = squareform(distance_matrix, checks=False)
        
        # Perform complete-linkage hierarchical clustering
        linkage_matrix = linkage(condensed_dist, method='complete')
        
        # Cut tree at distance threshold
        cluster_labels = fcluster(linkage_matrix, self.distance_threshold, criterion='distance')
        
        # Group probes by cluster label
        clusters_by_label: Dict[int, List[int]] = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters_by_label:
                clusters_by_label[label] = []
            clusters_by_label[label].append(idx)
        
        print(f"    Formed {len(clusters_by_label)} clusters")
        
        # Convert to representative-based format
        clusters: Dict[int, List[int]] = {}
        for label, member_indices in clusters_by_label.items():
            rep_idx = member_indices[0]  # Temporary, refined by selection_method later
            clusters[rep_idx] = member_indices
        
        return clusters
    
    def _cluster_probes(
        self,
        coordination_infos: List[CoordinationInfo]
    ) -> Dict[int, List[int]]:
        """Cluster probes by coordination signature similarity.
        
        Returns
        -------
        Dict[int, List[int]]
            Mapping from cluster representative index to list of member indices
        """
        return self._hierarchical_cluster(coordination_infos)
    
    def _compute_signature(self, info: CoordinationInfo) -> CoordinationSignature:
        """Compute coordination signature for a probe."""
        # Debug: check for duplicates in atom indices
        unique_indices = set(info.coordinating_atom_indices)
        if len(unique_indices) != len(info.coordinating_atom_indices):
            print(f"    WARNING: Duplicate atom indices detected!")
            print(f"      Original: {info.coordinating_atom_indices}")
            print(f"      Unique: {unique_indices}")
        
        return CoordinationSignature(
            atom_indices=frozenset(info.coordinating_atom_indices),
            coordination_number=info.coordination_number,
            donor_elements=tuple(sorted(info.coordinating_types))
        )
    
    def filter(
        self,
        probes: ProbeSet,
        coordination_infos: List[CoordinationInfo],
        residue_names: Optional[np.ndarray] = None,
        **kwargs
    ) -> FilterResult:
        """Deduplicate probes by coordination signature.
        
        Parameters
        ----------
        probes : ProbeSet
            Input probe set
        coordination_infos : List[CoordinationInfo]
            Coordination information for each probe
        residue_names : np.ndarray, optional
            Residue names for residue-based similarity (required if similarity_metric='residue')
            
        Returns
        -------
        FilterResult
            Deduplicated probes with cluster metadata
        """
        print(f"  Signature deduplicator: analyzing {len(probes)} probes")
        print(f"  Coordination infos provided: {len(coordination_infos)}")
        
        # Handle empty input
        if len(probes) == 0:
            print(f"    No probes to deduplicate - returning empty result")
            return FilterResult(
                probes=probes,
                mask=np.array([], dtype=bool),
                metadata={
                    'n_input': 0,
                    'n_output': 0,
                    'n_rejected': 0,
                    'rejection_rate': 0.0,
                    'n_clusters': 0
                }
            )
        
        # Validate that we have coordination info for each probe
        if len(coordination_infos) != len(probes):
            raise ValueError(
                f"Length mismatch: {len(probes)} probes but {len(coordination_infos)} coordination_infos. "
                "Each probe must have corresponding coordination information."
            )
        
        print(f"    Distance threshold: {self.distance_threshold:.3f} (Jaccard distance)")
        print(f"    Selection method: {self.selection_method}")
        if self.min_cluster_size > 1:
            print(f"    Minimum cluster size: {self.min_cluster_size}")
        
        # Cluster by coordination signature similarity
        signature_groups = self._cluster_probes(coordination_infos)
        print(f"    Found {len(signature_groups)} clusters")
        
        # Handle case where all clusters were filtered out
        if len(signature_groups) == 0:
            print(f"    All clusters were too small - returning empty result")
            return FilterResult(
                probes=probes.filter_by_mask(np.zeros(len(probes), dtype=bool)),
                mask=np.zeros(len(probes), dtype=bool),
                metadata={
                    'n_input': len(probes),
                    'n_output': 0,
                    'n_rejected': len(probes),
                    'rejection_rate': 1.0,
                    'n_clusters': 0
                }
            )
        
        # Show cluster size distribution
        sizes = [len(indices) for indices in signature_groups.values()]
        print(f"    Cluster size distribution: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
        if max(sizes) > 1:
            multi_clusters = sum(1 for s in sizes if s > 1)
            print(f"    Clusters with multiple probes: {multi_clusters}/{len(signature_groups)}")
        
        # Filter by minimum cluster size
        if self.min_cluster_size > 1:
            original_n_clusters = len(signature_groups)
            signature_groups = {k: v for k, v in signature_groups.items() if len(v) >= self.min_cluster_size}
            n_filtered = original_n_clusters - len(signature_groups)
            if n_filtered > 0:
                print(f"    Filtered {n_filtered} clusters with size < {self.min_cluster_size}")
        
        # Select representative from each group
        print(f"    Selecting representative from each group...")
        selected_indices = []
        cluster_sizes = []
        
        for sig, indices in signature_groups.items():
            cluster_sizes.append(len(indices))
            
            if self.selection_method == 'first':
                selected_indices.append(indices[0])
            elif self.selection_method == 'random':
                selected_indices.append(np.random.choice(indices))
            elif self.selection_method == 'centroid':
                # Compute centroid and find nearest probe
                # Validate indices are in range
                if max(indices) >= len(probes):
                    raise ValueError(f"Invalid index {max(indices)} for {len(probes)} probes")
                
                positions = probes.positions[indices]
                centroid = positions.mean(axis=0)
                distances = np.linalg.norm(positions - centroid, axis=1)
                nearest_idx = indices[np.argmin(distances)]
                selected_indices.append(nearest_idx)
        
        # Create mask
        print(f"    Selected {len(selected_indices)} representatives from {len(signature_groups)} groups")
        print(f"    Selected indices: {sorted(selected_indices)[:10]}..." if len(selected_indices) > 10 else f"    Selected indices: {sorted(selected_indices)}")
        
        mask = np.zeros(len(probes), dtype=bool)
        mask[selected_indices] = True
        filtered_probes = probes.filter_by_mask(mask)
        
        print(f"    Mask has {np.sum(mask)} True values")
        print(f"    Filtered probes: {len(filtered_probes)}")
        
        metadata = {
            'n_input': len(probes),
            'n_output': len(filtered_probes),
            'n_rejected': len(probes) - len(filtered_probes),
            'rejection_rate': 1.0 - (len(filtered_probes) / len(probes)) if len(probes) > 0 else 0.0,
            'n_clusters': len(signature_groups),
            'cluster_sizes': np.array(cluster_sizes),
            'mean_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0.0,
            'max_cluster_size': np.max(cluster_sizes) if cluster_sizes else 0,
            'selection_method': self.selection_method,
            'distance_threshold': self.distance_threshold,
            'min_cluster_size': self.min_cluster_size
        }
        
        return FilterResult(
            probes=filtered_probes,
            mask=mask,
            metadata=metadata
        )


def run_filter_pipeline(
    probes: ProbeSet,
    protein_atoms: np.ndarray,
    atom_names: np.ndarray,
    atom_types: np.ndarray,
    residue_names: np.ndarray,
    is_backbone: np.ndarray,
    distance_filter: Optional[DistanceFilter] = None,
    coordination_filter: Optional[CoordinationFilter] = None,
    hsab_filter: Optional[HardCoordinationFilter] = None,
    deduplicator: Optional[SignatureDeduplicator] = None,
    verbose: bool = True,
    save_intermediates: bool = False,
    output_prefix: str = "filter",
    protein_pdb: Optional[str] = None
) -> Tuple[ProbeSet, List[FilterResult]]:
    """Run the standard metal binding site filter pipeline.
    
    Applies filters in the standard order:
    1. Distance filter
    2. Coordination filter
    3. HSAB filter
    4. Signature deduplicator
    
    Filters can be skipped by passing None. The pipeline automatically handles
    passing coordination metadata between filters.
    
    Parameters
    ----------
    probes : ProbeSet
        Initial probe set from cavity detection
    protein_atoms : np.ndarray
        (M, 3) protein atom coordinates
    atom_names : np.ndarray
        (M,) PDB atom names for each atom (e.g., 'OD1', 'NE2')
    atom_types : np.ndarray
        (M,) element symbols for each atom
    residue_names : np.ndarray
        (M,) residue names for each atom
    is_backbone : np.ndarray
        (M,) whether each atom is backbone
    distance_filter : DistanceFilter, optional
        Distance filter instance (None to skip)
    coordination_filter : CoordinationFilter, optional
        Coordination filter instance (None to skip)
    hsab_filter : HardCoordinationFilter, optional
        HSAB filter instance (None to skip)
    deduplicator : SignatureDeduplicator, optional
        Deduplicator instance (None to skip)
    verbose : bool
        Print progress information
    save_intermediates : bool
        Save PDB files at each filter stage (default: False)
    output_prefix : str
        Prefix for intermediate PDB files (default: "filter")
    protein_pdb : str, optional
        Path to protein PDB file for combined output (required if save_intermediates=True)
        
    Returns
    -------
    final_probes : ProbeSet
        Probes that passed all filters
    results : List[FilterResult]
        Results from each filter stage (only includes filters that were run)
        
    Examples
    --------
    >>> # Run full pipeline with default parameters
    >>> final_probes, results = run_filter_pipeline(
    ...     probes, protein_atoms, atom_names, atom_types, residue_names, is_backbone,
    ...     distance_filter=DistanceFilter(),
    ...     coordination_filter=CoordinationFilter(),
    ...     hsab_filter=HardCoordinationFilter(min_hard_donors=2),
    ...     deduplicator=SignatureDeduplicator()
    ... )
    
    >>> # Skip HSAB filter
    >>> final_probes, results = run_filter_pipeline(
    ...     probes, protein_atoms, atom_names, atom_types, residue_names, is_backbone,
    ...     distance_filter=DistanceFilter(),
    ...     coordination_filter=CoordinationFilter(),
    ...     hsab_filter=None,
    ...     deduplicator=SignatureDeduplicator()
    ... )
    """
    if save_intermediates and protein_pdb is None:
        raise ValueError("protein_pdb must be provided when save_intermediates=True")
    
    results = []
    current_probes = probes
    coordination_infos = None
    stage_num = 0
    
    if verbose:
        print(f"Starting filter pipeline with {len(probes)} probes")
    
    # Save initial state
    if save_intermediates:
        filename = f"{output_prefix}_00_initial.pdb"
        if verbose:
            print(f"  Saving: {filename}")
        current_probes.to_pdb_with_protein(filename, protein_pdb)
    
    # Stage 1: Distance filter
    if distance_filter is not None:
        stage_num += 1
        if verbose:
            print(f"\nFilter {stage_num}: Distance Filter")
            print("-"*70)
        
        result = distance_filter.filter(current_probes, protein_atoms=protein_atoms)
        results.append(result)
        current_probes = result.probes
        
        if verbose:
            print(f"  Input: {result.metadata['n_input']} probes")
            print(f"  Output: {result.metadata['n_output']} probes")
            print(f"  Rejected: {result.metadata['n_rejected']} probes ({result.metadata['rejection_rate']*100:.1f}%)")
        
        if save_intermediates and len(current_probes) > 0:
            filename = f"{output_prefix}_{stage_num:02d}_distance.pdb"
            if verbose:
                print(f"  Saving: {filename}")
            current_probes.to_pdb_with_protein(filename, protein_pdb)
    
    # Stage 2: Coordination filter
    if coordination_filter is not None:
        stage_num += 1
        if verbose:
            print(f"\nFilter {stage_num}: Coordination Filter")
            print("-"*70)
        
        result = coordination_filter.filter(
            current_probes,
            protein_atoms=protein_atoms,
            atom_names=atom_names,
            atom_types=atom_types,
            residue_names=residue_names,
            is_backbone=is_backbone
        )
        results.append(result)
        current_probes = result.probes
        coordination_infos = result.metadata['coordination_infos']
        
        if verbose:
            print(f"  Input: {result.metadata['n_input']} probes")
            print(f"  Output: {result.metadata['n_output']} probes")
            print(f"  Rejected: {result.metadata['n_rejected']} probes ({result.metadata['rejection_rate']*100:.1f}%)")
        
        if save_intermediates and len(current_probes) > 0:
            filename = f"{output_prefix}_{stage_num:02d}_coordination.pdb"
            if verbose:
                print(f"  Saving: {filename}")
            current_probes.to_pdb_with_protein(filename, protein_pdb)
    
    # Stage 3: HSAB filter
    if hsab_filter is not None and len(current_probes) > 0:
        stage_num += 1
        if coordination_infos is None:
            raise ValueError("HSAB filter requires coordination filter to be run first")
        
        if verbose:
            print(f"\nFilter {stage_num}: HSAB Filter")
            print("-"*70)
        
        result = hsab_filter.filter(
            current_probes,
            coordination_infos=coordination_infos,
            atom_names=atom_names,
            residue_names=residue_names,
            is_backbone=is_backbone
        )
        results.append(result)
        current_probes = result.probes
        
        # Update coordination_infos to match filtered probes
        passed_indices = np.where(result.mask)[0]
        coordination_infos = [coordination_infos[i] for i in passed_indices]
        
        if verbose:
            print(f"  Input: {result.metadata['n_input']} probes")
            print(f"  Output: {result.metadata['n_output']} probes")
            print(f"  Rejected: {result.metadata['n_rejected']} probes ({result.metadata['rejection_rate']*100:.1f}%)")
        
        if save_intermediates and len(current_probes) > 0:
            filename = f"{output_prefix}_{stage_num:02d}_hsab.pdb"
            if verbose:
                print(f"  Saving: {filename}")
            current_probes.to_pdb_with_protein(filename, protein_pdb)
    elif hsab_filter is not None and len(current_probes) == 0:
        if verbose:
            print(f"\nFilter {stage_num + 1}: HSAB Filter")
            print("-"*70)
            print(f"  Skipping: no probes remaining from previous filter")
    
    # Stage 4: Signature deduplicator
    if deduplicator is not None and len(current_probes) > 0:
        stage_num += 1
        if coordination_infos is None:
            raise ValueError("Deduplicator requires coordination filter to be run first")
        
        if verbose:
            print(f"\nFilter {stage_num}: Signature Deduplicator")
            print("-"*70)
        
        result = deduplicator.filter(
            current_probes,
            coordination_infos=coordination_infos,
            residue_names=residue_names
        )
        results.append(result)
        current_probes = result.probes
        
        if verbose:
            print(f"  Input: {result.metadata['n_input']} probes")
            print(f"  Output: {result.metadata['n_output']} probes")
            print(f"  Rejected: {result.metadata['n_rejected']} probes ({result.metadata['rejection_rate']*100:.1f}%)")
        
        if save_intermediates and len(current_probes) > 0:
            filename = f"{output_prefix}_{stage_num:02d}_deduplicator.pdb"
            if verbose:
                print(f"  Saving: {filename}")
            current_probes.to_pdb_with_protein(filename, protein_pdb)
    elif deduplicator is not None and len(current_probes) == 0:
        if verbose:
            print(f"\nFilter {stage_num + 1}: Signature Deduplicator")
            print("-"*70)
            print(f"  Skipping: no probes remaining from previous filter")
    
    if verbose:
        print(f"\nPipeline complete: {len(probes)} → {len(current_probes)} probes")
        if len(probes) > 0:
            print(f"Overall retention: {len(current_probes)/len(probes)*100:.1f}%")
    
    return current_probes, results
