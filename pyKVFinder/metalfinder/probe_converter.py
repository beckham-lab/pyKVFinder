"""
Probe Converter Module

Converts pyKVFinder 3D grid representations to Cartesian coordinates,
extracting probe positions from cavity and protein surface grids.

Probe Types
-----------
- cavity_interior: Points inside cavities (not at cavity-protein interface)
- cavity_surface: Cavity points at the cavity-protein interface  
- protein_surface: All bulk solvent points (can be filtered downstream)
"""

from dataclasses import dataclass
from typing import Optional, List, Literal
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
    # Determine best available device
    if torch.cuda.is_available():
        TORCH_DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        TORCH_DEVICE = torch.device('mps')
    else:
        TORCH_DEVICE = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_DEVICE = None


@dataclass
class ProbeSet:
    """Container for probe positions with metadata.
    
    Attributes
    ----------
    positions : np.ndarray
        (N, 3) Cartesian coordinates in Ångströms
    sources : np.ndarray
        (N,) source labels: 'cavity_interior', 'cavity_surface', or 'protein_surface'
    cavity_ids : np.ndarray
        (N,) cavity ID for each probe (0 for protein surface probes)
    grid_indices : np.ndarray
        (N, 3) original grid indices (i, j, k)
    """
    
    positions: np.ndarray
    sources: np.ndarray
    cavity_ids: np.ndarray
    grid_indices: np.ndarray
    
    def __len__(self) -> int:
        """Return number of probes."""
        return len(self.positions)
    
    def __post_init__(self):
        """Validate array shapes."""
        n = len(self.positions)
        if self.positions.shape != (n, 3):
            raise ValueError(f"positions must be (N, 3), got {self.positions.shape}")
        if len(self.sources) != n:
            raise ValueError(f"sources must have length {n}, got {len(self.sources)}")
        if len(self.cavity_ids) != n:
            raise ValueError(f"cavity_ids must have length {n}, got {len(self.cavity_ids)}")
        if self.grid_indices.shape != (n, 3):
            raise ValueError(f"grid_indices must be (N, 3), got {self.grid_indices.shape}")
    
    def filter_by_mask(self, mask: np.ndarray) -> 'ProbeSet':
        """Return new ProbeSet with filtered probes.
        
        Parameters
        ----------
        mask : np.ndarray
            Boolean array of length N
            
        Returns
        -------
        ProbeSet
            New ProbeSet containing only probes where mask is True
        """
        if len(mask) != len(self):
            raise ValueError(f"mask length {len(mask)} doesn't match probe count {len(self)}")
        
        return ProbeSet(
            positions=self.positions[mask],
            sources=self.sources[mask],
            cavity_ids=self.cavity_ids[mask],
            grid_indices=self.grid_indices[mask]
        )
    
    def filter_by_source(self, source: Literal['cavity_interior', 'cavity_surface', 'protein_surface']) -> 'ProbeSet':
        """Filter probes by source type.
        
        Parameters
        ----------
        source : {'cavity_interior', 'cavity_surface', 'protein_surface'}
            Source type to keep
            
        Returns
        -------
        ProbeSet
            Filtered probe set
        """
        mask = self.sources == source
        return self.filter_by_mask(mask)
    
    def filter_by_cavity(self, cavity_ids: List[int]) -> 'ProbeSet':
        """Filter probes by cavity ID.
        
        Parameters
        ----------
        cavity_ids : list of int
            Cavity IDs to keep
            
        Returns
        -------
        ProbeSet
            Filtered probe set
        """
        mask = np.isin(self.cavity_ids, cavity_ids)
        return self.filter_by_mask(mask)
    
    def to_pdb(self, filename: str, atom_name: str = 'H', residue_name: str = 'PRB'):
        """Export probes as PDB file.
        
        Parameters
        ----------
        filename : str
            Output PDB filename
        atom_name : str, optional
            Atom name in PDB (default: 'H' for cavity interior, 'HA' for cavity surface, 'O' for protein surface)
        residue_name : str, optional
            Residue name in PDB (default: 'PRB' for probe)
            
        Notes
        -----
        - Cavity interior probes use chain 'I', atom 'H'
        - Cavity surface probes use chain 'C', atom 'HA', cavity_id as residue number
        - Protein surface probes use chain 'S', atom 'O', sequential numbering
        """
        with open(filename, 'w') as f:
            protein_surface_counter = 1
            for i, (pos, source, cav_id) in enumerate(zip(
                self.positions, self.sources, self.cavity_ids
            ), start=1):
                # Use different atom names and chain IDs based on source
                if source == 'cavity_interior':
                    atom = 'H' if atom_name == 'H' else atom_name
                    chain = 'I'
                    res_num = cav_id
                elif source == 'cavity_surface':
                    atom = 'HA' if atom_name == 'H' else atom_name
                    chain = 'C'
                    res_num = cav_id
                else:  # protein_surface
                    atom = 'O' if atom_name == 'H' else atom_name
                    chain = 'S'
                    res_num = protein_surface_counter
                    protein_surface_counter += 1
                
                # Wrap serial and residue numbers to stay within PDB format limits
                # Serial: 5 chars (1-99999), Residue: 4 chars (1-9999)
                serial = ((i - 1) % 99999) + 1
                res_num_wrapped = ((res_num - 1) % 9999) + 1
                
                # Format: HETATM serial atom res chain resSeq x y z occupancy tempFactor element
                line = (
                    f"HETATM{serial:5d}  {atom:<3s} {residue_name:3s} {chain}{res_num_wrapped:4d}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                    f"  1.00  0.00          {atom[0]:>2s}\n"
                )
                f.write(line)
            f.write("END\n")
    
    def to_pdb_with_protein(self, 
                           output_filename: str, 
                           protein_pdb: str,
                           atom_name: str = 'O',
                           residue_name: str = 'PRB'):
        """Export probes combined with protein structure as PDB file.
        
        Parameters
        ----------
        output_filename : str
            Output PDB filename
        protein_pdb : str
            Input protein PDB filename
        atom_name : str, optional
            Atom name in PDB (default: 'H' for cavity interior, 'HA' for cavity surface, 'O' for protein surface)
        residue_name : str, optional
            Residue name in PDB (default: 'PRB')
            
        Notes
        -----
        - Protein structure uses original chain IDs
        - Cavity interior probes use chain 'I', atom 'H'
        - Cavity surface probes use chain 'C', atom 'HA', cavity_id as residue number
        - Protein surface probes use chain 'S', atom 'O', sequential numbering
        """
        with open(output_filename, 'w') as outf:
            # Copy protein structure and track last serial number
            last_serial = 0
            with open(protein_pdb, 'r') as inf:
                for line in inf:
                    if line.startswith(('ATOM', 'HETATM')):
                        outf.write(line)
                        # Extract serial number from columns 7-11
                        try:
                            serial = int(line[6:11].strip())
                            last_serial = max(last_serial, serial)
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith('TER'):
                        outf.write(line)
                    elif line.startswith('END'):
                        break
            
            # Add probe positions starting after protein atoms
            for idx, (pos, source, cav_id) in enumerate(zip(
                self.positions, self.sources, self.cavity_ids
            ), start=1):
                # Use same chain and atom type for all probes
                atom = atom_name
                chain = 'B'  # All probes in chain B
                res_num = idx  # Sequential residue numbering
                
                # Continue serial numbering from protein, wrapping at 99999
                # Residue numbers wrap at 9999
                serial = ((last_serial + idx - 1) % 99999) + 1
                res_num_wrapped = ((res_num - 1) % 9999) + 1
                
                line = (
                    f"HETATM{serial:5d}  {atom:<3s} {residue_name:3s} {chain}{res_num_wrapped:4d}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                    f"  1.00  0.00          {atom[0]:>2s}\n"
                )
                outf.write(line)
            
            outf.write("END\n")


class ProbeConverter:
    """Convert KVFinder 3D grids to probe coordinates.
    
    This class provides static methods for extracting probe positions
    from pyKVFinder results, converting grid indices to Cartesian coordinates.
    
    Probe Types
    -----------
    - cavity_interior: Interior cavity points (not at cavity-protein boundary)
    - cavity_surface: Cavity points at cavity-protein interface
    - protein_surface: All bulk solvent points (can be filtered downstream)
    """
    
    @staticmethod
    def grid_to_cartesian(
        grid_indices: np.ndarray,
        vertices: np.ndarray,
        step: float
    ) -> np.ndarray:
        """Convert grid indices to Cartesian coordinates.
        
        The pyKVFinder grid is defined by 4 vertices:
        - vertices[0] (p1): Origin
        - vertices[1] (p2): X-axis endpoint  
        - vertices[2] (p3): Y-axis endpoint
        - vertices[3] (p4): Z-axis endpoint
        
        Conversion formula:
            x = p1[0] + i * step
            y = p1[1] + j * step
            z = p1[2] + k * step
        
        Parameters
        ----------
        grid_indices : np.ndarray
            (N, 3) array of grid indices (i, j, k)
        vertices : np.ndarray
            (4, 3) grid definition from pyKVFinder
        step : float
            Grid spacing in Ångströms
            
        Returns
        -------
        np.ndarray
            (N, 3) Cartesian coordinates in Ångströms
        """
        if grid_indices.ndim != 2 or grid_indices.shape[1] != 3:
            raise ValueError(f"grid_indices must be (N, 3), got {grid_indices.shape}")
        if vertices.shape != (4, 3):
            raise ValueError(f"vertices must be (4, 3), got {vertices.shape}")
        
        origin = vertices[0]  # p1
        
        # Convert: coords = origin + indices * step
        cartesian = origin + grid_indices * step
        
        return cartesian
    
    @staticmethod
    def extract_cavity_probes(
        results,
        cavity_ids: Optional[List[int]] = None,
        interior_only: bool = False,
        surface_only: bool = False,
        threshold: float = 2.0
    ) -> ProbeSet:
        """Extract probe positions from cavity grid.
        
        Parameters
        ----------
        results : pyKVFinderResults
            Results object from pyKVFinder.run_workflow()
        cavity_ids : list of int, optional
            Specific cavity IDs to extract (None = all cavities)
        interior_only : bool, optional
            Extract only cavity interior points (default: False)
        surface_only : bool, optional
            Extract only cavity surface points (default: False)
        threshold : float, optional
            Minimum grid value to consider as cavity (default: 2.0)
            
        Returns
        -------
        ProbeSet
            Probe positions from cavities (interior and/or surface)
            
        Notes
        -----
        - Cavity interior: cavity grid points >= threshold that are NOT in surface grid
        - Cavity surface: points marked in surface grid (cavity-protein interface)
        - If neither interior_only nor surface_only is True, both are returned
        """
        cavity_grid = results.cavities
        surface_grid = results.surface
        
        # Identify cavity interior points (in cavity grid but not surface)
        cavity_mask = cavity_grid >= threshold
        surface_mask = surface_grid >= threshold
        interior_mask = cavity_mask & ~surface_mask
        
        # Build combined mask based on flags
        if interior_only:
            final_mask = interior_mask
        elif surface_only:
            final_mask = surface_mask
        else:
            # Both interior and surface
            final_mask = cavity_mask  # This includes both interior and surface
        
        # Get grid indices where mask is True
        indices = np.column_stack(np.nonzero(final_mask))  # (N, 3)
        
        if len(indices) == 0:
            # Return empty ProbeSet
            return ProbeSet(
                positions=np.zeros((0, 3)),
                sources=np.array([]),
                cavity_ids=np.array([], dtype=int),
                grid_indices=np.zeros((0, 3), dtype=int)
            )
        
        # Get cavity IDs at these positions
        cav_ids = cavity_grid[final_mask].astype(int)
        
        # Determine source labels (interior vs surface)
        is_surface = surface_grid[final_mask] >= threshold
        sources = np.where(is_surface, 'cavity_surface', 'cavity_interior')
        
        # Filter by requested cavity IDs
        if cavity_ids is not None:
            keep_mask = np.isin(cav_ids, cavity_ids)
            indices = indices[keep_mask]
            cav_ids = cav_ids[keep_mask]
            sources = sources[keep_mask]
        
        if len(indices) == 0:
            return ProbeSet(
                positions=np.zeros((0, 3)),
                sources=np.array([]),
                cavity_ids=np.array([], dtype=int),
                grid_indices=np.zeros((0, 3), dtype=int)
            )
        
        # Convert to Cartesian coordinates
        positions = ProbeConverter.grid_to_cartesian(
            indices,
            results._vertices,
            results._step
        )
        
        return ProbeSet(
            positions=positions,
            sources=sources,
            cavity_ids=cav_ids,
            grid_indices=indices
        )
    
    @staticmethod
    def extract_protein_surface_probes(
        results,
        max_distance: Optional[float] = None
    ) -> ProbeSet:
        """Extract probe positions from points outside protein within distance cutoff.
        
        Returns grid points outside the protein (value 1 or -1) that are within max_distance
        of any protein atom. Uses GPU acceleration when available for fast distance computation.
        
        Parameters
        ----------
        results : pyKVFinderResults
            Results object from pyKVFinder.run_workflow()
        max_distance : float, optional
            Maximum distance (Ångströms) from protein atoms to keep probes (default: None).
            If None, no distance filtering is applied and all points outside the protein are returned.
            
        Returns
        -------
        ProbeSet
            Probe positions from outside protein within distance cutoff
            
        Notes
        -----
        Grid values in results.cavities:
        - -1: bulk points (far from protein)
        - 0: biomolecule points
        - 1: empty space points (regions that don't meet volume cutoff for cavities)
        - >=2: cavity points
        
        Uses GPU acceleration (CUDA > MPS > CPU) via PyTorch when available for fast
        distance computation on large point clouds.
        """
        cavity_grid = results.cavities
        
        # Find all points outside protein (value == 1 or value == -1)
        outside_mask = (cavity_grid == 1) | (cavity_grid == -1)
        
        # Get indices
        indices = np.column_stack(np.nonzero(outside_mask))
        
        if len(indices) == 0:
            return ProbeSet(
                positions=np.zeros((0, 3)),
                sources=np.array([]),
                cavity_ids=np.array([], dtype=int),
                grid_indices=np.zeros((0, 3), dtype=int)
            )
        
        # Convert to Cartesian coordinates
        positions = ProbeConverter.grid_to_cartesian(
            indices,
            results._vertices,
            results._step
        )
        
        # Filter by distance to protein atoms if max_distance is specified
        if max_distance is not None:
            # Get protein atom coordinates from results
            protein_coords = ProbeConverter._get_protein_coords_from_results(results)
            positions, indices = ProbeConverter._filter_by_distance(
                positions, indices, protein_coords, max_distance
            )
        
        if len(positions) == 0:
            return ProbeSet(
                positions=np.zeros((0, 3)),
                sources=np.array([]),
                cavity_ids=np.array([], dtype=int),
                grid_indices=np.zeros((0, 3), dtype=int)
            )
        
        # Create metadata (protein surface probes have cavity_id = 0)
        sources = np.full(len(positions), 'protein_surface', dtype=object)
        cavity_ids = np.zeros(len(positions), dtype=int)
        
        return ProbeSet(
            positions=positions,
            sources=sources,
            cavity_ids=cavity_ids,
            grid_indices=indices
        )
    
    @staticmethod
    def _get_protein_coords_from_results(results) -> np.ndarray:
        """Extract protein atom coordinates from results object.
        
        Reads the atomic data from the PDB/XYZ file stored in results._input
        and extracts the xyz coordinates.
        
        Parameters
        ----------
        results : pyKVFinderResults
            Results object from pyKVFinder.run_workflow()
            
        Returns
        -------
        np.ndarray
            (N, 3) array of protein atom coordinates
        """
        # Import here to avoid circular imports
        from pyKVFinder import read_pdb, read_xyz, read_vdw
        from pyKVFinder.main import VDW
        
        # Load VDW radii
        vdw = read_vdw(VDW)
        
        # Read atomic data from input file
        input_file = results._input
        if input_file.endswith('.pdb'):
            atomic = read_pdb(input_file, vdw)
        elif input_file.endswith('.xyz'):
            atomic = read_xyz(input_file, vdw)
        else:
            raise ValueError(f"Unsupported file format: {input_file}")
        
        # Extract xyz coordinates (columns 4:7)
        protein_coords = atomic[:, 4:7].astype(np.float32)
        
        return protein_coords
    
    @staticmethod
    def _filter_by_distance(
        probe_positions: np.ndarray,
        probe_indices: np.ndarray,
        protein_coords: np.ndarray,
        max_distance: float
    ) -> tuple:
        """Filter probe positions by distance to protein atoms.
        
        Uses GPU-accelerated distance computation when PyTorch is available.
        
        Parameters
        ----------
        probe_positions : np.ndarray
            (N, 3) probe coordinates
        probe_indices : np.ndarray
            (N, 3) probe grid indices
        protein_coords : np.ndarray
            (M, 3) protein atom coordinates
        max_distance : float
            Maximum distance cutoff
            
        Returns
        -------
        tuple
            (filtered_positions, filtered_indices)
        """
        if TORCH_AVAILABLE:
            return ProbeConverter._filter_by_distance_torch(
                probe_positions, probe_indices, protein_coords, max_distance
            )

        else:
            raise ImportError(
                "PyTorch is not available. Please install PyTorch to use GPU-accelerated distance filtering."
            )
    
    @staticmethod
    def _filter_by_distance_torch(
        probe_positions: np.ndarray,
        probe_indices: np.ndarray,
        protein_coords: np.ndarray,
        max_distance: float
    ) -> tuple:
        """GPU-accelerated distance filtering using PyTorch."""
        # Convert to torch tensors
        probes = torch.from_numpy(probe_positions).float().to(TORCH_DEVICE)
        protein = torch.from_numpy(protein_coords).float().to(TORCH_DEVICE)
        
        # Process in batches to avoid memory issues
        batch_size = 10000
        keep_mask = torch.zeros(len(probes), dtype=torch.bool, device=TORCH_DEVICE)
        
        for i in range(0, len(probes), batch_size):
            print(f'Processing probes {i} to {min(i+batch_size, len(probes))} out of {len(probes)}...')
            batch = probes[i:i+batch_size]
            
            # Compute pairwise distances: (batch_size, 1, 3) - (1, n_atoms, 3)
            # Using cdist for efficiency
            dists = torch.cdist(batch, protein)
            
            # Check if any distance is within cutoff
            min_dists = dists.min(dim=1)[0]
            keep_mask[i:i+batch_size] = min_dists <= max_distance
        
        # Convert back to numpy
        keep_mask_np = keep_mask.cpu().numpy()
        
        return probe_positions[keep_mask_np], probe_indices[keep_mask_np]
    
    @staticmethod
    def extract_all_probes(
        results,
        include_cavity_interior: bool = True,
        include_cavity_surface: bool = True,
        include_protein_surface: bool = True,
        cavity_ids: Optional[List[int]] = None,
        cavity_threshold: float = 2.0,
        protein_surface_max_distance: Optional[float] = None
    ) -> ProbeSet:
        """Extract all probes (cavity interior + cavity surface + protein surface).
        
        Parameters
        ----------
        results : pyKVFinderResults
            Results object from pyKVFinder.run_workflow()
        include_cavity_interior : bool, optional
            Include cavity interior probes (default: True)
        include_cavity_surface : bool, optional
            Include cavity surface probes (cavity-protein interface) (default: True)
        include_protein_surface : bool, optional
            Include protein surface probes (protein-solvent interface) (default: True)
        cavity_ids : list of int, optional
            Specific cavity IDs to extract (None = all)
        cavity_threshold : float, optional
            Cavity grid threshold (default: 2.0)
        protein_surface_max_distance : float, optional
            Max distance from protein atoms for protein surface probes (default: None).
            If None, no distance filtering is applied.
            
        Returns
        -------
        ProbeSet
            Combined probe set
        """
        probe_sets = []
        
        # Extract cavity probes (interior and/or surface)
        if include_cavity_interior or include_cavity_surface:
            cavity_probes = ProbeConverter.extract_cavity_probes(
                results,
                cavity_ids=cavity_ids,
                interior_only=include_cavity_interior and not include_cavity_surface,
                surface_only=include_cavity_surface and not include_cavity_interior,
                threshold=cavity_threshold
            )
            if len(cavity_probes) > 0:
                probe_sets.append(cavity_probes)
        
        # Extract protein surface probes
        if include_protein_surface:
            protein_surface_probes = ProbeConverter.extract_protein_surface_probes(
                results,
                max_distance=protein_surface_max_distance
            )
            if len(protein_surface_probes) > 0:
                probe_sets.append(protein_surface_probes)
        
        if not probe_sets:
            # Return empty ProbeSet
            return ProbeSet(
                positions=np.zeros((0, 3)),
                sources=np.array([]),
                cavity_ids=np.array([], dtype=int),
                grid_indices=np.zeros((0, 3), dtype=int)
            )
        
        # Concatenate all probe sets
        return ProbeSet(
            positions=np.vstack([ps.positions for ps in probe_sets]),
            sources=np.concatenate([ps.sources for ps in probe_sets]),
            cavity_ids=np.concatenate([ps.cavity_ids for ps in probe_sets]),
            grid_indices=np.vstack([ps.grid_indices for ps in probe_sets])
        )
