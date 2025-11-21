"""
PDB Parser Module

Simple PDB parser to extract atom information for metal binding site analysis.
"""

import numpy as np
from typing import Tuple


def parse_pdb(pdb_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse PDB file to extract atom information.
    
    Parameters
    ----------
    pdb_file : str
        Path to PDB file
    
    Returns
    -------
    atom_coords : np.ndarray
        (N, 3) atomic coordinates in Ångströms
    atom_names : np.ndarray
        (N,) PDB atom names (e.g., 'OD1', 'NE2', 'CA')
    atom_types : np.ndarray
        (N,) element symbols (e.g., 'O', 'N', 'C')
    residue_names : np.ndarray
        (N,) 3-letter residue codes (e.g., 'ASP', 'HIS')
    is_backbone : np.ndarray
        (N,) boolean array indicating backbone atoms (N, CA, C, O)
    """
    coords = []
    atom_names = []
    elements = []
    residues = []
    is_bb = []
    
    backbone_atoms = {'N', 'CA', 'C', 'O'}
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Parse coordinates
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coords.append([x, y, z])
                
                # Parse atom name
                atom_name = line[12:16].strip()
                atom_names.append(atom_name)
                
                # Try to get element from element column (77-78)
                if len(line) >= 78:
                    element = line[76:78].strip()
                    if not element:
                        # Fall back to atom name
                        element = ''.join(c for c in atom_name if c.isalpha())[:2]
                else:
                    # Extract element from atom name
                    element = ''.join(c for c in atom_name if c.isalpha())[:2]
                
                # Clean up element symbol (capitalize first, lowercase second)
                if len(element) == 2:
                    element = element[0].upper() + element[1].lower()
                elif len(element) == 1:
                    element = element.upper()
                
                elements.append(element)
                
                # Parse residue name
                residue = line[17:20].strip()
                residues.append(residue)
                
                # Check if backbone
                is_bb.append(atom_name in backbone_atoms)
    
    return (
        np.array(coords),
        np.array(atom_names),
        np.array(elements),
        np.array(residues),
        np.array(is_bb)
    )
