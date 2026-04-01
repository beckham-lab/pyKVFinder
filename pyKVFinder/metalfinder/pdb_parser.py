"""
PDB Parser Module

Simple PDB/CIF parser to extract atom information for metal binding site analysis.
Supports both PDB and mmCIF file formats.
"""

import numpy as np
from typing import Tuple
import os


def parse_pdb(pdb_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse PDB or mmCIF file to extract atom information.
    
    Parameters
    ----------
    pdb_file : str
        Path to PDB or mmCIF (.cif) file
    
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
    
    Note
    ----
    This function is used for metalfinder analysis. For CIF to PDB conversion,
    use parse_pdb_full() which also returns residue numbers and chain IDs.
    """
    # Check if file is mmCIF format
    if pdb_file.endswith('.cif'):
        return _parse_cif(pdb_file)
    
    # Otherwise parse as PDB format
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


def parse_pdb_full(pdb_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse PDB or mmCIF file with full residue information for format conversion.
    
    Parameters
    ----------
    pdb_file : str
        Path to PDB or mmCIF (.cif) file
    
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
    residue_numbers : np.ndarray
        (N,) residue sequence numbers
    chain_ids : np.ndarray
        (N,) chain identifiers
    is_backbone : np.ndarray
        (N,) boolean array indicating backbone atoms (N, CA, C, O)
    """
    if pdb_file.endswith('.cif'):
        return _parse_cif_full(pdb_file)
    else:
        return _parse_pdb_full(pdb_file)


def _parse_pdb_full(pdb_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse PDB file with full residue information."""
    coords = []
    atom_names = []
    elements = []
    residues = []
    residue_nums = []
    chain_ids = []
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
                
                # Parse element
                if len(line) >= 78:
                    element = line[76:78].strip()
                    if not element:
                        element = ''.join(c for c in atom_name if c.isalpha())[:2]
                else:
                    element = ''.join(c for c in atom_name if c.isalpha())[:2]
                
                if len(element) == 2:
                    element = element[0].upper() + element[1].lower()
                elif len(element) == 1:
                    element = element.upper()
                elements.append(element)
                
                # Parse residue info
                residue = line[17:20].strip()
                residues.append(residue)
                
                # Parse residue number
                res_num = int(line[22:26].strip())
                residue_nums.append(res_num)
                
                # Parse chain ID
                chain = line[21:22]
                chain_ids.append(chain)
                
                # Check if backbone
                is_bb.append(atom_name in backbone_atoms)
    
    return (
        np.array(coords),
        np.array(atom_names),
        np.array(elements),
        np.array(residues),
        np.array(residue_nums),
        np.array(chain_ids),
        np.array(is_bb)
    )


def _parse_cif(cif_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse mmCIF file to extract atom information.
    
    Parameters
    ----------
    cif_file : str
        Path to mmCIF file
    
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
    
    # Read CIF file and find atom_site loop
    with open(cif_file, 'r') as f:
        lines = f.readlines()
    
    # Find the atom_site loop section
    in_atom_loop = False
    column_map = {}
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Start of atom_site loop
        if line.startswith('_atom_site.'):
            in_atom_loop = True
            # Extract column name
            col_name = line.split('.')[1].split()[0]
            # Column index is the order we see these headers
            column_map[col_name] = len(column_map)
            continue
        
        # If we're in the atom loop and hit a non-underscore line with data
        if in_atom_loop and not line.startswith('_') and not line.startswith('#') and line:
            # Check if we've left the loop (new loop or data block)
            if line.startswith('loop_') or line.startswith('data_'):
                break
            
            # Parse the data line
            # Split by whitespace, but handle quoted strings
            parts = []
            in_quote = False
            current = []
            
            for char in line:
                if char in ('"', "'"):
                    in_quote = not in_quote
                elif char.isspace() and not in_quote:
                    if current:
                        parts.append(''.join(current))
                        current = []
                else:
                    current.append(char)
            if current:
                parts.append(''.join(current))
            
            # Skip if not enough columns
            if len(parts) < max(column_map.values()) + 1:
                continue
            
            # Extract atom type (group_PDB should be ATOM or HETATM)
            if 'group_PDB' in column_map:
                group = parts[column_map['group_PDB']]
                if group not in ('ATOM', 'HETATM'):
                    continue
            
            # Extract coordinates
            try:
                x = float(parts[column_map['Cartn_x']])
                y = float(parts[column_map['Cartn_y']])
                z = float(parts[column_map['Cartn_z']])
                coords.append([x, y, z])
            except (KeyError, ValueError, IndexError):
                continue
            
            # Extract atom name
            try:
                atom_name = parts[column_map['label_atom_id']]
                atom_names.append(atom_name)
            except (KeyError, IndexError):
                atom_names.append('UNK')
                atom_name = 'UNK'
            
            # Extract element
            try:
                element = parts[column_map['type_symbol']]
                # Clean up element symbol
                if len(element) == 2:
                    element = element[0].upper() + element[1].lower()
                elif len(element) == 1:
                    element = element.upper()
                elements.append(element)
            except (KeyError, IndexError):
                # Fall back to extracting from atom name
                element = ''.join(c for c in atom_name if c.isalpha())[:2]
                if len(element) == 2:
                    element = element[0].upper() + element[1].lower()
                elif len(element) == 1:
                    element = element.upper()
                elements.append(element)
            
            # Extract residue name
            try:
                residue = parts[column_map['label_comp_id']]
                residues.append(residue)
            except (KeyError, IndexError):
                residues.append('UNK')
            
            # Check if backbone
            is_bb.append(atom_name in backbone_atoms)
    
    return (
        np.array(coords),
        np.array(atom_names),
        np.array(elements),
        np.array(residues),
        np.array(is_bb)
    )


def _parse_cif_full(cif_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse mmCIF file with full residue information."""
    coords = []
    atom_names = []
    elements = []
    residues = []
    residue_nums = []
    chain_ids = []
    is_bb = []
    
    backbone_atoms = {'N', 'CA', 'C', 'O'}
    
    # Read CIF file and find atom_site loop
    with open(cif_file, 'r') as f:
        lines = f.readlines()
    
    # Find the atom_site loop section
    in_atom_loop = False
    column_map = {}
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Start of atom_site loop
        if line.startswith('_atom_site.'):
            in_atom_loop = True
            col_name = line.split('.')[1].split()[0]
            column_map[col_name] = len(column_map)
            continue
        
        if in_atom_loop and not line.startswith('_') and not line.startswith('#') and line:
            if line.startswith('loop_') or line.startswith('data_'):
                break
            
            # Parse the data line
            parts = []
            in_quote = False
            current = []
            
            for char in line:
                if char in ('"', "'"):
                    in_quote = not in_quote
                elif char.isspace() and not in_quote:
                    if current:
                        parts.append(''.join(current))
                        current = []
                else:
                    current.append(char)
            if current:
                parts.append(''.join(current))
            
            if len(parts) < max(column_map.values()) + 1:
                continue
            
            # Extract atom type
            if 'group_PDB' in column_map:
                group = parts[column_map['group_PDB']]
                if group not in ('ATOM', 'HETATM'):
                    continue
            
            # Extract coordinates
            try:
                x = float(parts[column_map['Cartn_x']])
                y = float(parts[column_map['Cartn_y']])
                z = float(parts[column_map['Cartn_z']])
                coords.append([x, y, z])
            except (KeyError, ValueError, IndexError):
                continue
            
            # Extract atom name
            try:
                atom_name = parts[column_map['label_atom_id']]
                atom_names.append(atom_name)
            except (KeyError, IndexError):
                atom_names.append('UNK')
                atom_name = 'UNK'
            
            # Extract element
            try:
                element = parts[column_map['type_symbol']]
                if len(element) == 2:
                    element = element[0].upper() + element[1].lower()
                elif len(element) == 1:
                    element = element.upper()
                elements.append(element)
            except (KeyError, IndexError):
                element = ''.join(c for c in atom_name if c.isalpha())[:2]
                if len(element) == 2:
                    element = element[0].upper() + element[1].lower()
                elif len(element) == 1:
                    element = element.upper()
                elements.append(element)
            
            # Extract residue name
            try:
                residue = parts[column_map['label_comp_id']]
                residues.append(residue)
            except (KeyError, IndexError):
                residues.append('UNK')
            
            # Extract residue number
            try:
                res_num = int(parts[column_map['label_seq_id']])
                residue_nums.append(res_num)
            except (KeyError, ValueError, IndexError):
                # Fall back to auth_seq_id or sequential numbering
                try:
                    res_num = int(parts[column_map['auth_seq_id']])
                    residue_nums.append(res_num)
                except (KeyError, ValueError, IndexError):
                    residue_nums.append(len(residue_nums) + 1)
            
            # Extract chain ID
            try:
                chain = parts[column_map['label_asym_id']]
                # CIF chain IDs can be longer, take first char for PDB compatibility
                chain_ids.append(chain[0] if chain else 'A')
            except (KeyError, IndexError):
                chain_ids.append('A')
            
            # Check if backbone
            is_bb.append(atom_name in backbone_atoms)
    
    return (
        np.array(coords),
        np.array(atom_names),
        np.array(elements),
        np.array(residues),
        np.array(residue_nums),
        np.array(chain_ids),
        np.array(is_bb)
    )
