from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey


def get_inchi_key(smiles: str) -> str:
    """
    Get the InChI key for a molecule from its SMILES.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        InChI key string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return MolToInchiKey(mol)