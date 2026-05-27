import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolfiles
from torch_geometric.data import Data

def bond_order_sum(atom):
    return sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())

def sanitize_for_charges(m):
    m.UpdatePropertyCache(strict=False)
    problems = Chem.DetectChemistryProblems(m)

    if not problems:
        Chem.SanitizeMol(m)
        return m

    for p in problems:
        if p.GetType() != 'AtomValenceException':
            continue

        at = m.GetAtomWithIdx(p.GetAtomIdx())
        Z = at.GetAtomicNum()
        q = at.GetFormalCharge()
        bos = bond_order_sum(at)

        # --------------------
        # POSITIVE CHARGES
        # --------------------

        # Nitrogen: ammonium / pyridinium
        if Z == 7 and q == 0:
            # typical neutral N valence ≤ 3
            if bos > 3:
                at.SetFormalCharge(1)

        # Phosphorus: phosphonium
        elif Z == 15 and q == 0:
            # neutral P typically valence 3 or 5
            if bos > 5:
                at.SetFormalCharge(1)
            elif bos > 3:
                at.SetFormalCharge(1)

        # Sulfur: sulfonium
        elif Z == 16 and q == 0:
            # neutral S typically 2 or 6
            if bos > 2 and bos <= 4:
                at.SetFormalCharge(1)

        # --------------------
        # NEGATIVE CHARGES
        # --------------------

        # Oxygen: carboxylate, phenolate, phosphate
        elif Z == 8 and q == 0:
            # single-bonded O with one neighbour
            if bos == 1:
                at.SetFormalCharge(-1)

        # Nitrogen anion (rare, but occurs)
        elif Z == 7 and q == 0:
            if bos <= 2:
                at.SetFormalCharge(-1)

        # Sulfur anion: thiolate
        elif Z == 16 and q == 0:
            if bos == 1:
                at.SetFormalCharge(-1)

        # Halides
        elif Z in (9, 17, 35, 53) and q == 0:
            if bos == 0:
                at.SetFormalCharge(-1)

    Chem.SanitizeMol(m)
    return m

def smiles_to_mol(smi, allow_charges=True):
    if not allow_charges:
        m = Chem.MolFromSmiles(smi)
        m = Chem.AddHs(mol)
        
    else:
        params = rdmolfiles.SmilesParserParams()
        params.removeHs = False
        params.sanitize = False
    
        m = Chem.MolFromSmiles(smi, params)
        m = sanitize_for_charges(m)
    
    if m is None:
        raise ValueError("SMILES parse failed")

    return m

# helpers to serialise RDKit enums
_HYB_STR_TO_ENUM = {str(v): v for v in [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]}

_BOND_STR_TO_ENUM = {str(v): v for v in [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]}

def _hyb_to_json(v) -> str:
    return str(v)

def _hyb_from_json(s: str):
    if s not in _HYB_STR_TO_ENUM:
        raise ValueError(f"Unknown hybridization token in config: {s}")
    return _HYB_STR_TO_ENUM[s]

def _bond_to_json(v) -> str:
    return str(v)

def _bond_from_json(s: str):
    if s not in _BOND_STR_TO_ENUM:
        raise ValueError(f"Unknown bond token in config: {s}")
    return _BOND_STR_TO_ENUM[s]

@dataclass
class FeatureSpec:
    """
    Feature spec that can be numeric (dim=1) or categorical (one-hot).
    For categorical features, vocab MUST be frozen for deployment.
    Unknown values map to all-zeros (by default) OR an UNK bucket if include_unk=True.
    """
    name: str
    func: Callable[[Any], Any]
    vocab: Optional[List[Any]] = None
    include_unk: bool = False
    # optional per-feature (de)serialisers for vocab entries
    vocab_to_json: Optional[Callable[[Any], Any]] = None
    vocab_from_json: Optional[Callable[[Any], Any]] = None

    def __post_init__(self):
        self.is_categorical = self.vocab is not None
        if self.is_categorical:
            if self.include_unk:
                # Reserve an explicit UNK slot at the end
                self.index = {v: i for i, v in enumerate(self.vocab)}
                self.unk_index = len(self.vocab)
                self.dim = len(self.vocab) + 1
            else:
                self.index = {v: i for i, v in enumerate(self.vocab)}
                self.unk_index = None
                self.dim = len(self.vocab)
        else:
            self.index = None
            self.unk_index = None
            self.dim = 1

    def __call__(self, obj) -> List[float]:
        value = self.func(obj)

        if not self.is_categorical:
            return [float(value)]

        vec = [0.0] * self.dim
        if value in self.index:
            vec[self.index[value]] = 1.0
        elif self.include_unk:
            vec[self.unk_index] = 1.0
        # else: unknown -> all zeros
        return vec

    def to_config(self) -> Dict[str, Any]:
        cfg = {
            "name": self.name,
            "is_categorical": self.is_categorical,
            "include_unk": self.include_unk,
            "vocab": None,
        }
        if self.is_categorical:
            if self.vocab_to_json is None:
                # assume entries already JSON-serialisable
                cfg["vocab"] = list(self.vocab)
            else:
                cfg["vocab"] = [self.vocab_to_json(v) for v in self.vocab]
        return cfg

    @staticmethod
    def from_config(
        cfg: Dict[str, Any],
        func: Callable[[Any], Any],
        vocab_from_json: Optional[Callable[[Any], Any]] = None,
        vocab_to_json: Optional[Callable[[Any], Any]] = None,
    ) -> "FeatureSpec":
        vocab = cfg.get("vocab", None)
        if vocab is not None and vocab_from_json is not None:
            vocab = [vocab_from_json(v) for v in vocab]
        return FeatureSpec(
            name=cfg["name"],
            func=func,
            vocab=vocab,
            include_unk=cfg.get("include_unk", False),
            vocab_to_json=vocab_to_json,
            vocab_from_json=vocab_from_json,
        )

# ---------- feature registries ----------
# Each registry entry defines:
#   - function to extract feature from atom/bond
#   - whether vocab is dynamic (needs fitting from smiles)
#   - optional vocab serialisers

ATOM_FEATURES: Dict[str, Dict[str, Any]] = {
    "species": {
        "func": lambda atom: atom.GetAtomicNum(),
        "dynamic_vocab": True,  # depends on training SMILES set
        "vocab_to_json": int,
        "vocab_from_json": int,
    },
    "degree": {
        "func": lambda atom: atom.GetDegree(),
        "dynamic_vocab": False,
    },
    "implicit_hydrogens": {
        "func": lambda atom: atom.GetNumImplicitHs(),
        "dynamic_vocab": False,
    },
    "explicit_hydrogens": {
        "func": lambda atom: atom.GetNumExplicitHs(),
        "dynamic_vocab": False,
    },
    "formal_charge": {
        "func": lambda atom: atom.GetFormalCharge(),
        "dynamic_vocab": False,
    },
    "is_aromatic": {
        "func": lambda atom: int(atom.GetIsAromatic()),
        "dynamic_vocab": False,
    },
    "hybridization": {
        "func": lambda atom: atom.GetHybridization(),
        "dynamic_vocab": False,
        "default_vocab": [
            Chem.HybridizationType.SP,
            Chem.HybridizationType.SP2,
            Chem.HybridizationType.SP3,
            Chem.HybridizationType.SP3D,
            Chem.HybridizationType.SP3D2,
        ],
        "vocab_to_json": _hyb_to_json,
        "vocab_from_json": _hyb_from_json,
    },
}

BOND_FEATURES: Dict[str, Dict[str, Any]] = {
    "bond_type": {
        "func": lambda bond: bond.GetBondType(),
        "dynamic_vocab": False,
        "default_vocab": [
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC,
        ],
        "vocab_to_json": _bond_to_json,
        "vocab_from_json": _bond_from_json,
    },
    "is_conjugated": {
        "func": lambda bond: int(bond.GetIsConjugated()),
        "dynamic_vocab": False,
    },
    "is_aromatic": {
        "func": lambda bond: int(bond.GetIsAromatic()),
        "dynamic_vocab": False,
    },
    "is_in_ring": {
        "func": lambda bond: int(bond.IsInRing()),
        "dynamic_vocab": False,
    },
}

class SmilesToGraph:
    """
    Frozen, deployable SMILES->graph transformer.
    - Choose atom/bond feature names
    - Fit dynamic vocabs from a training SMILES set (e.g., atomic species)
    - Save/load full config including vocabs so dimensions are consistent in deployment
    """

    def __init__(
        self,
        atom_feature_names: List[str],
        bond_feature_names: Optional[List[str]] = None,
        *,
        add_hs: bool = False,
        use_3d: bool = False,
        max_confs: int = 1,
        include_unk: bool = False,
        # optional pre-fitted vocabs for dynamic features
        fitted_vocabs: Optional[Dict[str, List[Any]]] = None,
    ):
        self.atom_feature_names = list(atom_feature_names)
        self.bond_feature_names = list(bond_feature_names) if bond_feature_names else []
        self.add_hs = bool(add_hs)
        self.use_3d = bool(use_3d)
        self.max_confs = int(max_confs)
        self.include_unk = bool(include_unk)

        self._validate_feature_names()

        # Build specs (some may need fitted vocabs)
        self.atom_specs: List[FeatureSpec] = self._build_specs(
            self.atom_feature_names, ATOM_FEATURES, fitted_vocabs or {}
        )
        self.bond_specs: List[FeatureSpec] = self._build_specs(
            self.bond_feature_names, BOND_FEATURES, fitted_vocabs or {}
        )

    @classmethod
    def fit_from_smiles(
        cls,
        smiles: Iterable[str],
        atom_feature_names: List[str],
        bond_feature_names: Optional[List[str]] = None,
        *,
        add_hs: bool = False,
        use_3d: bool = False,
        max_confs: int = 1,
        include_unk: bool = False,
    ) -> "SmilesToGraph":
        smiles = [s.strip() for s in smiles]
        fitted_vocabs = cls._fit_dynamic_vocabs(
            smiles,
            atom_feature_names=atom_feature_names,
            bond_feature_names=bond_feature_names or [],
            add_hs=add_hs,
        )
        return cls(
            atom_feature_names=atom_feature_names,
            bond_feature_names=bond_feature_names,
            add_hs=add_hs,
            use_3d=use_3d,
            max_confs=max_confs,
            include_unk=include_unk,
            fitted_vocabs=fitted_vocabs,
        )

    def mol_to_graph(self, mol: Chem.Mol, *, conf_id: int = 0) -> Data:
        # node features
        if self.atom_specs:
            node_features = []
            for atom in mol.GetAtoms():
                feats = []
                for spec in self.atom_specs:
                    feats.extend(spec(atom))
                node_features.append(feats)
            x = torch.tensor(node_features, dtype=torch.float32)
        else:
            x = torch.zeros((mol.GetNumAtoms(), 1), dtype=torch.float32)

        # atomic numbers (always useful)
        z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)

        # edges (+ optional edge_attr)
        edges: List[List[int]] = []
        edge_features: List[List[float]] = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])

            if self.bond_specs:
                feats = []
                for spec in self.bond_specs:
                    feats.extend(spec(bond))
                edge_features.append(feats)
                edge_features.append(feats)

        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        data = Data(x=x, z=z, edge_index=edge_index)

        if self.bond_specs and len(edge_features) > 0:
            data.edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        # positions (optional)
        if self.use_3d:
            conf = mol.GetConformer(conf_id)
            if not conf.Is3D():
                raise ValueError("use_3d=True but conformer is not 3D")
            pos = np.asarray(conf.GetPositions(), dtype=np.float32)
            data.pos = torch.tensor(pos, dtype=torch.float32)

        return data

    def smiles_to_graphs(
        self,
        smi: str,
        *,
        return_all_confs: bool = False,
        optimise_mmff: bool = True,
    ) -> Tuple[Data, List[Data]]:
        """
        Returns:
          g0, conf_pool
        where g0 is the lowest-energy conformer graph (or just the single graph),
        and conf_pool are remaining conformers (possibly empty).
        """
        mol = smiles_to_mol(smi)

        conf_ids = [0]
        if self.use_3d or self.max_confs > 1:
            conf_ids = self._embed_conformers(mol, n_confs=self.max_confs, optimise_mmff=optimise_mmff)

        g0 = self.mol_to_graph(mol, conf_id=conf_ids[0])

        conf_pool: List[Data] = []
        if return_all_confs and len(conf_ids) > 1:
            for cid in conf_ids[1:]:
                conf_pool.append(self.mol_to_graph(mol, conf_id=cid))

        return g0, conf_pool

    def save_config(self, path: str) -> None:
        cfg = self.to_config_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_config(cls, path: str) -> "SmilesToGraph":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # bypass __init__ to avoid calling _build_specs
        obj = cls.__new__(cls)

        # Set basic attributes expected by methods
        obj.atom_feature_names = list(cfg["atom_feature_names"])
        obj.bond_feature_names = list(cfg.get("bond_feature_names", []))
        obj.add_hs = bool(cfg["add_hs"])
        obj.use_3d = bool(cfg["use_3d"])
        obj.max_confs = int(cfg["max_confs"])
        obj.include_unk = bool(cfg.get("include_unk", False))

        # Validate feature names exist in registries
        obj._validate_feature_names()

        # rebuild specs from saved per-feature configs (includes vocabs)
        obj.atom_specs = obj._build_specs_from_feature_cfgs(cfg["atom_feature_cfgs"], ATOM_FEATURES)
        obj.bond_specs = obj._build_specs_from_feature_cfgs(cfg["bond_feature_cfgs"], BOND_FEATURES)
        return obj

    def to_config_dict(self) -> Dict[str, Any]:
        return {
            "add_hs": self.add_hs,
            "use_3d": self.use_3d,
            "max_confs": self.max_confs,
            "include_unk": self.include_unk,
            "atom_feature_names": self.atom_feature_names,
            "bond_feature_names": self.bond_feature_names,
            "atom_feature_cfgs": [spec.to_config() for spec in self.atom_specs],
            "bond_feature_cfgs": [spec.to_config() for spec in self.bond_specs],
        }

    @property
    def node_feature_dim(self) -> int:
        return sum(spec.dim for spec in self.atom_specs) if self.atom_specs else 1

    @property
    def edge_feature_dim(self) -> int:
        return sum(spec.dim for spec in self.bond_specs) if self.bond_specs else 0

    # ---------- internal ----------
    def _validate_feature_names(self) -> None:
        for n in self.atom_feature_names:
            if n not in ATOM_FEATURES:
                raise ValueError(f"Unknown atom feature name: {n}. Available: {sorted(ATOM_FEATURES)}")
        for n in self.bond_feature_names:
            if n not in BOND_FEATURES:
                raise ValueError(f"Unknown bond feature name: {n}. Available: {sorted(BOND_FEATURES)}")

    @staticmethod
    def _build_specs(
        names: List[str],
        registry: Dict[str, Dict[str, Any]],
        fitted_vocabs: Dict[str, List[Any]],
    ) -> List[FeatureSpec]:
        specs: List[FeatureSpec] = []
        for name in names:
            meta = registry[name]
            func = meta["func"]
            dynamic = meta.get("dynamic_vocab", False)

            vocab = None
            if "default_vocab" in meta:
                vocab = list(meta["default_vocab"])
            if dynamic:
                if name not in fitted_vocabs:
                    raise ValueError(
                        f"Feature '{name}' requires a fitted vocab, but none was provided. "
                        f"Use SmilesToGraph.fit_from_smiles(...) or pass fitted_vocabs."
                    )
                vocab = list(fitted_vocabs[name])

            vocab_to_json = meta.get("vocab_to_json", None)
            vocab_from_json = meta.get("vocab_from_json", None)

            specs.append(FeatureSpec(
                name=name,
                func=func,
                vocab=vocab,
                include_unk=False,  # set at object-level in build_specs_from_feature_cfgs
                vocab_to_json=vocab_to_json,
                vocab_from_json=vocab_from_json,
            ))
        return specs

    def _build_specs_from_feature_cfgs(
        self,
        feature_cfgs: List[Dict[str, Any]],
        registry: Dict[str, Dict[str, Any]],
    ) -> List[FeatureSpec]:
        specs: List[FeatureSpec] = []
        for cfg in feature_cfgs:
            name = cfg["name"]
            meta = registry[name]
            func = meta["func"]
            vocab_from_json = meta.get("vocab_from_json", None)
            vocab_to_json = meta.get("vocab_to_json", None)

            # override include_unk from saved config or object default
            cfg = dict(cfg)
            cfg["include_unk"] = cfg.get("include_unk", self.include_unk)

            specs.append(FeatureSpec.from_config(
                cfg=cfg,
                func=func,
                vocab_from_json=vocab_from_json,
                vocab_to_json=vocab_to_json,
            ))
        return specs

    @staticmethod
    def _fit_dynamic_vocabs(
        smiles: List[str],
        *,
        atom_feature_names: List[str],
        bond_feature_names: List[str],
        add_hs: bool,
    ) -> Dict[str, List[Any]]:
        fitted: Dict[str, List[Any]] = {}

        mols: List[Chem.Mol] = []
        for s in smiles:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            if add_hs:
                m = Chem.AddHs(m)
            mols.append(m)

        for name in atom_feature_names:
            meta = ATOM_FEATURES[name]
            if meta.get("dynamic_vocab", False):
                values = set()
                func = meta["func"]
                for m in mols:
                    for atom in m.GetAtoms():
                        values.add(func(atom))
                vocab = sorted(values)
                fitted[name] = list(vocab)

        for name in bond_feature_names:
            meta = BOND_FEATURES[name]
            if meta.get("dynamic_vocab", False):
                values = set()
                func = meta["func"]
                for m in mols:
                    for bond in m.GetBonds():
                        values.add(func(bond))
                vocab = sorted(values)
                fitted[name] = list(vocab)

        return fitted

    def _embed_conformers(self, mol: Chem.Mol, *, n_confs: int, optimise_mmff: bool = True) -> List[int]:
        # Embed
        res = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, numThreads=0)
        res = list(res)
        if res == []:
            # fallback 
            res = list(AllChem.EmbedMultipleConfs(
                mol,
                numConfs=n_confs,
                useBasicKnowledge=False,
                numThreads=0
            ))
            if res == []:
                raise ValueError("Complete conformer embed failure for molecule")

        if not optimise_mmff:
            return res

        mmff = AllChem.MMFFOptimizeMoleculeConfs(mol)
        # mmff returns list of tuples: (status, energy)
        pairs = list(zip(res, mmff))
        pairs.sort(key=lambda x: x[1][1])  # energy
        conf_ids = [cid for cid, _ in pairs]
        return conf_ids
