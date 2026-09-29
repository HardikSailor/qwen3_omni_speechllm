"""Mix YAML loading. Same format as the old trainer's dataset YAMLs (multimodal_trainer/config/dataset/*.yaml):

    dataset_basedir: /optional/common/prefix      # optional; relative `path`s are joined to it
    train:
      - name: ST_CoVoST2_en_id_30_ST
        path: .../train/ST/CoVoST2_en_id_30_ST
        task: ST
        choose: 100000        # optional: samples drawn per epoch (without replacement if <= size)
        repeat: 1.0           # optional: alternative to choose, as a multiple of the dataset size
        weight: 0.0053        # optional: weight in the aggregated eval score
        # new, optional:
        language: en          # overrides an empty/missing MDS `language` column
        src_lang: en          # ST source / target language (else parsed from the name)
        tgt_lang: id
    validation:
      - ...
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

KNOWN_KEYS = {'name', 'path', 'task', 'choose', 'repeat', 'weight', 'language', 'src_lang', 'tgt_lang'}
SPLITS = ('train', 'validation')


@dataclass
class DatasetSpec:
    name: str
    path: str
    task: str
    choose: Optional[int] = None
    repeat: Optional[float] = None
    weight: Optional[float] = None
    language: Optional[str] = None
    src_lang: Optional[str] = None
    tgt_lang: Optional[str] = None

    def meta(self) -> Dict:
        """The dict passed to sea_text.build_row as `meta`."""
        return {k: v for k, v in dict(name=self.name, task=self.task, language=self.language,
                                      src_lang=self.src_lang, tgt_lang=self.tgt_lang).items() if v}


def load_mix(path: str, check_paths: bool = True) -> Dict[str, List[DatasetSpec]]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.get('dataset_basedir') or ''
    unknown_top = set(cfg) - set(SPLITS) - {'dataset_basedir'}
    if unknown_top:
        raise ValueError(f'{path}: unknown top-level keys {sorted(unknown_top)}')

    mix, missing = {}, []
    for split in SPLITS:
        specs, names = [], set()
        for entry in cfg.get(split) or []:
            unknown = set(entry) - KNOWN_KEYS
            if unknown:
                raise ValueError(f'{path}: {entry.get("name")}: unknown keys {sorted(unknown)}')
            for key in ('name', 'path', 'task'):
                if not entry.get(key):
                    raise ValueError(f'{path}: {split} entry {entry} has no {key!r}')
            if entry['name'] in names:
                raise ValueError(f'{path}: duplicate {split} dataset name {entry["name"]!r}')
            if entry.get('choose') is not None and entry.get('repeat') is not None:
                raise ValueError(f'{path}: {entry["name"]}: set choose or repeat, not both')
            names.add(entry['name'])
            spec = DatasetSpec(**{**entry, 'path': os.path.join(base, entry['path'])})
            if check_paths and not os.path.isdir(spec.path):
                missing.append(f'{split}: {spec.name}: {spec.path}')
            specs.append(spec)
        mix[split] = specs
    if missing:
        raise FileNotFoundError(f'{path}: {len(missing)} dataset paths not found:\n  ' + '\n  '.join(missing))
    return mix
