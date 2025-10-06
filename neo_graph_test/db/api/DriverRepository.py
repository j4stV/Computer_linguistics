from __future__ import annotations

import os
import sys
import importlib.util

# Обёртка для пользовательского Neo4jRepository, доступная как db.api.DriverRepository

DriverRepository = None  # type: ignore[assignment]

try:
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    candidate = os.path.join(ROOT_DIR, 'Neo4jRepository.py')
    if os.path.isfile(candidate):
        spec = importlib.util.spec_from_file_location('user_neo4j_repository', candidate)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules['user_neo4j_repository'] = mod
            spec.loader.exec_module(mod)
            DriverRepository = getattr(mod, 'Neo4jRepository', None)
    else:
        from Neo4jRepository import Neo4jRepository as DriverRepository  # type: ignore
except Exception:
    DriverRepository = None  # type: ignore


