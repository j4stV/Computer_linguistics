from __future__ import annotations

import os
import sys
import importlib.util
from typing import Any, Dict, List, Optional


_UserOntologyRepository = None  # type: ignore[assignment]

# Пытаемся импортировать пользовательский файл OntologyRepository.py из корня рабочей директории
try:
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    candidate = os.path.join(ROOT_DIR, 'OntologyRepository.py')
    if os.path.isfile(candidate):
        spec = importlib.util.spec_from_file_location('user_ontology_repository', candidate)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules['user_ontology_repository'] = mod
            spec.loader.exec_module(mod)
            _UserOntologyRepository = getattr(mod, 'OntologyRepository', None)
            # опционально реэкспортируем типы, если они есть
            globals().update({
                name: getattr(mod, name)
                for name in ['TClass', 'TObject', 'DProperty', 'OProperty', 'Signature']
                if hasattr(mod, name)
            })
    else:
        # Попытка обычного импорта, если файл доступен в PYTHONPATH
        from OntologyRepository import OntologyRepository as _UserOntologyRepository  # type: ignore
except Exception:
    _UserOntologyRepository = None  # type: ignore


if _UserOntologyRepository is None:
    class OntologyRepository:  # type: ignore[misc]
        """Заглушка на время миграций/отсутствия пользовательского файла.
        Любой вызов методов выбрасывает RuntimeError.
        """

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise RuntimeError('OntologyRepository is unavailable: user OntologyRepository.py not found')
else:
    class OntologyRepository(_UserOntologyRepository):  # type: ignore[misc]
        pass


