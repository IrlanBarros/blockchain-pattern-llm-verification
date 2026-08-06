#!/usr/bin/env python3
"""Ponto de entrada do pipeline.

A implementação está no pacote ``llm_pipeline`` e foi separada por
responsabilidade para facilitar testes, manutenção e auditoria.
"""

from llm_pipeline.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
