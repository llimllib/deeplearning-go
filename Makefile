.PHONY: typecheck
typecheck:
	pyright **/*.py

go/zobrist.py: scripts/gen_zobrist_hashes.py
	scripts/gen_zobrist_hashes.py > go/zobrist.py
