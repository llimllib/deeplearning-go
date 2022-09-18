.PHONY: typecheck
typecheck:
	pyright **/*.py

.PHONY: requirements
requirements:
	pip install -r requirements.txt

go/zobrist.py: scripts/gen_zobrist_hashes.py
	scripts/gen_zobrist_hashes.py > go/zobrist.py
