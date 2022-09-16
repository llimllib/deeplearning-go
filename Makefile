.PHONY: gen_hashes
gen_hashes:
	scripts/gen_zobrist_hashes.py > go/zobrist.py
