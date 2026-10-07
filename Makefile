test:
	pytest -q
verify:
	python scripts/verify_reparam.py
inspect9:
	python scripts/inspect_model.py --model reptor9
inspectA:
	python scripts/inspect_model.py --model reptor_a
