.PHONY: generate-models lint typecheck

generate-models:
	datamodel-codegen \
		--input client/openapi.yaml \
		--output client/generated_models.py \
		--output-model-type pydantic_v2.BaseModel \
		--target-python-version 3.10 \
		--use-standard-collections \
		--use-union-operator \
		--field-constraints \
		--use-annotated \
		--collapse-root-models

lint:
	ruff check .

typecheck:
	pyright
