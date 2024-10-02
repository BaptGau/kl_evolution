test:
	poetry install --quiet
	poetry run coverage run -m pytest -p no:warnings
	poetry run coverage report -m --fail-under=85

clean-coverage:
	rm -rf .coverage htmlcov