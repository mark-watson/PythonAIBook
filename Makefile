.PHONY: clean

clean:
	find . -name '*~' -exec rm -f {} +
	find . -name '.DS_Store' -exec rm -f {} +
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '.venv' -type d -exec rm -rf {} +