init:
	pip install -r requirements.txt

test:
	py.test tests

run:
	python __main__.py
