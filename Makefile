.PHONY: sync data test learner lint

DATA := data/input.txt
URL  := https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
SHA  := 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed

sync:     ## install torch + dev tools, then git hooks
	uv sync
	uv run lefthook install

data: $(DATA)  ## download tiny-shakespeare (1.1 MB, checksum-verified)
$(DATA):
	mkdir -p data && curl -fsSL -o $@ $(URL)
	echo "$(SHA)  $@" | shasum -a 256 -c || (rm -f $@; exit 1)

test:     ## tests that must stay green (CI runs these)
	uv run pytest -q

learner:  ## your red tests for the piece you're writing now
	uv run pytest -q -m learner || test $$? -eq 5  # 5 = no learner tests open right now

lint:
	uv run ruff format --check .
	uv run ruff check .
