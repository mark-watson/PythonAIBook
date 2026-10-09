.PHONY: help run sync lock check lint install uninstall build completions clean distclean

PYTHON ?= python3
PREFIX ?= /usr/local
BINDIR := $(PREFIX)/bin

PY_MODULES := agent.py harness_config.py approval.py tools.py chat_loop.py fireworks_ai.py mlx_serve.py search.py line_input.py

help:
	@echo "Targets:"
	@echo "  run          start the coding-agent REPL (uv run agent.py)"
	@echo "  sync         create/update .venv from uv.lock"
	@echo "  lock         regenerate uv.lock"
	@echo "  check        byte-compile all Python sources"
	@echo "  lint         alias for check"
	@echo "  install      install as a global tool via 'uv tool install .'"
	@echo "  uninstall    remove the global tool"
	@echo "  build        build sdist + wheel into dist/ via uv build"
	@echo "  completions  generate shell completions into ./completions/"
	@echo "  clean        clean directory of dev artifacts"
	@echo "  distclean    clean plus remove .venv"

run:
	uv run agent.py

sync:
	uv sync

lock:
	uv lock

check: lint

lint:
	uv run python -m py_compile $(PY_MODULES)

install:
	uv tool install .
	@echo "Installed 'coding-agent' via uv tool (make sure ~/.local/bin is on PATH)"

uninstall:
	uv tool uninstall coding-harness-agent || true
	@echo "Removed global tool"

build:
	uv build
	@echo "Distributions in ./dist/"

completions:
	@mkdir -p completions
	@echo "# bash completion for coding-agent" > completions/coding-agent.bash
	@echo "_coding_agent() {" >> completions/coding-agent.bash
	@printf '  local cur="$${COMP_WORDS[COMP_CWORD]}"\n' >> completions/coding-agent.bash
	@echo "  local opts=\"--help --version --prompt --stdin --yes --dry-run --provider --model --cwd --debug --quiet --plain --no-color -h -p -y -q -v\"" >> completions/coding-agent.bash
	@printf '  COMPREPLY=($$(compgen -W "$$opts" -- "$$cur"))\n' >> completions/coding-agent.bash
	@echo "}" >> completions/coding-agent.bash
	@echo "complete -F _coding_agent coding-agent" >> completions/coding-agent.bash
	@echo "# zsh completion for coding-agent" > completions/coding-agent.zsh
	@echo "#compdef coding-agent" >> completions/coding-agent.zsh
	@echo "_coding_agent() { _arguments '-h[help]' '--help[help]' '-v[version]' '--version[version]' '-p[prompt]:prompt:' '--prompt[prompt]:prompt:' '--stdin[stdin]' '-y[yes]' '--yes[yes]' '--dry-run[dry-run]' '--provider[provider]:provider:(fireworks mlx)' '--model[model]:model:' '--cwd[cwd]:dir:_files -/' '--debug[debug]' '-q[quiet]' '--quiet[quiet]' '--plain[plain]' '--no-color[no-color]' '*:prompt:_files' }" >> completions/coding-agent.zsh
	@echo "compdef _coding_agent coding-agent" >> completions/coding-agent.zsh 2>/dev/null || true
	@echo "# fish completion for coding-agent" > completions/coding-agent.fish
	@echo "complete -c coding-agent -s h -l help -d 'Show help'" >> completions/coding-agent.fish
	@echo "complete -c coding-agent -s v -l version -d 'Show version'" >> completions/coding-agent.fish
	@echo "complete -c coding-agent -s p -l prompt -d 'Prompt text' -r" >> completions/coding-agent.fish
	@echo "complete -c coding-agent -l stdin -d 'Read prompt from stdin'" >> completions/coding-agent.fish
	@echo "complete -c coding-agent -s y -l yes -d 'Auto-approve edits'" >> completions/coding-agent.fish
	@echo "complete -c coding-agent -l dry-run -d 'Show diffs without writing'" >> completions/coding-agent.fish
	@echo "complete -c coding-agent -l provider -d 'Provider' -r -a 'fireworks mlx'" >> completions/coding-agent.fish
	@echo "complete -c coding-agent -l model -d 'Model id' -r" >> completions/coding-agent.fish
	@echo "complete -c coding-agent -l cwd -d 'Working directory' -r -a '(__fish_complete_directories)'" >> completions/coding-agent.fish
	@echo "complete -c coding-agent -l debug -d 'Debug logging'" >> completions/coding-agent.fish
	@echo "complete -c coding-agent -s q -l quiet -d 'Quiet'" >> completions/coding-agent.fish
	@echo "complete -c coding-agent -l plain -l no-color -d 'No colors'" >> completions/coding-agent.fish
	@echo "Generated completions/coding-agent.{bash,zsh,fish}"

clean:
	rm -rf compiled dist coding-agent *~ __pycache__ *.pyc *.egg-info .pytest_cache .venv

distclean: clean
	rm -rf uv.lock
