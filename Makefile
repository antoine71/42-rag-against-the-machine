# --- Project settings ---
NAME=rag
SRC_DIR=src

# --- Colors settings ---
C_GREY			=	\033[90m
C_CYAN			=	\033[36m
C_GREEN			=	\033[32m
C_RED			=	\033[31m
C_RESET			=	\033[0m

# --- Emoji ---
E_WAIT			=	$(C_CYAN)[-_-']$(C_RESET)
E_OK			=	$(C_GREEN)[^_^]$(C_RESET)
E_KO			=	$(C_RED)[X_X]$(C_RESET)

# --- Venv dir ---
VENV_DIR		=	.venv

# --- Options ---
MYPY_FLAGS_BASE		= --exclude $(VENV_DIR) \
			  --no-color-output

MYPY_FLAGS_STRICT 	= $(MYPY_FLAGS_BASE) \
			  --strict

MYPY_FLAGS_STANDARD 	= $(MYPY_FLAGS_BASE) \
			  --warn-return-any \
			  --warn-unused-ignores \
			  --ignore-missing-imports \
			  --disallow-untyped-defs \
			  --check-untyped-defs

MYPY_FLAGS		?= $(MYPY_FLAGS_STANDARD)

.PHONY: install run debug clean lint lint-strict test

install: 
	@command -v uv > /dev/null || { \
		echo "$(E_KO) uv not installed"; \
		echo "	--> Install it: https://docs.astral.sh/uv/"; \
		exit 1; }
	@echo "$(E_WAIT) Installing project..."
	@printf "$(C_GREY)"; uv sync --color never; printf "$(C_RESET)"
	@echo "$(E_OK) Project installed!"

run:
	@echo "$(E_OK) Running $(NAME)..."
	@uv run --env-file=.env.hf $(NAME) $(ARGS)

debug:
	@echo "$(E_WAIT) Entering debug mode..."
	@uv run -m pdb -m src

test:
	@echo "$(E_WAIT) Running tests..."
	@uv run pytest tests
	@echo "$(E_OK) All good!!!"

clean:
	@echo "$(E_WAIT) Cleaning up..."
	@printf "$(C_GREY)";\
		find . \( -path "$(VENV_DIR)" -o -path ".git" \) -prune -o \
		\( -name "__pycache__" -o -name ".mypy_cache" -o -name ".ruff_cache" -o -name "*.egg-info" \) \
		-type d -exec rm -rf {} +; \
		printf "$(C_RESET)"
	@echo "$(E_OK) All clean!"

lint:
	@echo "$(E_WAIT) Running flake8..."
	@printf "$(C_GREY)";
	@uv run flake8 --isolated --exclude $(VENV_DIR) --color never $(SRC_DIR) || { \
		printf "$(C_RESET)"; echo "$(E_KO) flake8 failed!!!!"; exit 1; }
	@printf "$(C_RESET)"
	@echo "$(E_WAIT) Running mypy..."
	@printf "$(C_GREY)"; \
		uv run mypy $(SRC_DIR) $(MYPY_FLAGS) || { \
		printf "$(C_RESET)"; echo "$(E_KO) mypy failed!!!!"; exit 1; }; \
		printf "$(C_RESET)"
	@echo "$(E_OK) All good!"

lint-strict: MYPY_FLAGS = $(MYPY_FLAGS_STRICT)
lint-strict: lint
