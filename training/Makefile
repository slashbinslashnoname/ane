CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface
LDFLAGS = $(FRAMEWORKS) -ldl
PYTHON ?= python3

HEADERS_LARGE = stories_config.h stories_io.h stories_mil.h stories_cpu_ops.h
DATA = train_data00.bin

.DEFAULT_GOAL := all
.PHONY: all run resume dashboard data build probes clean clean-data help

help:
	@echo "Targets:"
	@echo "  make            data + build + run training"
	@echo "  make data       fetch tokenizer + dataset, tokenize"
	@echo "  make build      compile train_large"
	@echo "  make run        run training (auto-builds + fetches data)"
	@echo "  make resume     resume training from checkpoint"
	@echo "  make dashboard  run TUI dashboard with training"
	@echo "  make probes     build ANE probe binaries"
	@echo "  make clean      remove built binaries"
	@echo "  make clean-data remove tokenized data"

all: run

run: build $(DATA)
	./train_large

resume: build $(DATA)
	./train_large --resume

dashboard: build $(DATA)
	$(PYTHON) dashboard.py --resume

build: train_large

data: $(DATA)

$(DATA): prepare_data.py
	$(PYTHON) prepare_data.py

train_large: train_large.m $(HEADERS_LARGE)
	$(CC) $(CFLAGS) -o $@ train_large.m $(LDFLAGS) -framework Accelerate

# Legacy small toy training binary
train: train.m ane_runtime.h ane_mil_gen.h model.h forward.h backward.h
	$(CC) $(CFLAGS) -o $@ train.m $(LDFLAGS)

PROBES = test_weight_reload test_perf_stats test_qos_sweep test_ane_advanced
probes: $(PROBES)
$(PROBES): %: %.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f train train_large $(PROBES)

clean-data:
	rm -f $(DATA)
