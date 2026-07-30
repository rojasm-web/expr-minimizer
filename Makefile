# Top-level Makefile for the NetBeans "Makefile Project" wrapper.
# The real build system is Cargo; this just gives NetBeans's generic
# Build/Clean/Run actions something conventional to call.

.PHONY: all build clean run test check release

all: build

build:
	cargo build

release:
	cargo build --release

clean:
	cargo clean

run: build
	cargo run

test:
	cargo test

check:
	cargo check
