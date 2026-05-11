.PHONY: build run test check clean fmt clippy

build:
	cargo build

run:
	cargo run

test:
	cargo test --test $(NAME)

check:
	cargo check

clean:
	cargo clean

fmt:
	cargo fmt

clippy:
	cargo clippy -- -D warnings