default:
	@cargo build
	@target/debug/toylang main.tl
	@readelf -a main
	@objdump -d main
	@./main
