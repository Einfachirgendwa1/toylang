default:
	@cargo run -- main.tl
	@readelf -a main
	@objdump -d main
	@./main
