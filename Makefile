test:
	cargo test -- --nocapture

# LD_LIBRARY_PATH - problems when starting from VSCode
plot:
	gnuplot -e 'plot "/tmp/ecc.log" using 1 with lines title "ecc",	"/tmp/ecc.log" using 2 with lines title "tx", "/tmp/ecc.log" using 3 with lines title "ty"; pause -1'

