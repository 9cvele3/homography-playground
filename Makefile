test:
	cargo test -- --nocapture

plot:
	gnuplot -e 'plot "/tmp/ecc.log" using 1 with lines title "ecc",	"/tmp/ecc.log" using 2 with lines title "tx", "/tmp/ecc.log" using 3 with lines title "ty"; pause -1'
