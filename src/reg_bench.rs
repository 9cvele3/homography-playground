
#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use crate::reg::test_ecc_impl;


    #[bench]
    fn bench_ecc(b: &mut Bencher) {
        b.iter(|| {
            test_ecc_impl();
        });
    }
}

