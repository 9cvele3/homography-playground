use nalgebra::{Complex, ComplexField};

pub fn fft_2D(img: &egui::ColorImage) -> egui::ColorImage {
    let mut pixels = egui_img_2_complex_vec(img);

    // todo: rayon
    for y in 0..img.height() {
        fft_1D(&mut pixels[y*img.width() ..], img.width(), 1);
    }

    // todo: rayon
    for x in 0..img.width() {
        fft_1D(&mut pixels[x*img.height() ..], img.height(), img.width());
    }

    // todo: smart schedule for fft 2D ?

    complex_2_egui_img(&pixels, img.size)
}

// is this slice type ?
fn fft_1D(input: &mut [Complex<f32>], len: usize, pitch: usize) {
    let levels = (len as f32).log2().ceil() as usize;

    let mut k = pitch;

    for l in 1..levels+1 {
        let angle = l as f32 * 2.0 * 3.1415 / (levels as f32);
        let w = Complex::<f32>::new(angle.cos(), angle.sin());

        for i in 0..len-k {
            let left = input[i] + w * input[i + k];
            let right = input[i] - w * input[i + k];

            input[i] = left;
            input[i + k] = right;
        }

        k *= 2;
    }
}

fn egui_img_2_complex_vec(im: &egui::ColorImage) -> Vec<Complex<f32>> {
    let size = im.size;
    let mut pixels = Vec::with_capacity(size[0]*size[1]);

    for pix in im.pixels.iter() {
        pixels.push(Complex::<f32>::new((pix.r() + pix.g() + pix.b()) as f32 / 3.0, 0_f32));
    }

    pixels
}

fn complex_2_egui_img(complex: &Vec<Complex<f32>>, out_size: [usize; 2]) -> egui::ColorImage {
    let pixels = complex.iter().map(|p| {
            egui::Color32::from_gray(p.abs() as u8)
        })
        .collect();

    egui::ColorImage{size: out_size, pixels}
}

