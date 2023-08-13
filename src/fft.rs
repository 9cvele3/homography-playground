use nalgebra::{Complex, ComplexField};

pub fn fft_2D(img: &egui::ColorImage) -> egui::ColorImage {
    let mut pixels = egui_img_2_complex_vec(img);

    // todo: rayon
    // fft for rows

    /*
    for y in 0..img.height() {
        fft_1D(&mut pixels[y*img.width() ..], img.width(), 1);
    }
    */

    // todo: rayon
    for x in 0..img.width() {
        fft_1D(&mut pixels[x..], img.height(), img.width());
    }

    // todo: smart schedule for fft 2D ?

    complex_2_egui_img(&pixels, img.size)
}

/*
    r   r   r   r   r   r   r   r   |   r   r   r   r   r   r   r   r     |   r     r   r   r   r   r   r   r   |


    c1   c2   c3   c4   c5   c6   c7   c8   |    c1   c2   c3   c4   c5   c6   c7   c8   |    c1   c2   c3   c4   c5   c6   c7   c8   |

*/
// is this slice type ?
fn fft_1D(input: &mut [Complex<f32>], len: usize, pitch: usize) {
    let levels = (len as f32).log2().ceil() as usize;

    let mut k = 1;

    for _l in 0..levels {
        let elements_per_group = 2 * k;
        let butterflies_per_group = k;
        let angle = 2.0 * 3.1415 / (elements_per_group as f32);

        for group in 0..len/elements_per_group {
            for butterfly in 0..butterflies_per_group {
                let ind1 = (group * elements_per_group + butterfly) * pitch;
                let ind2 = ind1 + k * pitch;

                if ind1 >= input.len() || ind2 >= input.len() {
                    println!("group ind {}, group_size {}, butterfly {}, ind1 {}, ind2 {}", group, elements_per_group, butterfly, ind1, ind2);
                }


                let w = Complex::<f32>::new((butterfly as f32 * angle).cos(), (butterfly as f32 * -angle).sin());
                let left = input[ind1] + w * input[ind2];
                let right = input[ind1] - w * input[ind2];

                input[ind1] = left;
                input[ind2] = right;
            }
        }

        k *= 2;
    }

    for i in 0..len {
        input[i * pitch] = input[i * pitch] / (len as f32).sqrt();
        //println!("{}", input[i]);
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
            let val = p.abs();
            //println!("val: {}", val);
            egui::Color32::from_gray(val as u8)
        })
        .collect();

    egui::ColorImage{size: out_size, pixels}
}

