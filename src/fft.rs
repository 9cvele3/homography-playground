use nalgebra::{Complex, ComplexField};

#[derive(PartialEq, Default)]
pub enum FFTType {
    Horizontal,
    Vertical,
    #[default]
    TwoDimensional,
}

#[derive(Default)]
pub struct FFTParams {
    pub central: bool,
    pub fft_type: FFTType,
}

pub fn fft_2D(img: &egui::ColorImage, params: &FFTParams) -> egui::ColorImage {
    let mut pixels = egui_img_2_complex_vec(img);

    // todo: rayon
    // fft for rows
    let central = params.central;

    if matches!(params.fft_type, FFTType::Horizontal) || matches!(params.fft_type, FFTType::TwoDimensional) {
        for y in 0..img.height() {
            fft_1D(&mut pixels[y*img.width() ..], img.width(), 1, central);
        }
    }

    // todo: rayon

    if matches!(params.fft_type, FFTType::Vertical) || matches!(params.fft_type, FFTType::TwoDimensional) {
        for x in 0..img.width() {
            fft_1D(&mut pixels[x..], img.height(), img.width(), central);
        }
    }

    if false {
        // swap 1st and 3rd quadrant
        for y in 0..img.height() / 2 {
            for x in 0..img.width() / 2 {
                let tmp = pixels[y * img.width() + x];
                pixels[y * img.width() + x] = pixels[(y + img.height()/2) * img.width() + x + img.width()/2];
                pixels[(y + img.height()/2) * img.width() + x + img.width()/2] = tmp;
            }
        }

        // swap 2nd and 4th quadrant
        for y in 0..img.height() / 2 {
            for x in img.width() / 2 .. img.width() {
                let tmp = pixels[y * img.width() + x];
                pixels[y * img.width() + x] = pixels[(y + img.height()/2) * img.width() + x - img.width()/2];
                pixels[(y + img.height()/2) * img.width() + x - img.width()/2] = tmp;
            }
        }
    }

    // todo: smart schedule for fft 2D ?

    complex_2_egui_img(&pixels, img.size)
}

/*
    r   r   r   r   r   r   r   r   |   r   r   r   r   r   r   r   r     |   r     r   r   r   r   r   r   r   |


    c1   c2   c3   c4   c5   c6   c7   c8   |    c1   c2   c3   c4   c5   c6   c7   c8   |    c1   c2   c3   c4   c5   c6   c7   c8   |

*/
// is this slice type ?
fn fft_1D(input: &mut [Complex<f32>], len: usize, pitch: usize, central: bool) {
    // shuffle input
    {
        let mut tmp: Vec<Complex<f32>> = vec![Complex::<f32>::new(1_f32, 0_f32); len];

        for i in 0..len/2 {
            if i % 2 == 0 {
                tmp[i] = input[i * pitch];
            } else {
                tmp[i] = input[(i+len/2 - 1) * pitch];
            }
        }

        for i in 0..len/2 {
            if i % 2 == 0 {
                tmp[len/2 + i] = input[(i + 1) * pitch];
            } else {
                tmp[len/2 + i] = input[(i + 1 + len/2 - 1) * pitch];
            }
        }

        for i in 0..len {
            input[i * pitch] = tmp[i];
        }
    }

    // println!("{:?}", input);

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
                // println!("ind1 {}, ind2 {}, w {:?}, {}, {}", ind1, ind2, w, input[ind1], input[ind2]);
                let left = input[ind1]  + w * input[ind2];
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

    if central {
        for i in 0..len/2 {
            let tmp = input[i * pitch];
            input[i * pitch] = input[(i + len/2) * pitch];
            input[(i + len/2) * pitch] = tmp;
        }
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

#[test]
fn test_fft_2() {
    let mut input: Vec<Complex<f32>> = (1..3).into_iter()
                                        .map(|v| {
                                                Complex::<f32>::new(v as f32, 0_f32)
                                        })
                                        .collect();

    println!("{:?}", input);

    let len = input.len();
    fft_1D(&mut input, len, 1, false);

    let output = vec![
        Complex::<f32>::new(3.0000,     0.0),
        Complex::<f32>::new(-1.0000,     0.0),
    ];

    assert_eq!(output, input);
}

#[test]
fn test_fft_4() {
    let mut input: Vec<Complex<f32>> = (1..5).into_iter()
                                        .map(|v| {
                                                Complex::<f32>::new(v as f32, 0_f32)
                                        })
                                        .collect();

    let len = input.len();
    fft_1D(&mut input, len, 1, false);

    let output = vec![
        Complex::<f32>::new(10.0000,     0.0),
        Complex::<f32>::new(-2.0000,     2.0),
        Complex::<f32>::new(-2.0000,     0.0),
        Complex::<f32>::new(-2.0000,    -2.0),
    ];

    assert_eq!(output, input);
}

#[test]
fn test_fft_8() {
    let mut input: Vec<Complex<f32>> = (1..9).into_iter()
                                        .map(|v| {
                                                Complex::<f32>::new(v as f32, 0_f32)
                                        })
                                        .collect();

    let len = input.len();
    fft_1D(&mut input, len, 1, false);

    let output = vec![
        Complex::<f32>::new(36.0000,     0.0),
        Complex::<f32>::new(-4.0000,  9.6569),
        Complex::<f32>::new(-4.0000,  4.0000),
        Complex::<f32>::new(-4.0000,  1.6569),
        Complex::<f32>::new(-4.0000,     0.0),
        Complex::<f32>::new(-4.0000,  1.6569),
        Complex::<f32>::new(-4.0000,  4.0000),
        Complex::<f32>::new(-4.0000,  9.6569)
    ];

    assert_eq!(output, input);
}