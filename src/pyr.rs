use imageproc::{filter::filter3x3, geometric_transformations::{warp, Projection, Interpolation, warp_into}};


type ImgBufferU8 = image::ImageBuffer<image::Luma<u8>, Vec<u8>>;

type ImgBufferF = image::ImageBuffer<image::Luma<f32>, Vec<f32>>;

fn convert_luma_u8_to_luma_f32(img: &ImgBufferU8) -> ImgBufferF {
    let mut img_f = ImgBufferF::new(img.width(), img.height());

    for (pixel_u8, pixel_f32) in img.as_raw().iter().zip(img_f.as_mut().iter_mut()) {
        *pixel_f32 = (*pixel_u8).into();
    }

    img_f
}

fn convert_luma_f32_to_luma_u8(img_f: &ImgBufferF) -> ImgBufferU8 {
    let mut img_u8 = ImgBufferU8::new(img_f.width(), img_f.height());

    for (pixel_f32, pixel_u8) in img_f.as_raw().iter().zip(img_u8.as_mut().iter_mut()) {
        *pixel_u8 = (*pixel_f32).clamp(0.0, 255.0) as u8;
    }

    img_u8
}

#[allow(non_snake_case)]
fn smooth(I: &ImgBufferF) -> ImgBufferF {
    let kernel = [
        0.0625,     0.125,  0.0625,
        0.125,       0.25,  0.125,
        0.0625,     0.125,  0.0625,
    ];

    filter3x3(&I, &kernel)
}

#[allow(non_snake_case)]
fn downscale(I: &ImgBufferF) -> ImgBufferF {
    let mut out = ImgBufferF::new(I.width() / 2, I.height() / 2);

    warp_into(I, &Projection::scale(0.5, 0.5), Interpolation::Bilinear, [0.0].into(), &mut out);

    out
}

#[allow(non_snake_case)]
pub fn create_pyramid(I: &ImgBufferU8) -> Vec<ImgBufferF> {
    let wf = I.width() as f32;
    let hf = I.height() as f32;
    let num_levels = std::cmp::min(5, std::cmp::min(wf.log2() as usize, hf.log2() as usize));

    let mut res = vec![];
    res.push(convert_luma_u8_to_luma_f32(I));

    for i in 1..num_levels {
        let Ismooth = smooth(&res[i - 1]);
        res.push(downscale(&Ismooth));
    }

    res
}

#[test]
fn test_pyr() {
    let img1 = image::open("img/lena-gray.png").expect("File not found!").grayscale().into_luma8();
    let pyr = create_pyramid(&img1);

    for (ind, p) in pyr.iter().enumerate() {
        let p_u8 = convert_luma_f32_to_luma_u8(p);
        let _ = p_u8.save(format!("img/pyr{}.png", ind));
    }
}


