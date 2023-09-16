// https://docs.rs/arrsac/latest/arrsac/struct.Arrsac.html

use std::ops::Mul;

use imageproc::geometric_transformations::{Projection, warp_into, Interpolation};
use image::{self, GenericImageView};
use nalgebra::{SMatrix, DVector, SVector, DMatrix, Matrix, Const, Dynamic, VecStorage};
use rand::Rng;

const K: usize = 1000;
const N: usize = 8;
type Jacobian = DMatrix::<f32>;//Matrix::<f32, nalgebra::Const<K>, nalgebra::Const<N>, nalgebra::RawStorage<f32, nalgebra::Const<K>, nalgebra::Const<N>>>;
type FeatureVector = DVector::<f32>;//SVector::<f32, K>;
type ImgBuffer = image::ImageBuffer<image::Luma<u8>, Vec<u8>>;
type CMatrix = nalgebra::Matrix<f32, Dynamic, Const<1>, VecStorage<f32, Dynamic, Const<1>>>;

enum ParamsType {
    Trans,
    Proj,
}

struct Params {
    params: DVector::<f32>,
    ptype: ParamsType,
}

impl Params {
    fn new(ptype: ParamsType) -> Params {
        Params {
            params: DVector::<f32>::from_column_slice(&vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            ptype
        }
    }

    fn get_projection_matrix(&self) -> Option<Projection> {
        Projection::from_matrix([self.params[0], self.params[1], self.params[2], self.params[3], self.params[4], self.params[5], self.params[6], self.params[7], self.params[8]])
    }

    fn update(&mut self, inc: &DVector::<f32>, dump_factor: f32) {
        match self.ptype {
            ParamsType::Trans => {
                self.params[2] += inc[0] * dump_factor;
                self.params[5] += inc[1] * dump_factor;
            },
            ParamsType::Proj => {
                for i in 0..8 {
                    self.params[i] += inc[i] * dump_factor;
                }
            }
        }

    }

    fn print_params(&self) {
        println!("{} {} {} | {} {} {} | {} {}", self.params[0], self.params[1], self.params[2], self.params[3], self.params[4], self.params[5], self.params[6], self.params[7]);
    }
}


fn dump_feature_vector(fv: &FeatureVector) {

}

fn convert_luma_u8_to_luma_f32(img: &image::ImageBuffer<image::Luma<u8>, Vec<u8>>) -> ImgBuffer {
    let img_f = ImgBuffer::new(img.width(), img.height());



    img_f
}

#[test]
fn test_ecc() {
    test_ecc_impl();
}

pub fn test_ecc_impl() {
    use crate::reg::ecc;

    let paths = std::fs::read_dir("./img/").unwrap();

    for p in paths {
        if p.is_ok() {
            let path = p.unwrap().path();

            if path.to_str().unwrap().starts_with("./img/iter") {
                println!("{:?}", path);
                std::fs::remove_file(path);
            } else {
                println!("skip {:?}", path);
            }
        }
    }

    let img1 = image::open("img/lena-gray.png").expect("File not found!").grayscale().into_luma8();
    let img2 = image::open("img/lena-trans-x5-y10.png").expect("File not found!").grayscale().into_luma8();

//    let img1_f = ImgBuffer::new(img1.width(), img1.height());
//    let img2_f = ImgBuffer::new(img2.width(), img2.height());


    let proj = ecc(&img1, &img2).expect("Registration failed");
    println!("{:?}", proj);
}

fn get_feature_vector(im: &ImgBuffer, coords: &Vec<(u32, u32)>, normalize: bool) -> FeatureVector {
    let mut fv = FeatureVector::zeros(coords.len());

    for (i, (x, y)) in coords.iter().enumerate() {
        let pixel = get_pixel_value(im, *x, *y);
        fv[i] = pixel as f32;
    }

    let norm = fv.norm();
    fv.add_scalar_mut(-fv.mean());

    if normalize {
        for el in fv.iter_mut() {
            *el = *el / norm;
        }
    }

    fv
}

fn draw_points(img: &mut image::ImageBuffer<image::Luma<u8>, Vec<u8>>, X: &Vec<(u32, u32)>, P: &Params) {
    for (x0, x1) in X.iter() {
//        println!("{} {}", x0, x1);
        let (y0, y1) = warp_coords(P, (*x0, *x1));

        if let Some(pix) = img.get_pixel_mut_checked(y0, y1) {
            pix.0[0] = 255;
        }
    }
}

#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct GradInd {
    grad: i32,
    y: u32,
    x: u32
}

fn get_max_gradients(img: &image::ImageBuffer<image::Luma<u8>, Vec<u8>>) -> Vec<(u32, u32)> {
    use std::collections::BinaryHeap;

    let mut heap = BinaryHeap::new();
    let mut res = vec![];

    for y in 3..img.height()-3 {
        for x in 3..img.width()-3 {
            let gx =
                img.get_pixel(x, y - 1).0[0] as f32 + 2.0 * img.get_pixel(x, y - 1).0[0]  as f32 + img.get_pixel(x, y - 1 ).0[0] as f32
                - img.get_pixel(x, y + 1).0[0] as f32 - 2.0 * img.get_pixel(x, y + 1).0[0] as f32 - img.get_pixel(x, y + 1 ).0[0] as f32;

            let gy =
                img.get_pixel(x - 1, y).0[0] as f32 + 2.0 * img.get_pixel(x - 1, y).0[0] as f32 + img.get_pixel(x - 1, y).0[0] as f32
                - img.get_pixel(x + 1, y).0[0] as f32 - 2.0 * img.get_pixel(x + 1, y).0[0] as f32 - img.get_pixel(x + 1, y).0[0] as f32;

            let gxy = gx.abs() + gy.abs();

            heap.push((gxy as i32, y, x));
        }
    }

    for _k in 1..100 {
        let (_, y, x) = heap.pop().unwrap();

        println!("{} {}", y, x);
        res.push((y, x));
    }

    res
}

fn draw_max_gradients(img: &mut image::ImageBuffer<image::Luma<u8>, Vec<u8>>) {
    let max_gradients = get_max_gradients(img);

    for y in 0..img.height() {
        for x in 0..img.width() {
            if let Some(pix) = img.get_pixel_mut_checked(x, y) {
                pix.0[0] = 0;
            }
        }
    }

    for (yi, xi) in max_gradients {
        if let Some(pix) = img.get_pixel_mut_checked(xi, yi) {
            pix.0[0] = 255;
        }
    }
}

#[allow(non_snake_case)]
fn ecc(Ir: &ImgBuffer, Iw: &ImgBuffer) -> Option<Projection> {
    let mut w = std::fs::File::create("/tmp/ecc.log").unwrap();
    use std::io::Write;

    let mut num_points = K;
    let mut border = 40;
    let mut X = get_k_points(num_points, Ir.width(), Ir.height(), false, border);
    let mut ir = get_feature_vector(Ir, &X, true);


    let mut P = Params::new(ParamsType::Trans);

    let threshold = 0.0001;
    let max_num_iter = 50;
    let mut num_iter = 0;
    let mut dump_factor = 5.0;
    let mut ecc_coeff_max = -1000.0;
    let mut h3_best = None;

    loop {
        println!("##########################\niteration {}", num_iter);

        // warp into
        let mut Imgw = ImgBuffer::new(Ir.width(), Ir.height());

        let h3 = P.get_projection_matrix().expect("Unable to form projection matrix").invert(); // use inverse matrix here

        warp_into(&Iw, &h3, Interpolation::Bilinear, [0].into(), &mut Imgw);

        if true {
            let mut Imgw_clone = Imgw.clone();
            //draw_points(&mut Imgw_clone, &X, &P);
            //draw_max_gradients(&mut Imgw_clone);
            Imgw_clone.save(format!("img/iter{}.png", num_iter));
        }

        let iw = get_feature_vector(&Imgw, &X, false);

        let G = calculate_jacobian(&Imgw, &X, &P); // dims: K x N = 1000 x 8
        let GT = G.transpose(); // dims: N x K = 8 x 1000
        let GT_G = GT.mul(&G); // dims: N x N = 8 x 8, Hessain
        let GT_G_inv_GT = (GT_G).pseudo_inverse(0.0000000000001).unwrap() * G.transpose(); // dims: N x K = 8 x 1000
        let PG = G.clone() * GT_G_inv_GT.clone(); // dims: K x K = 1000 x 1000

        // is PG projection ? projection == idempotent : once projected, always projected
        // PG * PG = (G * (GT * G) ^ {-1} * GT) * (G * (GT * G) ^ {-1} * GT)
        // PG * PG = G * (GT * G) ^ {-1} * (GT * G) * (GT * G) ^ {-1} * GT
        // PG * PG = G * (GT * G) ^ {-1} * GT = PG => PG is a projection

        let ir_iw = (ir.transpose() * iw.clone())[(0, 0)];
        let ir_pg_iw = (ir.transpose() * PG.clone() * iw.clone())[(0, 0)];
        let iw_pg_iw = (iw.transpose() * PG.clone() * iw.clone())[(0, 0)];

        println!("ir mean {}, iw mean {}", ir.mean(), iw.mean());

        {
            let inc1 = increment1(&ir, &iw, &GT_G_inv_GT, ir_iw, ir_pg_iw, iw_pg_iw);
            let ir_pg_ir = (ir.transpose() * PG.clone() * ir.clone())[(0, 0)];
            let inc2 = increment2(&ir, &iw, &GT_G_inv_GT, ir_iw, iw_pg_iw, ir_pg_ir, ir_pg_iw);

            println!("inc1 {:?} inc2 {:?}", inc1, inc2);
        }

        let inc = if ir_iw > ir_pg_iw {
            println!("inc1, {} > {}", ir_iw, ir_pg_iw);
            increment1(&ir, &iw, &GT_G_inv_GT, ir_iw, ir_pg_iw, iw_pg_iw)
        } else {
            println!("inc2, {} <= {}", ir_iw, ir_pg_iw);
            let ir_pg_ir = (ir.transpose() * PG * ir.clone())[(0, 0)];
            increment2(&ir, &iw, &GT_G_inv_GT, ir_iw, iw_pg_iw, ir_pg_ir, ir_pg_iw)
        };

        println!("inc.norm {}", inc.norm());

        // ecc_coef is in range [0, 1];
        let ecc_coeff = (ir.transpose() * iw.clone() / iw.norm())[(0, 0)];
        let ecc_coeff_approximaiton = (ir.transpose() * (iw.clone() + G.clone() * inc.clone()) / (iw.clone() + G.clone()*inc.clone()).norm())[(0, 0)];

        if ecc_coeff < ecc_coeff_max {
            //border += 10;
            //X = get_k_points(num_points, Ir.width(), Ir.height(), false, border);
            //ir = get_feature_vector(Ir, &X, true);
            //dump_factor *= 1.05;
        } else {
            ecc_coeff_max = ecc_coeff;
            h3_best = P.get_projection_matrix();
        }

        println!("ecc_coeff: {}, ecc_coeff_approximation: {}", ecc_coeff, ecc_coeff_approximaiton);
        writeln!(&mut w, "{} {} {}", ecc_coeff, P.params[2], P.params[5]).unwrap();

        if inc.norm() < threshold {
            break;
        } else if num_iter < max_num_iter {
            for inc_el in inc.iter() {
                println!("inc el {}", inc_el);
            }

            P.update(&inc, dump_factor);
            P.print_params();

            num_iter += 1;
        } else {
            break;
        }
    }

    h3_best
}

//  increment: coeff * ir - iw
//  to 1xN matrix: GT_G_inv_GT * increment
//
#[allow(non_snake_case)]
fn increment1(ir: &FeatureVector, iw: &FeatureVector, GT_G_inv_GT: &DMatrix::<f32>, ir_iw: f32, ir_pg_iw: f32, iw_pg_iw: f32) -> CMatrix {
    GT_G_inv_GT * (((iw.norm() * iw.norm() - iw_pg_iw) / (ir_iw - ir_pg_iw))* ir - iw)
}

#[allow(non_snake_case)]
fn increment2(ir: &FeatureVector, iw: &FeatureVector, GT_G_inv_GT: &DMatrix::<f32>, ir_iw: f32, iw_pg_iw: f32, ir_pg_ir: f32, ir_pg_iw: f32) -> CMatrix {
    let ir_pg_ir = ir_pg_ir;
    let lambda1 = (iw_pg_iw / ir_pg_ir).sqrt();
    let lambda2 = (ir_pg_iw - ir_iw) / ir_pg_ir;

    let lambda = lambda1.max(lambda2);
    GT_G_inv_GT * (lambda * ir - iw)
}

fn get_k_points(k: usize, w: u32, h: u32, rand: bool, border: u32) -> Vec<(u32, u32)> {
    let inc = (((w - border) * (h - border)) as f32 / (2.0 * k as f32)) as u32;
    let mut rng = rand::thread_rng();

    let border = std::cmp::min(border, 150);
    let mut ind = border * w + border;
    let mut res = vec![];

    while ind < w * (h - border) && res.len() < k {
        let x = border + ind % (w - 2 * border);
        let y = ind / h;

        res.push((x, y));           res.push((x, y+1));

        ind += inc;

        if rand {
            ind += rng.gen_range(0..=3);
        }
    }

    res = res.drain(res.len() - k..).collect();

    println!("K points: {}", res.len());

    res
}

fn warp_coords(P: &Params, X: (u32, u32)) -> (u32, u32) {
    let y1 =    P.params[0] * (X.0 as f32)      + P.params[1] * (X.1 as f32)    + P.params[2];
    let y2 =    P.params[3] * (X.0 as f32)      + P.params[4] * (X.1 as f32)    + P.params[5];
    let den =   P.params[6] * (X.0 as f32)      + P.params[7] * (X.1 as f32)    + 1.0;

    ((0.5 + y1 / den) as u32, (0.5 + y2 / den) as u32)
}

fn get_pixel_value(im: &ImgBuffer, x: u32, y: u32) -> f32 {
    im.get_pixel_checked(x, y)
        .and_then(|e| Some(e.0[0] as f32))
        .unwrap_or(0.0)
}

fn calculate_jacobian(Iw: &ImgBuffer, X: &Vec<(u32, u32)>, P: &Params) -> Jacobian {
    match P.ptype {
        ParamsType::Trans => {
            calculate_jacobian_trans(Iw, X, P)
        },
        ParamsType::Proj => {
            calculate_jacobian_proj(Iw, X, P)
        }
    }
}

#[allow(non_snake_case)]
fn calculate_jacobian_trans(Iw: &ImgBuffer, X: &Vec<(u32, u32)>, P: &Params) -> Jacobian {
    let mut G = Jacobian::zeros(K, 2);

    for (k, (x0, x1)) in X.iter().enumerate() {
        let Y = warp_coords(&P, (*x0, *x1));
        let dI_per_dy1 = get_pixel_value(Iw, Y.0 + 1, Y.1) - get_pixel_value(Iw, Y.0, Y.1);
        let dI_per_dy2 = get_pixel_value(Iw, Y.0, Y.1 + 1) - get_pixel_value(Iw, Y.0, Y.1);

        // N = 0
        {
            let dphi_ofx1_per_dp0 = 1.0;
            let dphi_ofx2_per_dp0 = 0.0;
            let el = dI_per_dy1 * dphi_ofx1_per_dp0 + dI_per_dy2 * dphi_ofx2_per_dp0;
            G[(k, 0)] = el;
        }

        // N = 1
        {
            let dphi_ofx1_per_dp1 = 0.0;
            let dphi_ofx2_per_dp1 = 1.0;
            let el = dI_per_dy1 * dphi_ofx1_per_dp1 + dI_per_dy2 * dphi_ofx2_per_dp1;
            G[(k, 1)] = el;
        }
    }

    zero_mean_over_columns(&mut G);

    G
}

#[allow(non_snake_case)]
fn calculate_jacobian_proj(Iw: &ImgBuffer, X: &Vec<(u32, u32)>, P: &Params) -> Jacobian {
    let mut G = Jacobian::zeros(K, 8);

    for (k, (x0, x1)) in X.iter().enumerate() {
        let Y = warp_coords(&P, (*x0, *x1));
        let dI_per_dy1 = get_pixel_value(Iw, Y.0 + 1, Y.1) - get_pixel_value(Iw, Y.0, Y.1);
        let dI_per_dy2 = get_pixel_value(Iw, Y.0, Y.1 + 1) - get_pixel_value(Iw, Y.0, Y.1);
/*
        p0  p1  p2      x0      p0x0+p1x1+p2
        p3  p4  p5  *   x1  =   p3x0+p4x1+p5
        p6  p7  1      1        p6x0+p7x1+1

        phi1 = (p0x0 + p1x1 + p2) / (p6x0 + p7x1 + 1)
        phi2 = (p3x0 + p4x1 + p5) / (p6x0 + p7x1 + 1)
*/

        let numerator1 = P.params[0] * (*x0 as f32) + P.params[1] * (*x1 as f32) + P.params[2];
        let numerator2 = P.params[3] * (*x0 as f32) + P.params[4] * (*x1 as f32) + P.params[5];
        let denominator = P.params[6] * (*x0 as f32) + P.params[7] * (*x1 as f32) + 1.0;

        // N = 0
        {
            let dphi_ofx1_per_dp0 = *x0 as f32 / denominator;
            let dphi_ofx2_per_dp0 = 0.0;
            let el = dI_per_dy1 * dphi_ofx1_per_dp0 + dI_per_dy2 * dphi_ofx2_per_dp0;
            G[(k, 0)] = el;
        }
        // N = 1
        {
            let dphi_ofx1_per_dp1 = *x1 as f32 / denominator;
            let dphi_ofx2_per_dp1 = 0.0;
            let el = dI_per_dy1 * dphi_ofx1_per_dp1 + dI_per_dy2 * dphi_ofx2_per_dp1;
            G[(k, 1)] = el;
        }
        // N = 2
        {
            let dphi_ofx1_per_dp2 = 1.0 / denominator;
            let dphi_ofx2_per_dp2 = 0.0;
            let el = dI_per_dy1 * dphi_ofx1_per_dp2 + dI_per_dy2 * dphi_ofx2_per_dp2;
            G[(k, 2)] = el;
        }
        // N = 3
        {
            let dphi_ofx1_per_dp3 = 0.0;
            let dphi_ofx2_per_dp3 = *x0 as f32 / denominator;
            let el = dI_per_dy1 * dphi_ofx1_per_dp3 + dI_per_dy2 * dphi_ofx2_per_dp3;
            G[(k, 3)] = el;
        }
        // N = 4
        {
            let dphi_ofx1_per_dp4 = 0.0;
            let dphi_ofx2_per_dp4 = *x1 as f32 / denominator;
            let el = dI_per_dy1 * dphi_ofx1_per_dp4 + dI_per_dy2 * dphi_ofx2_per_dp4;
            G[(k, 4)] = el;
        }
        // N = 5
        {
            let dphi_ofx1_per_dp5 = 0.0;
            let dphi_ofx2_per_dp5 = 1.0 / denominator;
            let el = dI_per_dy1 * dphi_ofx1_per_dp5 + dI_per_dy2 * dphi_ofx2_per_dp5;
            G[(k, 5)] = el;
        }
        // N = 6
        {
            let dphi_ofx1_per_dp6 = - numerator1 * (*x0 as f32) / (denominator * denominator);
            let dphi_ofx2_per_dp6 = - numerator2 * (*x0 as f32) / (denominator * denominator);
            let el = dI_per_dy1 * dphi_ofx1_per_dp6 + dI_per_dy2 * dphi_ofx2_per_dp6;
            G[(k, 6)] = el;
        }
        // N = 7
        {
            let dphi_ofx1_per_dp7 = - numerator1 * (*x1 as f32) / (denominator * denominator);
            let dphi_ofx2_per_dp7 = - numerator2 * (*x1 as f32) / (denominator * denominator);
            let el = dI_per_dy1 * dphi_ofx1_per_dp7 + dI_per_dy2 * dphi_ofx2_per_dp7;
            G[(k, 7)] = el;
        }
    } // for

    zero_mean_over_columns(&mut G);

    G
}

fn zero_mean_over_columns(G: &mut DMatrix::<f32>) {
    // let GColMean = G.row_mean(); // nalgebra implementation is strage, row_mean returns col_mean actually

    let mut GColMean = DMatrix::<f32>::zeros(1, N);

    for y in 0..G.nrows() {
        for x in 0..G.ncols() {
            GColMean[x] += G[(y, x)];
        }
    }

    for x in 0..G.ncols() {
        GColMean[x] = GColMean[x] / G.nrows() as f32;
    }

    // zero mean G over columns
    for y in 0..G.nrows() {
        for x in 0..G.ncols() {
            G[(y, x)] -= GColMean[x];
        }
    }
}

/*
todos:
* svd epsilon
* normalize G - done
* normalize ir - done
* form ir - done
* Image to ImageBuffer - done
* gradients - done
* warp image - done
* Projection - unable to form
* feature vector - zero mean ? normalized ?
*/





