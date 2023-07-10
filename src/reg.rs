// https://docs.rs/arrsac/latest/arrsac/struct.Arrsac.html

use std::ops::Mul;

use imageproc::geometric_transformations::{Projection, warp_into, Interpolation};
use image::{self};
use nalgebra::{SMatrix, DVector, SVector, DMatrix, Matrix};

#[test]
fn test_ecc() {
    let img1 = image::open("img/lena-gray.png").expect("File not found!").grayscale().into_luma8();
    let img2 = image::open("img/lena-trans-10.png").expect("File not found!").grayscale().into_luma8();

    let proj = ecc(&img1, &img2).expect("Registration failed");
}

const K: usize = 1000;
const N: usize = 9;
type Jacobian = DMatrix::<f32>;//Matrix::<f32, nalgebra::Const<K>, nalgebra::Const<N>, nalgebra::RawStorage<f32, nalgebra::Const<K>, nalgebra::Const<N>>>;
type Params = DVector::<f32>;//Matrix::<f32, nalgebra::Const<N>, nalgebra::U1, nalgebra::ArrayStorage<f32, N, 1>>;
type FeatureVector = DVector::<f32>;//SVector::<f32, K>;

fn get_feature_vector(im: &image::ImageBuffer<image::Luma<u8>, Vec<u8>>, coords: &Vec<(u32, u32)>, normalize: bool) -> FeatureVector {
    let mut fv = FeatureVector::zeros(K);

    for (i, (x, y)) in coords.iter().enumerate() {
        let pixel = get_pixel_value(im, *x, *y);
        fv[i] = pixel as f32;
    }

    fv.add_scalar(-fv.mean());

    if normalize {
        let norm = fv.norm();

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
        img.get_pixel_mut(y0, y1).0[0] = 255;
    }
}

#[allow(non_snake_case)]
fn ecc(Ir: &image::ImageBuffer<image::Luma<u8>, Vec<u8>>, Iw: &image::ImageBuffer<image::Luma<u8>, Vec<u8>>) -> Option<Projection> {
    let X = get_k_points(K, Ir.width(), Ir.height());

    let mut P: Params = Params::from_column_slice(&vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let threshold = 0.01;
    let max_num_iter = 1000;
    let mut num_iter = 0;
    let ir = get_feature_vector(Ir, &X, true);

    loop {
        println!("iteration {}", num_iter);

        // warp into
        let mut Imgw: image::ImageBuffer<image::Luma<u8>, Vec<_>> = image::ImageBuffer::new(Ir.width(), Ir.height());
        let h3 = Projection::from_matrix([P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8]]).expect("Invalid homography");
        warp_into(&Iw, &h3, Interpolation::Bilinear, [0].into(), &mut Imgw);
        //draw_points(&mut Imgw, &X, &P);
        Imgw.save(format!("img/iter{}.png", num_iter));
        let iw = get_feature_vector(&Imgw, &X, false);

        // ecc_coef is in range [0, 1];
        let ecc_coeff = (ir.transpose() * iw.clone() / iw.norm())[(0, 0)];
        println!("ecc_coeff {}", ecc_coeff);


        let G = calculate_jacobian(&Imgw, &X, &P);
        let GT = G.transpose();
        let GT_G = GT.mul(&G);
        let GT_G_inv_GT = (GT_G).pseudo_inverse(0.00000001).unwrap() * G.transpose();
        let PG = G.clone() * GT_G_inv_GT.clone();

        let ir_iw = (ir.transpose() * iw.clone())[(0, 0)];
        let ir_pg_iw = (ir.transpose() * PG.clone() * iw.clone())[(0, 0)];
        let iw_pg_iw = (iw.transpose() * PG.clone() * iw.clone())[(0, 0)];

        let inc = if ir_iw > ir_pg_iw {
            println!("inc1, {} > {}", ir_iw, ir_pg_iw);
            increment1(&ir, &iw, &GT_G_inv_GT, ir_iw, ir_pg_iw, iw_pg_iw)
        } else {
            println!("inc2, {} <= {}", ir_iw, ir_pg_iw);
            let ir_pg_ir = (ir.transpose() * PG * ir.clone())[(0, 0)];
            increment2(&ir, &iw, &GT_G_inv_GT, ir_iw, iw_pg_iw, ir_pg_ir, ir_pg_iw)
        };

        println!("inc.norm {}", inc.norm());
        //println!("inc: {} {} {} \n {} {} {} \n {} {} {}", inc[0], inc[1], inc[2], inc[3], inc[4], inc[5], inc[6], inc[7], inc[8]);
        let ecc_coeff_approximaiton = (ir.transpose() * (iw.clone() + G.clone() * inc.clone()) / (iw.clone() + G.clone()*inc.clone()).norm())[(0, 0)];
        println!("ecc_coeff_approximation: {}", ecc_coeff_approximaiton);

        if inc.norm() < threshold {
            return Projection::from_matrix([P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8]]);
        } else if num_iter < max_num_iter {
            for i in 0..inc.len() {
                P[i] -= inc[i];
            }
            let proj =  Projection::from_matrix([P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8]]);
            println!("proj: {:?}", proj);
            num_iter += 1;
        } else {
            break;
        }
    }

    None
}

fn get_k_points(k: usize, w: u32, h: u32) -> Vec<(u32, u32)> {
    let inc = (((w - 10) * (h - 10)) as f32/ k as f32) as u32;

    let mut ind = 10;
    let mut res = vec![];

    while ind < w * h && res.len() < k {
        let x = ind % w;
        let y = ind / h;
        res.push((x, y));
        ind += inc;
    }

    res
}

fn warp_coords(P: &Params, X: (u32, u32)) -> (u32, u32) {
    let y1 = P[0] * (X.0 as f32) + P[1] * (X.1 as f32) + P[2];
    let y2 = P[3] * (X.0 as f32) + P[4] * (X.1 as f32) + P[5];
    let den = P[6] * (X.0 as f32) + P[7] * (X.1 as f32) + P[8];

    ((0.5 + y1 / den) as u32, (0.5 + y2 / den) as u32)
}

fn get_pixel_value(im: &image::ImageBuffer<image::Luma<u8>, Vec<u8>>, x: u32, y: u32) -> f32 {
    im.get_pixel_checked(x, y)
        .and_then(|e| Some(e.0[0] as f32))
        .unwrap_or(0.0)
}

#[allow(non_snake_case)]
fn calculate_jacobian(Iw: &image::ImageBuffer<image::Luma<u8>, Vec<u8>>, X: &Vec<(u32, u32)>, P: &Params) -> Jacobian {
    let mut G = Jacobian::zeros(K, N);

    for (k, (x0, x1)) in X.iter().enumerate() {
        let Y = warp_coords(P, (*x0, *x1));
        let dI_per_dy1 = get_pixel_value(Iw, Y.0 + 1, Y.1) - get_pixel_value(Iw, Y.0, Y.1);
        let dI_per_dy2 = get_pixel_value(Iw, Y.0, Y.1 + 1) - get_pixel_value(Iw, Y.0, Y.1);
/*
        p0  p1  p2      x0      p0x0+p1x1+p2
        p3  p4  p5  *   x1  =   p3x0+p4x1+p5
        p6  p7  p8      1       p6x0+p7x1+p8

        phi1 = (p0x0 + p1x1 + p2) / (p6x0 + p7x1 + p8)
        phi2 = (p3x0 + p4x1 + p5) / (p6x0 + p7x1 + p8)
*/

        let numerator1 = P[0] * (*x0 as f32) + P[1] * (*x1 as f32) + P[2];
        let numerator2 = P[3] * (*x0 as f32) + P[4] * (*x1 as f32) + P[5];
        let denominator = P[6] * (*x0 as f32) + P[7] * (*x1 as f32) + P[8];

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
        // N = 8
        {
            let dphi_ofx1_per_dp8 = - numerator1 * 1.0 / (denominator * denominator);
            let dphi_ofx2_per_dp8 = - numerator2 * 1.0 / (denominator * denominator);
            let el = dI_per_dy1 * dphi_ofx1_per_dp8 + dI_per_dy2 * dphi_ofx2_per_dp8;
            G[(k, 8)] = el;
        }
    } // for

    // zero mean G over columns
    let GColMean = G.row_mean(); // row_mean returns col_mean actually

    // zero mean G over columns
    for i in 0..K {
        for j in 0..N {
            G[(i, j)] -= GColMean[(0, j)];
        }
    }

    G
}


#[allow(non_snake_case)]
fn increment1(ir: &FeatureVector, iw: &FeatureVector, GT_G_inv_GT: &DMatrix::<f32>, ir_iw: f32, ir_pg_iw: f32, iw_pg_iw: f32) -> Params {
    GT_G_inv_GT * (((iw.norm() * iw.norm() - iw_pg_iw) / (0.00000000001 + ir_iw - ir_pg_iw))* ir - iw)
}

#[allow(non_snake_case)]
fn increment2(ir: &FeatureVector, iw: &FeatureVector, GT_G_inv_GT: &DMatrix::<f32>, ir_iw: f32, iw_pg_iw: f32, ir_pg_ir: f32, ir_pg_iw: f32) -> Params {
    let ir_pg_ir = 0.000000001 + ir_pg_ir;
    let lambda1 = (iw_pg_iw / ir_pg_ir).sqrt();
    let lambda2 = ir_pg_iw - ir_iw / ir_pg_ir;

    let lambda = lambda1.max(lambda2);
    GT_G_inv_GT * (lambda * ir - iw)
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
 */
