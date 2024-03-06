// https://docs.rs/arrsac/latest/arrsac/struct.Arrsac.html

use std::ops::Mul;

use imageproc::{geometric_transformations::{Projection, warp_into, Interpolation, warp}, filter::filter3x3, drawing::Canvas};
use image::{self, GenericImageView};
use nalgebra::{SMatrix, DVector, SVector, DMatrix, Matrix, Const, Dynamic, VecStorage, zero};
use rand::Rng;

use crate::pyr::{create_pyramid, ImgBufferF, ImgBufferU8, convert_luma_u8_to_luma_f32, convert_luma_f32_to_luma_u8};

const N: usize = 8;
type Jacobian = DMatrix::<f32>;//Matrix::<f32, nalgebra::Const<K>, nalgebra::Const<N>, nalgebra::RawStorage<f32, nalgebra::Const<K>, nalgebra::Const<N>>>;
type FeatureVector = DVector::<f32>;//SVector::<f32, K>;
type CMatrix = nalgebra::Matrix<f32, Dynamic, Const<1>, VecStorage<f32, Dynamic, Const<1>>>;

#[derive(Clone)]
pub enum ParamsType {
    Trans,
    Trz,
    Proj,
}

#[derive(Clone)]
pub struct Params {
    params: DVector::<f32>,
    ptype: ParamsType,
}

fn invert(input: &DVector::<f32>) -> DVector::<f32> {
    let mut params = DVector::<f32>::from_column_slice(&vec![  1.0, 0.0, 0.0,
                                                                                                        0.0, 1.0, 0.0,
                                                                                                        0.0, 0.0, 1.0]);

    let d4875 = input[4] * input[8] - input[7] * input[5];
    let d3865 = input[3] * input[8] - input[6] * input[5];
    let d3764 = input[3] * input[7] - input[6] * input[4];

    let D = input[0] * d4875 - input[1] * d3865 + input[2] * d3764;

    params[0] =   d4875/D;
    params[1] = - (input[1] * input[8] - input[7] * input[2]) / D;
    params[2] =   (input[1] * input[5] - input[4] * input[2]) / D;
    params[3] = - d3865/D;
    params[4] =   (input[0] * input[8] - input[6] * input[2]) / D;
    params[5] = - (input[0] * input[5] - input[3] * input[2]) / D;
    params[6] =   d3764/D;
    params[7] = - (input[0] * input[7] - input[6] * input[1]) / D;
    params[8] =   (input[0] * input[4] - input[3] * input[1]) / D;

    params
}

#[test]
fn test_invert() {
    let params = DVector::<f32>::from_column_slice(&vec![  1.0, 0.0, 2.0,
                                                                                                        0.0, 1.0, 8.0,
                                                                                                        0.0, 0.0, 1.0]);
    let inv_params = invert(&params);

    assert_eq!(1.0, inv_params[0]);
    assert_eq!(0.0, inv_params[1]);
    assert_eq!(-2.0, inv_params[2]);

    assert_eq!(0.0, inv_params[3]);
    assert_eq!(1.0, inv_params[4]);
    assert_eq!(-8.0, inv_params[5]);

    assert_eq!(0.0, inv_params[6]);
    assert_eq!(0.0, inv_params[7]);
    assert_eq!(1.0, inv_params[8]);
}

impl Params {
    pub fn new(ptype: ParamsType) -> Params {
        Params {
            params: DVector::<f32>::from_column_slice(&vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            ptype
        }
    }

    pub fn get_inverse(&self) -> Params {
        let params = invert(&self.params);

        Params {
            params,
            ptype : self.ptype.clone()
        }
    }

    pub fn get_projection_matrix(&self) -> Option<Projection> {
        Projection::from_matrix([self.params[0], self.params[1], self.params[2], self.params[3], self.params[4], self.params[5], self.params[6], self.params[7], self.params[8]])
    }

    pub fn update(&mut self, inc: &DVector::<f32>, dump_factor: f32) {
        match self.ptype {
            ParamsType::Trans => {
                self.params[2] += inc[0] * dump_factor;
                self.params[5] += inc[1] * dump_factor;
            },
            ParamsType::Trz => {
                for i in 0..6 {
                    self.params[i] += inc[i] * dump_factor;
                }
            }
            ParamsType::Proj => {
                for i in 0..8 {
                    self.params[i] += inc[i] * dump_factor;
                }
            }
        }

    }

    pub fn double_translation_params(&mut self) {
        self.params[2] *= 2.0;
        self.params[5] *= 2.0;
    }

    fn print_params(&self) {
        println!("{} {} {} | {} {} {} | {} {}", self.params[0], self.params[1], self.params[2], self.params[3], self.params[4], self.params[5], self.params[6], self.params[7]);
    }
}

fn get_feature_vector_warped(im: &ImgBufferF, P: &Params, coords: &Vec<(u32, u32)>, normalize: bool) -> FeatureVector {
    let mut fv = FeatureVector::zeros(coords.len());

    for (i, (x, y)) in coords.iter().enumerate() {
        let (x, y) = warp_coords(P, (*x, *y));
        let pixel = get_pixel_value(im, x.round() as i32, y.round() as i32);
        //println!("get_feature_vector_warped {}, {}: {}", x, y, pixel);
        fv[i] = pixel as f32;
    }

    let norm = coords.len() as f32;

    fv.add_scalar_mut(-fv.mean());

    if normalize {
        for el in fv.iter_mut() {
            *el = *el / norm;

            assert!(*el == *el); // check for NaNs
        }
    }

    fv
}

fn get_feature_vector(im: &ImgBufferF, coords: &Vec<(u32, u32)>, normalize: bool) -> FeatureVector {
    let mut fv = FeatureVector::zeros(coords.len());

    for (i, (x, y)) in coords.iter().enumerate() {
        let pixel = get_pixel_value(im, *x as i32, *y as i32);
        //println!("get_feature_vector {}, {}: {}", x, y, pixel);

        fv[i] = pixel as f32;
    }

    let norm = coords.len() as f32;
    fv.add_scalar_mut(-fv.mean());

    if normalize {
        for el in fv.iter_mut() {
            *el = *el / norm;

            assert!(*el == *el); // check for NaNs
        }
    }

    fv
}

fn get_max_gradients(img: &ImgBufferF, num_points: u32) -> Vec<(u32, u32)> {
    use std::collections::BinaryHeap;

    let mut heap = BinaryHeap::new();
    let mut res = vec![];

    println!("img.height() {}", img.height());

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

    for _k in 0..num_points {
        let (_, y, x) = heap.pop().unwrap();

        println!("{} {}", y, x);
        res.push((y, x));
    }

    res
}

/*
#[allow(non_snake_case)]
fn ecc_pyr(Ir: &ImgBufferU8, Iw: &ImgBufferU8, params_type: ParamsType) -> Option<Projection> {
    let mut w = std::fs::OpenOptions::new()
                                        .create(true)
                                        .truncate(true)
                                        .write(true)
                                        .open("/tmp/ecc.log").unwrap();

    let pyr_r = create_pyramid(Ir);
    let pyr_w = create_pyramid(Iw);
    let mut initial_params = Params::new(params_type);
    let mut ecc_max = 0.0;
    let mut num_points = 15;
    let mut points: Option<Vec<(u32, u32)>> = None;

    for (level, (layer_r, layer_w))
                                                            in pyr_r.iter().rev().zip(pyr_w.iter().rev()).enumerate() {
        println!("########################Level {}#########################", level);
        initial_params.double_translation_params();

        let X = get_max_gradients(layer_r, num_points);
        num_points = 3 * num_points / 2;

        let res1 = ecc(layer_r, layer_w, &initial_params, &Some(X), level);

        if let Some(points) = points.as_mut() {
            for (x, y) in points.iter_mut() {
                *x *= 2;
                *y *= 2;
            }
        }
        let res2 = ecc(layer_r, layer_w, &initial_params, &points, level);

        match (res1, res2) {
            (Some(params1), Some(params2)) => {
                if ecc1 >= ecc2 {
                    initial_params = params1.clone();
                    points = Some(X1);

                    if ecc1 > ecc_max {
                        ecc_max = ecc1;
                    }

                } else {
                    initial_params = params2.clone();
                    points = Some(X2);

                    if ecc2 > ecc_max {
                        ecc_max = ecc2;
                    }
                }
            },
            (Some((params1, X1, ecc1)), None) => {
                if ecc1 > ecc_max {
                    ecc_max = ecc1;
                }

                initial_params = params1.clone();
                points = Some(X1);

            },
            (None, Some((params2, X2, ecc2))) => {
                if ecc2 > ecc_max {
                    ecc_max = ecc2;
                }

                initial_params = params2.clone();
                points = Some(X2);
            },
            (None, None) => {
                return None;
            }
        }
    }

    initial_params.get_projection_matrix()
}
*/

fn get_GT_G_inv_GT(G:  &DMatrix::<f32>) ->  DMatrix::<f32> {
    //dump_matrix(&G, "G");
    let GT = G.transpose(); // dims: N x K = 8 x 1000
    //dump_matrix(&GT, "GT");
    let GT_G = GT.mul(G); // dims: N x N = 8 x 8, Hessain . Does JacobianT * Jacobian always give Hessain ? yes
    //dump_matrix(&GT_G, "GT_G = H");
    let GT_G_inv_GT = (GT_G).pseudo_inverse(0.0000000000001).unwrap() * G.transpose(); // dims: N x K = 8 x 1000
    GT_G_inv_GT
}

fn get_PG(G: &DMatrix::<f32>, GT_G_inv_GT:  &DMatrix::<f32>) ->  DMatrix::<f32> {
    // is PG projection ? projection == idempotent : once projected, always projected
    // PG * PG = (G * (GT * G) ^ {-1} * GT) * (G * (GT * G) ^ {-1} * GT)
    // PG * PG = G * (GT * G) ^ {-1} * (GT * G) * (GT * G) ^ {-1} * GT
    // PG * PG = G * (GT * G) ^ {-1} * GT = PG => PG is a projection

    let PG = G.clone() * GT_G_inv_GT.clone(); // dims: K x K = 1000 x 1000 // PG = G * H-1 * GT
    PG
}

fn ecc_increment(Ir: &ImgBufferF, Iw: &ImgBufferF, X: &Vec<(u32, u32)>, P: &mut Params, normalize_feature_vector: bool) -> (f32, FeatureVector) {
    // warp into
    /*
    let mut Imgw = ImgBufferF::new(Iw.width(), Iw.height());
    let h3 = P.get_projection_matrix().expect("Unable to form projection matrix").invert(); // use inverse matrix here
    warp_into(&Iw, &h3, Interpolation::Bilinear, [0.0].into(), &mut Imgw);
    */

    //dump_ecc_intermediate_results(Imgw.clone(), level, num_iter);
    let G = calculate_jacobian(&Iw, &X, &P); // dims: K x N = 1000 x 8
    let GT_G_inv_GT = get_GT_G_inv_GT(&G);
    let PG = get_PG(&G, &GT_G_inv_GT);


    //dump_matrix(&GT_G_inv_GT);


    //dump_image(&Ir, "Ir");
    //dump_image(&Iw, "Iw");

    // cosine similarity between referent (ir) and warped image (iw)
    let ir = get_feature_vector(Ir, &X, normalize_feature_vector);
    let iw = get_feature_vector_warped(&Iw, &P, &X, normalize_feature_vector);

    //dump_vector(&ir, "ir");
    //dump_vector(&iw, "iw");

    let diff = ir.clone() - iw.clone();
    //dump_vector(&diff, "diff");

    let ir_iw = (ir.transpose() * iw.clone())[(0, 0)];
    let ir_pg_iw = (ir.transpose() * PG.clone() * iw.clone())[(0, 0)];
    let iw_pg_iw = (iw.transpose() * PG.clone() * iw.clone())[(0, 0)];

    println!("ir mean {}, iw mean {}", ir.mean(), iw.mean());

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

    (ecc_coeff_approximaiton, inc)
}

//  increment1 and increment2 are simmilar: coeff * ir - iw
// imerror = coeff * ir - iw
//  to 1xN matrix: GT_G_inv_GT * increment
// only this is implemented in matlab code
#[allow(non_snake_case)]
fn increment1(ir: &FeatureVector, iw: &FeatureVector, GT_G_inv_GT: &DMatrix::<f32>, ir_iw: f32, ir_pg_iw: f32, iw_pg_iw: f32) -> CMatrix {
    GT_G_inv_GT * (((iw.norm() * iw.norm() - iw_pg_iw) / (ir_iw - ir_pg_iw))* ir - iw)
}

#[allow(non_snake_case)]
fn increment2(ir: &FeatureVector, iw: &FeatureVector, GT_G_inv_GT: &DMatrix::<f32>, ir_iw: f32, iw_pg_iw: f32, ir_pg_ir: f32, ir_pg_iw: f32) -> CMatrix {
    let ir_pg_ir = ir_pg_ir + 0.0000001;
    let lambda1 = (iw_pg_iw / ir_pg_ir).sqrt();
    let lambda2 = (ir_pg_iw - ir_iw) / ir_pg_ir;

    let lambda = lambda1.max(lambda2);
    GT_G_inv_GT * (lambda * ir - iw)
}

#[derive(Clone)]
pub struct ECCPyr {
    src_pyramid: Vec<ImgBufferF>,
    dst_pyramid: Vec<ImgBufferF>,
    level: usize,
    params: Option<Params>,
    num_points: u32,
    ecc_impl: Option<ECCImpl>,
}

impl ECCPyr
{
    pub fn new(src: ImgBufferU8, dst: ImgBufferU8) -> Self {
        let src_pyramid = create_pyramid(&src);
        let dst_pyramid = create_pyramid(&dst);
        let params = Some(Params::new(ParamsType::Trz));

        Self {
            src_pyramid,
            dst_pyramid,
            level: 0,
            params,
            num_points: 15,
            ecc_impl: None,
        }
    }

    pub fn tick(&mut self) -> Option<Params> {
        assert!(self.src_pyramid.len() == self.dst_pyramid.len());

        let done = (self.level >= self.src_pyramid.len() && self.ecc_impl.is_some() && self.ecc_impl.as_ref().unwrap().is_done()) || self.params.is_none();

        if done {
            return self.params.clone();
        }

        if self.ecc_impl.is_none() || self.ecc_impl.as_ref().unwrap().is_done() {
            self.params.as_mut().unwrap().double_translation_params();
            let ind = self.src_pyramid.len() - 1 - self.level;
            let x = get_max_gradients(&self.src_pyramid[ind], self.num_points);
            self.ecc_impl = Some(ECCImpl::new(&self.src_pyramid[ind], &self.dst_pyramid[ind], x, self.params.as_ref().unwrap()));
            self.level += 1;
        }

        self.params = self.ecc_impl.as_mut().unwrap().tick();
        self.params.clone()
    }

    fn get_progress() {
    }
}

#[derive(Clone)]
struct ECCImpl {
    x: Vec<(u32, u32)>,
    ecc_coeff_max: f32,
    params: Params,
    params_best: Option<Params>,
    num_iter: u32,
    done: bool,
}

impl ECCImpl {
    fn new(x: Vec<(u32, u32)>, params: &Params) -> ECCImpl {
        ECCImpl {
            x,
            ecc_coeff_max: -1000.0,
            params: params.clone(),
            params_best: None,
            num_iter: 0,
            done: false,
        }
    }

    fn is_done(&self) -> bool {
        self.done
    }

    fn tick(&mut self, ir: &ImgBufferF, iw: &ImgBufferF,) -> Option<Params> {
        if self.done {
            return self.params_best.clone();
        }

        let normalize_feature_vector = false; // don't normalize feature vector. TODO: remove this parameter
        let (ecc_coeff_approximation, inc) = ecc_increment(&ir, &iw, &self.x, &mut self.params, normalize_feature_vector);

        let mut last_is_largest = false;

        if ecc_coeff_approximation < self.ecc_coeff_max {
            last_is_largest = false;
            println!("no improvement");

        } else {
            self.ecc_coeff_max = ecc_coeff_approximation;
            last_is_largest = true;
            self.params.update(&inc, 1.0);
            self.params_best = Some(self.params.clone());
        }

        let should_continue = if last_is_largest {
            self.num_iter < 15
        } else {
            self.num_iter < 5
        };

        let threshold = 0.00001;
        self.done = should_continue && inc.norm() < threshold;

        self.num_iter += 1;
        self.params_best.clone()
    }
}

#[allow(non_snake_case)]
fn ecc(Ir: &ImgBufferF, Iw: &ImgBufferF, initial_params: &Params, X: &Option<Vec<(u32, u32)>>, level: usize) -> Option<Params> {
    if X.is_none() {
        return None;
    }

    let X = X.as_ref().unwrap();

    let mut w: std::fs::File = std::fs::OpenOptions::new().append(true).open("/tmp/ecc.log").unwrap();
    use std::io::Write;

    let mut ecc_impl = ECCImpl::new(Ir, Iw, X.clone(), initial_params);

    loop {
        ecc_impl.tick();

        if ecc_impl.is_done() {
            break;
        }
    }

    // writeln!(&mut w, "{}", ecc_coeff_max).unwrap();
    // writeln!(&mut w, "{}", 0).unwrap();

    ecc_impl.tick()
}

fn calculate_jacobian(Iw: &ImgBufferF, X: &Vec<(u32, u32)>, P: &Params) -> Jacobian {
    match P.ptype {
        ParamsType::Trans => {
            calculate_jacobian_trans(Iw, X, P)
        },
        ParamsType::Trz => {
            calculate_jacobian_trz(Iw, X, P)
        }
        ParamsType::Proj => {
            calculate_jacobian_proj(Iw, X, P)
        }
    }
}

#[allow(non_snake_case)]
fn calculate_jacobian_trans(Iw: &ImgBufferF, X: &Vec<(u32, u32)>, P: &Params) -> Jacobian {
    let mut G = Jacobian::zeros(X.len(), 2);

    for (k, (x0, x1)) in X.iter().enumerate() {
        let Y = warp_coords(&P, (*x0, *x1));
        let dI_per_dy1 = get_warped_grad_x(Iw, Y.0, Y.1);
        let dI_per_dy2 = get_warped_grad_y(Iw, Y.0, Y.1);

/*
        1   0  p2      x0      x0+p2
        0   1  p5  *   x1  =   x1+p5
        0   0   1       1        1

        phi1 = x0 + p2
        phi2 = x1 + p5
*/

        // N = 0
        {
            let dphi_ofx1_per_dp0 = 1.0;
            let dphi_ofx2_per_dp0 = 0.0;
            let el: f32 = dI_per_dy1 * dphi_ofx1_per_dp0 + dI_per_dy2 * dphi_ofx2_per_dp0;
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

    //zero_mean_over_columns(&mut G);

    G
}

#[allow(non_snake_case)]
fn calculate_jacobian_trz(Iw: &ImgBufferF, X: &Vec<(u32, u32)>, P: &Params) -> Jacobian {
    let mut G = Jacobian::zeros(X.len(), 8);

    for (k, (x0, x1)) in X.iter().enumerate() {
        let Y = warp_coords(&P, (*x0, *x1));
        let dI_per_dy1 = get_warped_grad_x(Iw, Y.0, Y.1);
        let dI_per_dy2 = get_warped_grad_y(Iw, Y.0, Y.1);

/*
        p0  p1  p2      x0      p0x0+p1x1+p2
        p3  p4  p5  *   x1  =   p3x0+p4x1+p5
        0   0   1       1           1

        phi1 = (p0x0 + p1x1 + p2)
        phi2 = (p3x0 + p4x1 + p5)
*/

        let numerator1 = P.params[0] * (*x0 as f32) + P.params[1] * (*x1 as f32) + P.params[2];
        let numerator2 = P.params[3] * (*x0 as f32) + P.params[4] * (*x1 as f32) + P.params[5];

        // N = 0
        {
            let dphi_ofx1_per_dp0 = *x0 as f32;
            let dphi_ofx2_per_dp0 = 0.0;
            let el = dI_per_dy1 * dphi_ofx1_per_dp0 + dI_per_dy2 * dphi_ofx2_per_dp0;
            G[(k, 0)] = el;
        }
        // N = 1
        {
            let dphi_ofx1_per_dp1 = *x1 as f32;
            let dphi_ofx2_per_dp1 = 0.0;
            let el = dI_per_dy1 * dphi_ofx1_per_dp1 + dI_per_dy2 * dphi_ofx2_per_dp1;
            G[(k, 1)] = el;
        }
        // N = 2
        {
            let dphi_ofx1_per_dp2 = 1.0;
            let dphi_ofx2_per_dp2 = 0.0;
            let el = dI_per_dy1 * dphi_ofx1_per_dp2 + dI_per_dy2 * dphi_ofx2_per_dp2;
            G[(k, 2)] = el;
        }
        // N = 3
        {
            let dphi_ofx1_per_dp3 = 0.0;
            let dphi_ofx2_per_dp3 = *x0 as f32;
            let el = dI_per_dy1 * dphi_ofx1_per_dp3 + dI_per_dy2 * dphi_ofx2_per_dp3;
            G[(k, 3)] = el;
        }
        // N = 4
        {
            let dphi_ofx1_per_dp4 = 0.0;
            let dphi_ofx2_per_dp4 = *x1 as f32;
            let el = dI_per_dy1 * dphi_ofx1_per_dp4 + dI_per_dy2 * dphi_ofx2_per_dp4;
            G[(k, 4)] = el;
        }
        // N = 5
        {
            let dphi_ofx1_per_dp5 = 0.0;
            let dphi_ofx2_per_dp5 = 1.0;
            let el = dI_per_dy1 * dphi_ofx1_per_dp5 + dI_per_dy2 * dphi_ofx2_per_dp5;
            G[(k, 5)] = el;
        }
    } // for

    zero_mean_over_columns(&mut G);

    G
}

#[allow(non_snake_case)]
fn calculate_jacobian_proj(Iw: &ImgBufferF, X: &Vec<(u32, u32)>, P: &Params) -> Jacobian {
    let mut G = Jacobian::zeros(X.len(), 8);

    for (k, (x0, x1)) in X.iter().enumerate() {
        let Y = warp_coords(&P, (*x0, *x1));
        let dI_per_dy1 = get_warped_grad_x(Iw, Y.0, Y.1);
        let dI_per_dy2 = get_warped_grad_y(Iw, Y.0, Y.1);
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


fn get_indices(Ir: &ImgBufferF, Iw: &ImgBufferF, P_inv: &Params, num_indices_to_extract: i32) -> Vec<(u32, u32)> {

    let mut res = vec![];

    for y in 0..Iw.height() {
        for x in 0..Iw.width() {
            res.push((x, y));
        }
    }

    return res;

    use std::collections::BinaryHeap;

    let mut x_heap = BinaryHeap::new();
    let mut y_heap = BinaryHeap::new();


    println!("img.height() {}", Iw.height());

    let margin = 0;
    // get max gradient_x and max gradient_y of Iw
    for y in margin..Iw.height() - margin - 1 {
        for x in margin..Iw.width() - margin - 1 {
            let gx = get_grad_x_at(Iw, x, y);
            let gy = get_grad_y_at(Iw, x, y);

            x_heap.push(((gx as i32).abs(), x, y));
            y_heap.push(((gy as i32).abs(), x, y));
        }
    }

    // filter out ```num_indices_to_extract``` valid points in both images
    while res.len() < num_indices_to_extract as usize {
        if x_heap.is_empty() && y_heap.is_empty() {
            break;
        }

        if let Some((_, x, y)) = x_heap.pop() {
            let (x, y) = warp_coords(P_inv, (x, y));
            let x = x.round() as i32;
            let y = y.round() as i32;

            if (x >= 0 && x < Ir.width() as i32 && y >= 0 && y < Ir.height() as i32) {
                res.push((x as u32, y as u32));
            }
        }

        if let Some((_, x, y)) = y_heap.pop() {
            let (x, y) = warp_coords(P_inv, (x, y));
            let x = x.round() as i32;
            let y = y.round() as i32;

            if (x >= 0 && x < Ir.width() as i32 && y >= 0 && y < Ir.height() as i32) {
                res.push((x as u32, y as u32));
            }
        }
    }

    res.sort_by(|(ix1, iy1), (ix2, iy2)|{
        ix1.cmp(ix2).then(iy1.cmp(iy2))
    });

    /*
    let res : Vec<(u32, u32)> = res.windows(2).filter_map(|pair| {
        if pair[0] != pair[1] {
            Some(pair[0])
        } else {
            None
        }
    }).collect();
    */

    // remove duplicates ?

    println!("indices: {:?}", res);

    res
}

#[test]
fn test_sort_filter_map(){
    let mut res2 = vec![1, 8, 10, -3, 8, 9, 3, 6, -3, 4, 10, 3];

    res2.sort_by(|el1, el2|{
        el1.cmp(el2)
    });

    println!("{:?}", res2);

    let res2 : Vec<i32> = res2.windows(2).filter_map(|pair| {
        println!("pair: ({}, {})", pair[0], pair[1]);

        if pair[0] != pair[1] {
            Some(pair[0])
        } else {
            None
        }
    }).collect();

    println!("{:?}", res2);
}
/*
todos:
* Projection - unable to form
* feature vector - zero mean ? normalized ?
*/


fn get_grad_x_at(img: &ImgBufferF, x: u32, y: u32) -> f32 {
    /*
    let gx = -0.5 * img.get_pixel(x - 1, y).0[0] as f32
                    + img.get_pixel(x, y).0[0]  as f32
                    - 0.5 * img.get_pixel(x, y + 1 ).0[0] as f32;
                    */
    let x = x as i32;
    let y = y as i32;
    let v1 = get_pixel_value(img, x + 1, y);
    let v2 = get_pixel_value(img, x - 1, y);

    let gx = (v1 - v2) / 2.0;
    //println!("grad_x {}, x: {}, y: {}, v1: {}, v2: {}", gx, x, y, v1, v2);

    gx
}

fn get_grad_y_at(img: &ImgBufferF, x: u32, y: u32) -> f32 {
    /*
    let gy = -0.5 * img.get_pixel(x, y - 1).0[0] as f32
                    + img.get_pixel(x, y).0[0] as f32
                    - 0.5 * img.get_pixel(x, y + 1).0[0] as f32;
                    */
    let x = x as i32;
    let y = y as i32;
    let v1 = get_pixel_value(img, x, y + 1);
    let v2 = get_pixel_value(img, x, y - 1);
    let gy = (v1 - v2) / 2.0;

    //println!("gy: {}, x: {}, y: {}, v1: {}, v2: {}", gy, x, y, v1, v2);

    gy
}

fn get_warped_grad_x(img: &ImgBufferF, x: f32, y: f32) -> f32 {
    let ix = x.round() as u32;
    let iy = y.round() as u32;

    let mut dx = x - (ix as f32);
    let mut dy = y - (iy as f32);

    if x >= (img.width() - 1) as f32 {
        dx = 1.0;
    }

    if y >= (img.height() - 1) as f32 {
        dy = 1.0;
    }

    let pix00 = get_grad_x_at(img, ix, iy);
    let pix01 = get_grad_x_at(img, ix, iy + 1);
    let pix02 = get_grad_x_at(img, ix + 1, iy);
    let pix03 = get_grad_x_at(img, ix + 1, iy + 1);
    //println!("warped_grad_x {} {} {} {}", pix00, pix01, pix02, pix03);

    let res = (1.0 - dx) * (1.0 - dy) * pix00
                + (1.0 - dx) * dy * pix01
                +  dx * (1.0 - dy) * pix02
                + dx * dy * pix03;

    res
}

fn get_warped_grad_y(img: &ImgBufferF, x: f32, y: f32) -> f32 {
    let ix = x.round() as u32;
    let iy = y.round() as u32;

    let mut dx = x - (ix as f32);
    let mut dy = y - (iy as f32);

    if x >= (img.width() - 1) as f32 {
        dx = 1.0;
    }

    if y >= (img.height() - 1) as f32 {
        dy = 1.0;
    }

    let pix00 = get_grad_y_at(img, ix, iy);
    let pix01 = get_grad_y_at(img, ix, iy + 1);
    let pix02 = get_grad_y_at(img, ix + 1, iy);
    let pix03 = get_grad_y_at(img, ix + 1, iy + 1);

    //println!("warped_grad_y {} {} {} {}", pix00, pix01, pix02, pix03);

    let res = (1.0 - dx) * (1.0 - dy) * pix00
                + (1.0 - dx) * dy * pix01
                +  dx * (1.0 - dy) * pix02
                +  dx * dy * pix03;

    res
}

fn zero_mean_over_columns(G: &mut DMatrix::<f32>) {
    // let GColMean = G.row_mean(); // nalgebra implementation is strage, row_mean returns col_mean actually

    let mut GColMean = DMatrix::<f32>::zeros(1, G.ncols());

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

#[test]
fn test_zero_mean_over_columns() {
    let rows = 10;
    let cols = 12;
    let mut matrix = DMatrix::<f32>::zeros(rows, cols);

    let mut value = 1.0;

    for y in 0..rows {
        for x in 0..cols {
            matrix[(y, x)] = value;
            value += 1.0;
        }
    }

    assert_eq!(rows, matrix.nrows());
    assert_eq!(cols, matrix.ncols());

    println!("Matrix {}", matrix);

    zero_mean_over_columns(&mut matrix);

    println!("zero mean Matrix {}", matrix);
}

fn warp_coords(P: &Params, X: (u32, u32)) -> (f32, f32) {
    let y1 =    P.params[0] * (X.0 as f32)      + P.params[1] * (X.1 as f32)    + P.params[2];
    let y2 =    P.params[3] * (X.0 as f32)      + P.params[4] * (X.1 as f32)    + P.params[5];
    let den =   P.params[6] * (X.0 as f32)      + P.params[7] * (X.1 as f32)    + 1.0;

    (y1 / den, y2 / den)
}

fn get_pixel_value(im: &ImgBufferF, x: i32, y: i32) -> f32 {
    let x : i32 = std::cmp::max(0, std::cmp::min(im.width() as i32 - 1, x));
    let y : i32 = std::cmp::max(0, std::cmp::min(im.height() as i32 - 1, y));

    im.get_pixel_checked(x as u32, y as u32)
        .and_then(|e| Some(e.0[0] as f32))
        .unwrap_or(0.0)
}

fn draw_max_gradients(img: &mut ImgBufferF) {
    let max_gradients = get_max_gradients(img, 100);

    for y in 0..img.height() {
        for x in 0..img.width() {
            if let Some(pix) = img.get_pixel_mut_checked(x, y) {
                pix.0[0] = 0.0;
            }
        }
    }

    for (yi, xi) in max_gradients {
        if let Some(pix) = img.get_pixel_mut_checked(xi, yi) {
            pix.0[0] = 255.0;
        }
    }
}

fn draw_points(img: &mut ImgBufferF, X: &Vec<(u32, u32)>, P: &Params) {
    for (x0, x1) in X.iter() {
//        println!("{} {}", x0, x1);
        let (y0, y1) = warp_coords(P, (*x0, *x1));

        if let Some(pix) = img.get_pixel_mut_checked(y0.round() as u32, y1.round() as u32) {
            pix.0[0] = 255.0;
        }
    }
}

fn dump_ecc_intermediate_results(Imgw: ImgBufferF, level: usize, num_iter: usize) {
    let mut imgw_clone = Imgw.clone();
    draw_max_gradients(&mut imgw_clone);

    let imgw_clone = convert_luma_f32_to_luma_u8(&imgw_clone);
    //draw_points(&mut Imgw_clone, &X, &P);
    let _ = imgw_clone.save(format!("img/level_{}_iter_{}.png", level, num_iter));
}

fn dump_image(Im: &ImgBufferF, comment: &str) {
    println!("P2, {}", comment);
    println!("{}", Im.width());
    println!("{}", Im.height());
    println!("65536");

    for y in 0..Im.height() {
        for x in 0..Im.width() {
            print!("{} ", get_pixel_value(&Im, x as i32, y as i32));
        }

        println!();
    }
}

fn dump_matrix(M: &DMatrix::<f32>, comment: &str) {
    println!("P2, {}", comment);
    println!("{}", M.ncols());
    println!("{}", M.nrows());
    println!("65536");

    for y in 0..M.nrows() {
        for x in 0..M.ncols() {
            print!("{} ", M[(y, x)]);
        }

        println!();
    }
}

fn dump_vector(M: &DVector::<f32>, comment: &str) {
    println!("P2 {}", comment);

    for y in 0..M.len() {
        print!("{} ", M[y] as i32);
    }

    println!();
}

#[test]
fn test_get_feature_vector() {
    let img1 = image::open("img/rect16x16.png").expect("File not found!").grayscale().into_luma8();
    let img2 = image::open("img/rect16x16-trans-x.png").expect("File not found!").grayscale().into_luma8();

    let img1f = convert_luma_u8_to_luma_f32(&img1);
    let img2f = convert_luma_u8_to_luma_f32(&img2);

    let normalize_feature_vector = false;

    let p = Params::new(ParamsType::Trans);
    let X = get_indices(&img1f, &img2f, &p.get_inverse(), 2 * 16 * 16); // 2 - x and y
    let ir = get_feature_vector(&img1f, &X, normalize_feature_vector);
    let iw = get_feature_vector_warped(&img2f, &p, &X, normalize_feature_vector);

    dump_vector(&ir, "ir");
    dump_vector(&iw, "iw");

    let diff = ir - iw;
    dump_vector(&diff, "diff");
}

#[test]
fn test_get_pixel_value() {
    let img1 = image::open("img/rect4x4-test.png").expect("File not found!").grayscale().into_luma8();
    let img1f = convert_luma_u8_to_luma_f32(&img1);

    assert_eq!(0.0, get_pixel_value(&img1f, 2, 0));
    assert_eq!(255.0, get_pixel_value(&img1f, 0, 2));
}

#[test]
fn test_g_matrix() {
    let img2 = image::open("img/rect16x16-trans-x.png").expect("File not found!").grayscale().into_luma8();
    let img2f = convert_luma_u8_to_luma_f32(&img2);

    let mut p = Params::new(ParamsType::Trans);

    if false
    {
        let (x, y) =  (0, 15);
        let (x, y) = warp_coords(&p, (x, y));
        assert_eq!(0.0, x);
        assert_eq!(15.0, y);

        let grad_x = get_warped_grad_x(&img2f, x, y);
        assert_eq!(0.0, grad_x);

        let grad_y = get_warped_grad_y(&img2f, x, y);
        assert_eq!(0.0, grad_y);
    }

    if false
    {
        let (x, y) =  (5, 2);
        let (x, y) = warp_coords(&p, (x, y));
        assert_eq!(5.0, x);
        assert_eq!(2.0, y);

        let grad_x = get_warped_grad_x(&img2f, x, y);
        assert_eq!(0.0, grad_x);

        let grad_y = get_warped_grad_y(&img2f, x, y);
        assert_eq!(0.0, grad_y);
    }


    {
        let (x, y) =  (4, 3);
        let (x, y) = warp_coords(&p, (x, y));
        assert_eq!(4.0, x);
        assert_eq!(3.0, y);

        let grad_x = get_warped_grad_x(&img2f, x, y);
        assert_eq!(0.0, grad_x);

        let grad_y = get_warped_grad_y(&img2f, x, y);
        assert_eq!(0.0, grad_y);
    }

    let g_exp_values : Vec<f32> = vec![
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        -43.0,    0.0,
        -43.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,  -43.0,
        -43.0,  -43.0,
        -43.0,   43.0,
        0.0,   43.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,  -43.0,
        0.0,  -43.0,
        0.0,   43.0,
        0.0,   43.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,  -43.0,
        0.0,  -43.0,
        0.0,   43.0,
        0.0,   43.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,  -43.0,
        43.0,  -43.0,
        43.0,   43.0,
        0.0,   43.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        43.0,    0.0,
        43.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,   0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,    0.0,
        0.0,   0.0,
        0.0,    0.0,
    ];


    let x = get_indices(&img2f, &img2f, &p, (img2f.width() * img2f.height()) as i32);

    let G = calculate_jacobian(&img2f, &x, &p);
    //let G_exp = DMatrix::from_vec(G.nrows(), G.ncols(), g_exp_values);
    let G_exp = DMatrix::from_row_slice(G.nrows(), G.ncols(), &g_exp_values);

    //dump_matrix(&G, "G");
    //dump_matrix(&G_exp, "G_exp");

    /*
    for y in 0..G.nrows() {
        for x in 0..G.ncols() {
            //println!("{} {} {} {}", y, x, 1 + y / 16, y - 16 * (y / 16));
            //assert_eq!(G_exp[(y, x)], G[(y, x)]);

            if G_exp[(y, x)] != G[(y, x)] {
                println!("diff at {}, {}: {} != {}, indices to test {} {}", y, x, G_exp[(y, x)], G[(y, x)], y / 16, y - 16 * (y / 16));
            }
        }
    }
    */

    //assert_eq!(G_exp, G);
}

#[test]
fn test_ecc_increment() {
    use crate::reg::ecc;

    let paths = std::fs::read_dir("./img/").unwrap();

    for p in paths {
        if p.is_ok() {
            let path = p.unwrap().path();

            if path.to_str().unwrap().starts_with("./img/level") {
                println!("{:?}", path);
                std::fs::remove_file(path);
            } else {
                //debug!("skip {:?}", path);
            }
        }
    }

    let img1 = image::open("img/rect16x16.png").expect("File not found!").grayscale().into_luma8();
    let img2 = image::open("img/rect16x16-trans-x.png").expect("File not found!").grayscale().into_luma8();

    let img1f = convert_luma_u8_to_luma_f32(&img1);
    let img2f = convert_luma_u8_to_luma_f32(&img2);

    let mut p = Params::new(ParamsType::Trans);
    let num_points_per_parameter = 15;
    let X = get_indices(&img1f, &img2f, &p.get_inverse(), 2 * 16 * 16); // 2 - x and y


    let mut w = std::fs::OpenOptions::new()
                                        .create(true)
                                        .truncate(true)
                                        .write(true)
                                        .open("/tmp/ecc.log").unwrap();

    let (ecc_approximation, increment) = ecc_increment(&img1f, &img2f, &X, &mut p, false /* normalization is not done in matlab */);
    //let (params, vec, _) = ecc(&img2f, &img1f, &p, &X, 1)
    //                .expect("Registration failed");


    println!("End result: ");
    println!("ecc_approximation {}, increment {:?}", ecc_approximation, increment);
    //params.print_params();
}

#[test]
fn test_ecc_no_pyr() {
    use crate::reg::ecc;

    let paths = std::fs::read_dir("./img/").unwrap();

    for p in paths {
        if p.is_ok() {
            let path = p.unwrap().path();

            if path.to_str().unwrap().starts_with("./img/level") {
                println!("{:?}", path);
                std::fs::remove_file(path);
            } else {
                //debug!("skip {:?}", path);
            }
        }
    }

    let img1 = image::open("img/rect16x16.png").expect("File not found!").grayscale().into_luma8();
    let img2 = image::open("img/rect16x16-trans-x.png").expect("File not found!").grayscale().into_luma8();

    let img1f = convert_luma_u8_to_luma_f32(&img1);
    let img2f = convert_luma_u8_to_luma_f32(&img2);

    let mut p = Params::new(ParamsType::Trans);
    let num_points_per_parameter = 15;
    let X = get_indices(&img1f, &img2f, &p.get_inverse(), 2 * 16 * 16); // 2 - x and y


    let mut w = std::fs::OpenOptions::new()
                                        .create(true)
                                        .truncate(true)
                                        .write(true)
                                        .open("/tmp/ecc.log").unwrap();


    if let Some((p_res, v, f)) = ecc(&img1f, &img2f, &p, &Some(X), 0) {
        println!("End result: ");
        p_res.print_params();
        //params.print_params();
    } else {
        println!("ecc failed");
    }
}
