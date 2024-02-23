use eframe::egui;
use imageproc::geometric_transformations::{Projection, warp_into, Interpolation};
use egui::ColorImage;
use derivative::Derivative;

use crate::fft::{fft_2D, FFTType, FFTParams};


fn save_image(im: &egui::ColorImage, path: &str) {
    let size = im.size;
    let mut pixels = Vec::with_capacity(size[0]*4*size[1]);

    for pix in im.pixels.iter() {
        pixels.push(pix.r());
        pixels.push(pix.g());
        pixels.push(pix.b());
        pixels.push(pix.a());
    }

    let img: image::ImageBuffer<image::Rgba<u8>, Vec<_>> =
        image::ImageBuffer::from_raw(size[0] as u32, size[1] as u32, pixels)
        .expect("bad conversion");

    img.save(path).expect("Failed to save");
}

fn warp_image(out_w: u32, out_h: u32, im: &egui::ColorImage, h3: &Projection, alpha: u8) -> egui::ColorImage {
    // convert to image::Image
    let size = im.size;
    let mut pixels = Vec::with_capacity(size[0]*4*size[1]);
    for pix in im.pixels.iter() {
        pixels.push(pix.r());
        pixels.push(pix.g());
        pixels.push(pix.b());
        pixels.push(alpha);
    }

    let tmp_img: image::ImageBuffer<image::Rgba<u8>, Vec<_>> =
        image::ImageBuffer::from_raw(size[0] as u32, size[1] as u32, pixels)
        .expect("bad conversion");

    let out_size: [usize; 2] = [out_w as usize, out_h as usize];
    let mut new_img: image::ImageBuffer<image::Rgba<u8>, Vec<_>> =
        image::ImageBuffer::new(out_w, out_h);

    warp_into(&tmp_img, h3, Interpolation::Bilinear, [255, 0, 255, alpha].into(), &mut new_img);

    // convert back to egui::ColorImage
    let pixels = new_img.as_raw()
        .chunks_exact(4)
        .map(|p| {
            let alpha = if p[0] == 255 && p[1] == 0 && p[2] == 255 {
                0
            } else {
                p[3]
            };

            let lr = p[0];
            let lg = p[1];
            let lb = p[2];
            let la = alpha;
            egui::Color32::from_rgba_unmultiplied(lr, lg, lb, la)
        })
    .collect();

    egui::ColorImage{size: out_size, pixels}
}

fn load_image_from_path(path: &std::path::Path) -> Result<egui::ColorImage, image::ImageError> {
    let image = image::io::Reader::open(path)?.decode()?;
    let size = [image.width() as _, image.height() as _];
    let image_buffer = image.to_rgba8();
    let px = image_buffer.as_flat_samples();
    let pixels = px.as_slice()
        .chunks_exact(4)
        .map(|p| {
            let lr = p[0];
            let lg = p[1];
            let lb = p[2];
            egui::Color32::from_rgb(lr, lg, lb)
        })
        .collect();
    let image = egui::ColorImage{size, pixels};
    Ok(image)
}

fn get_lena() -> Result<egui::ColorImage, image::ImageError> {
    let rgba = include_bytes!("../img/lena.bgra").to_vec();

    let pixels = rgba.chunks_exact(4)
        .map(|p| {
             let lb = p[0];
             let lg = p[1];
             let lr = p[2];

             egui::Color32::from_rgb(lr, lg, lb)
         })
        .collect();

    let size = [225, 225];
    let image = egui::ColorImage{ size, pixels };
    Ok(image)
}

fn display_homography(ui: &mut egui::Ui, h3: &Projection) {
    ui.vertical(|ui|{
        egui::Grid::new("some_unique_id")
        .striped(true)
        .show(ui, |ui| {
            let h3_str = format!("{:?}", h3);
            let istart = h3_str.find("[");
            let iend = h3_str.find("]");

            if istart.is_some() && iend.is_some() {
                let coeffs = &h3_str[istart.unwrap() + 1..iend.unwrap()];
                //ui.label(coeffs);
                for (i, coeff) in coeffs.split(',').enumerate() {
                    if i % 3 == 0 && i > 0{
                        ui.end_row();
                    }

                    ui.label(format!("{:.5}", coeff));
                }
            }
        });
    });
}

#[derive(Derivative, Clone)]
#[derivative(PartialEq)]
enum Homography {
    I,
    R {
        angle: f32
    },
    T {
        tx: f32,
        ty: f32,
    },
    S {
        sx: f32,
        sy: f32,
        isotropic: bool,
    },
    P {
        h31: f32,
        h32: f32,
        h33: f32
    } ,
    Points {
        new_points_src: [(f32, f32); 4],
        new_points_dst: [(f32, f32); 4],

        // used for caching, so projection is not calculated for each frame
        prev_points_dst: [(f32, f32); 4],
        prev_points_src: [(f32, f32); 4],
        #[derivative(PartialEq = "ignore")]
        proj: Option<Projection>,
    },
    Reg {
        #[derivative(PartialEq = "ignore")]
        proj: Option<Projection>,
    },
}

fn get_projection(uimx: &UIMatrix) -> Projection {
    if !uimx.on {
        return Projection::scale(1.0, 1.0);
    }

    let h3 = match uimx.h3 {
        Homography::I => { Projection::scale(1.0, 1.0) },
        Homography::R{angle} => Projection::rotate(angle * 2.0 * 3.14 / 360.0 ),
        Homography::T{tx, ty} => Projection::translate(tx, ty),
        Homography::S{sx, sy, isotropic: _ } => Projection::scale(sx, sy),
        Homography::P{h31, h32, h33} => Projection::from_matrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, h31, h32, h33]).expect("non invertible"),
        Homography::Points{prev_points_src : _, prev_points_dst : _, new_points_src : _, new_points_dst : _, proj} => proj.unwrap_or(Projection::scale(1.0, 1.0)),
        Homography::Reg{ proj } => proj.unwrap_or(Projection::scale(1.0, 1.0)),
    };

    if uimx.inverse {
        return h3.invert();
    }

    h3
}

fn display_h3(ui: &mut egui::Ui, uimx: &mut UIMatrix, index: i64) {
    let mut selected_text = "";
    let h3 = &mut uimx.h3;

    ui.vertical(|ui|{
        match h3 {
            Homography::I => {
                selected_text = "I";
                ui.label("Eye");
            },
            Homography::R{angle} => {
                selected_text = "Rot";

                ui.label("Rot");
                ui.add(egui::Slider::new(angle, -360.0..=360.0).text("deg"));
            },
            Homography::S{sx, sy, isotropic} => {
                selected_text = "Scale";

                ui.label("Scale");
                ui.add(egui::Slider::new(sx, 0.00001..=5.0));
                ui.add(egui::Slider::new(sy, 0.00001..=5.0));
                ui.checkbox(isotropic, "isotropic".to_string());

                if *isotropic {
                    *sy = *sx;
                }
            },
            Homography::T{tx, ty}=>{
                selected_text = "Trans";

                ui.label("Trans");
                ui.add(egui::Slider::new(tx, -2000.0..=2000.0));
                ui.add(egui::Slider::new(ty, -2000.0..=2000.0));
            },
            Homography::P { h31, h32, h33 } => {
                selected_text = "Proj";

                ui.label("Proj");
                ui.add(egui::Slider::new(h31, -0.01..=0.01));
                ui.add(egui::Slider::new(h32, -0.01..=0.01));
                ui.add(egui::Slider::new(h33, -5.0..=5.0));
            },
            Homography::Points { prev_points_src, prev_points_dst, new_points_src, new_points_dst, proj } => {
                selected_text = "Points";
                ui.label("Points");

                ui.horizontal(|ui| {
                    ui.label("src x");
                    ui.label("src y");
                    ui.label("dst x");
                    ui.label("dst y");
                });

                for i in 0..=3 {
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut new_points_src[i].0).speed(0.1));
                        ui.add(egui::DragValue::new(&mut new_points_src[i].1).speed(0.1));
                        ui.add(egui::DragValue::new(&mut new_points_dst[i].0).speed(0.1));
                        ui.add(egui::DragValue::new(&mut new_points_dst[i].1).speed(0.1));
                    });
                }

                if new_points_src != prev_points_src || new_points_dst != prev_points_dst {
                    for i in 0..4 {
                        prev_points_src[i] = new_points_src[i];
                        prev_points_dst[i] = new_points_dst[i];
                    }

                    *proj = Projection::from_control_points(new_points_src.clone(), new_points_dst.clone());
                }

                if proj.is_some() {
                    ui.label("Valid");
                } else {
                    ui.label("Invalid");
                }
            },
            Homography::Reg {proj} => {
                selected_text = "Reg";
                ui.label("Reg");
            },
        }

        ui.checkbox(&mut uimx.on, "on/off".to_string());
        ui.checkbox(&mut uimx.inverse, "inverse".to_string());

        // combo - change homography type
        egui::ComboBox::from_id_source(index)
            .width(100.0)
            .selected_text(selected_text)
            .show_ui(ui, |ui|{
                ui.selectable_value(h3, Homography::I, format!("I"));
                ui.selectable_value(h3, Homography::R{angle: 0.0}, format!("Rot"));
                ui.selectable_value(h3, Homography::S{sx: 1.0, sy: 1.0, isotropic: false}, format!("Scale"));
                ui.selectable_value(h3, Homography::T{tx: 0.0, ty: 0.0}, format!("Trans"));
                ui.selectable_value(h3, Homography::P{h31: 0.0, h32: 0.0, h33: 1.0}, format!("Proj"));
                let points: [(f32, f32); 4]  = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)];

                ui.selectable_value(h3, Homography::Points{
                    prev_points_src: points.clone(),
                    prev_points_dst: points.clone(),
                    new_points_src: points.clone(),
                    new_points_dst: points.clone(),
                    proj: Some(Projection::scale(1.0, 1.0)),
                }, format!("Points"));

                ui.selectable_value(h3, Homography::Reg { proj: None }, format!("Reg"));
            });
    });
}

#[derive(Clone)]
struct UIMatrix {
    h3: Homography,
    on: bool,
    inverse: bool,
    name: String,
}

impl UIMatrix {
    fn new() -> Self {
        UIMatrix {
            h3: Homography::I,
            on: true,
            inverse: false,
            name: "".to_string(),
        }
    }
}

struct SingleImage {
    color_image: ColorImage,
    alpha: u8,
    h3s: Vec<UIMatrix>,
}

pub struct AppData {
    images: Vec<SingleImage>,
    central_index: usize,
    fill_canvas: bool,
    out_size_factor: f32,
    blend_all: bool,
    save_path: Option<String>,
    fft: FFTParams,
}

impl AppData {
    pub fn new() -> Self {
        let path = std::path::PathBuf::from("./img/lena-gray.png");
        let color_image = {
            let ci = load_image_from_path(&path);

            if ci.is_ok() {
                ci.unwrap()
            } else {
                get_lena().unwrap()
            }
        };

        let h3s = vec![UIMatrix::new(); 10];

        let images = vec![SingleImage {
            color_image,
            alpha: 255,
            h3s,
        }];

        Self {
            images,
            central_index: 0,
            fill_canvas: true,
            out_size_factor: 1.0,
            blend_all: false,
            save_path: None,
            fft: Default::default(),
        }
    }

    fn files_dropped(&mut self, files: &[egui::DroppedFile]) {
        if !files.is_empty() {
            let mut todo_files: Vec<_> = files.iter()
                .filter_map(|f| f.clone().path)
                .map(|f| f.to_path_buf())
                .collect();

            todo_files.sort();

            for f in todo_files.iter() {
                if let Ok(color_image) = load_image_from_path(&f) {
                    let h3s = vec![UIMatrix::new(); 10];

                    let si = SingleImage {
                        color_image,
                        alpha: 255,
                        h3s,
                    };

                    self.images.push(si);
                }
            }
        }
    }

    fn get_central_image_mut(&mut self) -> &mut SingleImage {
        &mut self.images[self.central_index]
    }

    fn get_central_image(&self) -> &SingleImage {
        &self.images[self.central_index]
    }

    fn display_thumbs(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let out_size = egui::Vec2::new(140.0, 140.0);

            for (ind, img) in self.images.iter_mut().enumerate() {
                let texture = ctx.load_texture(format!("thumb"), img.color_image.clone(), egui::TextureFilter::Linear);

                ui.vertical(|ui| {
                    if ui.add(egui::ImageButton::new(&texture, out_size)).clicked() {
                        self.central_index = ind;
                    }

                    //ui.image(&texture, out_size);
                    //ui.add(egui::DragValue::new(&mut img.alpha).speed(1));
                    ui.add(egui::Slider::new(&mut img.alpha, 0..=255));
                });
            }

            ui.label("Drag and drop images");
        });
    }

    fn display_homographies_panel(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .hscroll(true)
            .vscroll(false)
            .always_show_scroll(true)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                ui.horizontal(|ui|{
                    for (index, uimx) in self.get_central_image_mut().h3s.iter_mut().enumerate() {
                        display_h3(ui, uimx, index.try_into().unwrap());
                    }
                });
            });
    }

    fn display_image(&self, ctx: &egui::Context, ui: &mut egui::Ui, single_image: &SingleImage, available_width: f32, available_height: f32, texid: String, rect: &mut Option<egui::Rect>) {
        let mut h = Projection::scale(1.0, 1.0);

        for uimx in single_image.h3s.iter() {
            h = get_projection(&uimx) * h;
        }

        //display_homography(ui, &h);

        let (out_w, out_h) = {
            if self.fill_canvas {
                let out_w = available_width * self.out_size_factor;
                let out_h = available_height * self.out_size_factor;

                let tx = (out_w - self.get_central_image().color_image.size[0] as f32) / 2.0;
                let ty = (out_h - self.get_central_image().color_image.size[1] as f32)/ 2.0;
                let translation = Projection::translate(tx, ty);
                h = translation * h;

                (out_w as u32, out_h as u32)
            } else {
                (self.get_central_image().color_image.size[0] as u32, self.get_central_image().color_image.size[1] as u32)
            }
        };

        let img = warp_image(out_w, out_h, &single_image.color_image, &h, single_image.alpha);

        if let Some(file_path) = &self.save_path {
            save_image(&img, &file_path);
        }

        let out_size = egui::Vec2::new(out_w as f32 / self.out_size_factor, out_h as f32 / self.out_size_factor);

        let texture = ctx.load_texture(texid, img.clone(), egui::TextureFilter::Linear);

        if let Some(rect) = rect {
            let imgw = egui::Image::new(&texture, out_size);
            ui.put(*rect, imgw);
        } else {
            *rect = Some(ui.image(&texture, out_size).rect);
        }

        if false && self.should_display_fft() {
            let fft_img = fft_2D(&img, &self.fft);
            let texture = ctx.load_texture("fft", fft_img, egui::TextureFilter::Linear);
            ui.image(&texture, out_size);
        }
    }

    fn should_display_fft(&self) -> bool {
        !(self.blend_all || self.fill_canvas)
    }

    fn display_images(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let available_width = ui.available_width();
        let available_height = ui.available_height();
        let mut rect = None;

        if self.blend_all {
            for (ind, si) in self.images.iter().enumerate() {
                self.save_path = None;
                self.display_image(ctx, ui, si, available_width, available_height, format!("dddimg{}", ind), &mut rect);
            }
        } else {
            self.display_image(ctx, ui, self.get_central_image(), available_width, available_height, format!("daimg0"), &mut rect);
        }

        self.save_path = None;
    }

    fn display_out_size_factor(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.fill_canvas, "Fill canvas".to_string());

        if self.fill_canvas {
            ui.add(egui::Slider::new(&mut self.out_size_factor, 0.5..=1.0).text("out size factor (change if slow performance)"));
        }

        ui.checkbox(&mut self.blend_all, "Blend all".to_string());

        if !self.blend_all {
            if ui.button("Save").clicked() {
                self.save_path = tinyfiledialogs::save_file_dialog("Save File", "");
            }
        }

        if self.should_display_fft() {
            ui.radio_value(&mut self.fft.fft_type, FFTType::Horizontal, "FFT horizontal");
            ui.radio_value(&mut self.fft.fft_type, FFTType::Vertical, "FFT vertical");
            ui.radio_value(&mut self.fft.fft_type, FFTType::TwoDimensional, "FFT 2D");

            ui.checkbox(&mut self.fft.central, "central");
        }
    }
}

impl eframe::App for AppData {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.files_dropped(&ctx.input().raw.dropped_files[..]);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui|{
                self.display_thumbs(ctx, ui);
                self.display_homographies_panel(ui);
                self.display_out_size_factor(ui);
                self.display_images(ctx, ui);
            });
        });

        // if cfg!
        //ctx.request_repaint(); // we want max framerate
    }
}

