use eframe::egui;
use imageproc::geometric_transformations::{Projection, warp_into, Interpolation};
use egui::ColorImage;
use puffin_egui::puffin;
use std::io::{BufRead, BufReader, Cursor, Read, Seek};

fn warp_image(out_w: u32, out_h: u32, im: &egui::ColorImage, h3: &Projection, alpha: u8) -> egui::ColorImage {
    puffin::profile_function!();

    // convert to image::Image
    let tmp_img: image::ImageBuffer<image::Rgba<u8>, Vec<_>> = {
        puffin::profile_scope!("to_image::Image", "convert");

        let size = im.size;
        let mut pixels = Vec::with_capacity(size[0]*4*size[1]);
        for pix in im.pixels.iter() {
            pixels.push(pix.r());
            pixels.push(pix.g());
            pixels.push(pix.b());
            pixels.push(alpha);
        }

        image::ImageBuffer::from_raw(size[0] as u32, size[1] as u32, pixels)
        .expect("bad conversion")
    };

    let out_size: [usize; 2] = [out_w as usize, out_h as usize];
    let new_img: image::ImageBuffer<image::Rgba<u8>, Vec<_>> = {
        puffin::profile_scope!("warp_into", "warp_into");
        let mut new_img = image::ImageBuffer::new(out_w, out_h);

        warp_into(&tmp_img, h3, Interpolation::Bilinear, [255, 0, 255, alpha].into(), &mut new_img);
        new_img
    };

    // convert back to egui::ColorImage
    let color_image = {
        puffin::profile_scope!("to_egui", "convert");

        let pixels = new_img.as_raw()
        .chunks_exact(4)
        .map(|p| {
            let lr = p[0];
            let lg = p[1];
            let lb = p[2];
            let la = p[3];
            egui::Color32::from_rgba_unmultiplied(lr, lg, lb, la)
        })
        .collect();

        egui::ColorImage{size: out_size, pixels}
    };

    color_image
}

fn load_image_from_path(path: &std::path::Path) -> Result<egui::ColorImage, image::ImageError> {
    let mut reader = image::io::Reader::open(path)?;
    load_egui_image_from_image_reader(reader)
}

fn load_image_from_bytes(data: &[u8], name: &str) -> Result<egui::ColorImage, image::ImageError> {
    println!("loading from {} bytes", data.len());
    let format = image::ImageFormat::from_path(name)?;
    let mut reader = image::io::Reader::with_format(Cursor::new(data), format);
    load_egui_image_from_image_reader(reader)
}

fn load_egui_image_from_image_reader<R: Read + BufRead + Seek>(reader: image::io::Reader<R>) -> Result<egui::ColorImage, image::ImageError> {
    let image = reader.decode()?;
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

#[derive(PartialEq, Clone)]
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
    }
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
        Homography::P{h31, h32, h33} => Projection::from_matrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, h31, h32, h33]).expect("non invertible")
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
                ui.add(egui::Slider::new(tx, -1000.0..=1000.0));
                ui.add(egui::Slider::new(ty, -1000.0..=1000.0));
            },
            Homography::P { h31, h32, h33 } => {
                selected_text = "Proj";

                ui.label("Proj");
                ui.add(egui::Slider::new(h31, -0.01..=0.01));
                ui.add(egui::Slider::new(h32, -0.01..=0.01));
                ui.add(egui::Slider::new(h33, -5.0..=5.0));
            }
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
    use_puffin_profiler: bool,
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

        let use_puffin_profiler = std::env::var("USE_PUFFIN")
                                    .map(|s| s.trim().parse::<bool>().unwrap_or(false))
                                    .unwrap_or(false);
        Self {
            images,
            central_index: 0,
            fill_canvas: true,
            out_size_factor: 1.0,
            blend_all: false,
            use_puffin_profiler,
        }
    }

    fn files_dropped(&mut self, files: &[egui::DroppedFile]) {
            let mut images : Vec<SingleImage> = files.into_iter()
                .filter_map(|df| {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        if df.path.is_some() {
                            if let Ok(color_image) = load_image_from_path(&df.path.as_ref().unwrap().to_path_buf()) {
                                let h3s = vec![UIMatrix::new(); 10];

                                let si = SingleImage {
                                    color_image,
                                    alpha: 255,
                                    h3s,
                                };

                                return Some(si);
                            }
                        }
                    }

                    #[cfg(target_arch = "wasm32")]
                    {
                        tracing::debug!("Loading image");

                        if df.bytes.is_some() {
                            match load_image_from_bytes(&df.bytes.as_ref().unwrap(), &df.name) {
                                Ok(color_image) => {
                                    let h3s = vec![UIMatrix::new(); 10];

                                    let si = SingleImage {
                                        color_image,
                                        alpha: 255,
                                        h3s,
                                    };

                                    return Some(si);
                                },
                                Err(e) => {
                                    tracing::error!("Error while loading image: {:?}", e);
                                }
                            }
                        }
                    }

                    return None
                })
                .collect();

            self.images.append(&mut images);
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
        let out_size = egui::Vec2::new(out_w as f32 / self.out_size_factor, out_h as f32 / self.out_size_factor);

        let texture = ctx.load_texture(texid, img.clone(), egui::TextureFilter::Linear);

        if let Some(rect) = rect {
            let imgw = egui::Image::new(&texture, out_size);
            ui.put(*rect, imgw);
        } else {
            *rect = Some(ui.image(&texture, out_size).rect);
        }
    }

    fn display_images(&self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let available_width = ui.available_width();
        let available_height = ui.available_height();
        let mut rect = None;

        if self.blend_all {
            for (ind, si) in self.images.iter().enumerate() {
                self.display_image(ctx, ui, si, available_width, available_height, format!("dddimg{}", ind), &mut rect);
            }
        } else {
            self.display_image(ctx, ui, self.get_central_image(), available_width, available_height, format!("daimg0"), &mut rect);
        }
    }

    fn display_out_size_factor(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.fill_canvas, "Fill canvas".to_string());

        if self.fill_canvas {
            ui.add(egui::Slider::new(&mut self.out_size_factor, 0.5..=1.0).text("out size factor (change if slow performance)"));
        }

        ui.checkbox(&mut self.blend_all, "Blend all".to_string());
    }
}

impl eframe::App for AppData {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.use_puffin_profiler {
            puffin::profile_function!();
            puffin::GlobalProfiler::lock().new_frame(); // call once per frame!
            puffin_egui::profiler_window(ctx);
        }



        egui::CentralPanel::default().show(ctx, |ui| {
            self.files_dropped(&ctx.input().raw.dropped_files[..]);

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

