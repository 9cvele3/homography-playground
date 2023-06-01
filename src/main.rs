use eframe::egui;
use imageproc::geometric_transformations::{Projection, warp_into, Interpolation};
use egui::ColorImage;

mod types;

fn warp_image(out_w: u32, out_h: u32, im: &egui::ColorImage, h3: &Projection) -> egui::ColorImage {
    // convert to image::Image
    let size = im.size;
    let mut pixels = Vec::with_capacity(size[0]*4*size[1]);
    for pix in im.pixels.iter() {
        pixels.push(pix.r());
        pixels.push(pix.g());
        pixels.push(pix.b());
        pixels.push(pix.a());
    }

    let tmp_img: image::ImageBuffer<image::Rgba<u8>, Vec<_>> =
        image::ImageBuffer::from_raw(size[0] as u32, size[1] as u32, pixels)
        .expect("bad conversion");

    let out_size: [usize; 2] = [out_w as usize, out_h as usize];
    let mut new_img: image::ImageBuffer<image::Rgba<u8>, Vec<_>> =
        image::ImageBuffer::new(out_w, out_h);

    warp_into(&tmp_img, h3, Interpolation::Bilinear, [255, 0, 255, 117].into(), &mut new_img);

    // convert back to egui::ColorImage
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
}

fn get_projection(uimx: &UIMatrix) -> Projection {
    if !uimx.on {
        return Projection::scale(1.0, 1.0);
    }

    let h3 = match uimx.h3 {
        Homography::I => { Projection::scale(1.0, 1.0) },
        Homography::R{angle} => { Projection::rotate(angle * 2.0 * 3.14 / 360.0 ) },
        Homography::T{tx, ty} => {Projection::translate(tx, ty)},
        Homography::S{sx, sy, isotropic} => {Projection::scale(sx, sy)},
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

                /*
                if *isotropic {
                    sy = *sx;
                }
                */
            },
            Homography::T{tx, ty}=>{
                selected_text = "Trans";

                ui.label("Trans");
                ui.add(egui::Slider::new(tx, -1000.0..=1000.0));
                ui.add(egui::Slider::new(ty, -1000.0..=1000.0));
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
            });

        // anything can be R*S*T (just the 3)
        // projection
        // coordinate axes
        // local coordinate system
        // global coordinate system
        // resize output image
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

struct AppData {
    color_image: ColorImage,
    h3s: Vec<UIMatrix>,
}

impl AppData {
    fn new() -> Self {
        let path = std::path::PathBuf::from("./img/lena-gray.png");
        let color_image = load_image_from_path(&path).unwrap();

        let h3s = vec![UIMatrix::new(); 10];

        Self {
            color_image,
            h3s,
        }
    }

    fn display_homographies_panel(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .hscroll(true)
            .vscroll(false)
            .always_show_scroll(true)
            .show(ui, |ui| {
                ui.horizontal(|ui|{
                    for (index, uimx) in self.h3s.iter_mut().enumerate() {
                        display_h3(ui, uimx, index.try_into().unwrap());
                    }
                });
            });
    }

    fn display_image(&self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let mut h = Projection::scale(1.0, 1.0);

        for uimx in self.h3s.iter() {
            h = get_projection(&uimx) * h;
        }

        display_homography(ui, &h);

        let (out_w, out_h) = {
            if false {
                (self.color_image.size[0] as u32, self.color_image.size[1] as u32)
            } else {
                let tx = (ui.available_width() - self.color_image.size[0] as f32) / 2.0;
                let ty = (ui.available_height() - self.color_image.size[1] as f32)/ 2.0;
                let translation = Projection::translate(tx, ty);
                h = translation * h;
                (ui.available_width() as u32, ui.available_height() as u32)
            }
        };

        let img = warp_image(out_w, out_h, &self.color_image, &h);
        let out_size = egui::Vec2::new(out_w as f32, out_h as f32);

        let texture = ctx.load_texture(format!("img1"), img.clone());
        ui.image(&texture, out_size);
    }
}

impl eframe::epi::App for AppData {
    fn name(&self) -> &str {
        "Homography Playground"
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &epi::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui|{
                self.display_homographies_panel(ui);
                self.display_image(ctx, ui);
            });
        });

        ctx.request_repaint(); // we want max framerate
    }
}

fn main() -> std::io::Result<()> {
    let options = Default::default();

    eframe::run_native(Box::new(AppData::new()), options);
}

