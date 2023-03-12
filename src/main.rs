use eframe::egui;
use imageproc::geometric_transformations::{Projection, warp, Interpolation};
use egui::ColorImage;

mod types;

struct AppData {
    color_image: ColorImage,
    h3s: Vec<Projection>,
}

impl AppData {
    fn new() -> Self {
        let path = std::path::PathBuf::from("./img/lena-gray.png");
        let color_image = load_image_from_path(&path).unwrap();

        let h3s = vec![Projection::scale(1.0, 1.0), Projection::translate(30.0, 15.0)];

        Self {
            color_image,
            h3s,
        }
    }
}

impl eframe::epi::App for AppData {
    fn name(&self) -> &str {
        "Homography Playground"
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &epi::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui|{
                ui.horizontal(|ui|{

                });

                let mut h = Projection::scale(1.0, 1.0);

                for h3 in self.h3s.iter() {
                    h = *h3 * h;
                }

                let img = warp_image(&self.color_image, &h);
                let size = egui::Vec2::new(img.size[0] as f32, img.size[1] as f32);

                let texture = ctx.load_texture(format!("img1"), img.clone());
                ui.image(&texture, size);
            });
        });

        ctx.request_repaint(); // we want max framerate
    }
}

fn warp_image(im: &egui::ColorImage, h3: &Projection) -> egui::ColorImage {
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

    let new_img = warp(&tmp_img, h3, Interpolation::Bilinear, [255, 0, 255, 117].into());


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
    egui::ColorImage{size, pixels}
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

fn main() -> std::io::Result<()> {
    let options = Default::default();

    eframe::run_native(Box::new(AppData::new()), options);
}
