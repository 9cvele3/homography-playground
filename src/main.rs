use eframe::egui;

#[derive(Default)]
struct AppData {
    img: Image,
}

impl AppData {
    fn new() -> Self {
        let path = std::path::PathBuf::from("./img/lenna.png");
        let color_image = load_image_from_path(&path).unwrap();
        Self{ img }
    }
}

impl eframe::epi::App for AppData {
    fn name(&self) -> &str {
        "Homography Playground"
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &epi::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.image(self.img.texture, self.img.size);
        });

        ctx.request_repaint(); // we want max framerate
    }
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

    eframe::run_native(Box::new(AppData::default()), options);
}
