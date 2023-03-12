pub struct Image {
    pub size: [usize; 2],
    pub texture: egui::TextureHandle,
    pub image: egui::ColorImage,
}

impl Image {
    pub fn new(image: egui::ColorImage, texture: egui::TextureHandle) -> Image {
        Image {
            size: image.size,
            image,
            texture,
        }
    }
}

