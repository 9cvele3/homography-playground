use puffin_egui::puffin;

// native
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let use_puffin_profiler = std::env::var("USE_PUFFIN")
                                    .map(|s| s.trim().parse::<bool>().unwrap_or(false))
                                    .unwrap_or(false);

    if use_puffin_profiler {
        puffin::set_scopes_on(true); // Remember to call this, or puffin will be disabled!
    }

    tracing_subscriber::fmt::init();

    let native_options = eframe::NativeOptions::default();

    eframe::run_native("Homography Playground",
                        native_options,
                        Box::new(|_cc| Box::new(homography_playground::AppData::new())));
}

// web using trunk
#[cfg(target_arch = "wasm32")]
fn main() {
    console_error_panic_hook::set_once();// panic -> console.error
    tracing_wasm::set_as_global_default(); //tracing -> console.log

    let web_options = eframe::WebOptions::default();

    eframe::start_web("the_canvas_id", //hardcoded
                        web_options,
                        Box::new(|_cc| Box::new(homography_playground::AppData::new())));
}

