# Homography Playground

Interactive homography playground.


## Build for wasm

This is based on https://benw.is/posts/better-desktop-web-apps

In short, `trunk` should be used to serve the data. On first build / page hit - it will build the wasm code and store in `dist` directory.

Later, you can just use the files from `dist` directory.

* update egui to `0.19.0` to match the ```eframe_template``` sample

* steal the `index.html` and `assets/sw.js` from ```egui_template``` and remove references to unused assets. Key element in `index.html` is the canvas and canvas id. 

* add additional sections to Cargo.toml like in ```eframe_template```

* reorganize your code into `app.rs`, `lib.rs` and `main.rs` (with two main functions)

* run `cargo build` to build x86 app and generate `Cargo.lock`

* edit `Cargo.lock` to avoid later problems when compiling to wasm: `failed to verify the checksum of `ndk-sys v0.4.0`. Make the change based on this diff: https://github.com/alacritty/alacritty/pull/6665/files

* `cargo install trunk`

* `trunk serve` and hit `127.0.0.1:8080` from private tab

# Run

`cd dist`

`python3 -m http.server 9000` and hit `127.0.0.1:9000`

