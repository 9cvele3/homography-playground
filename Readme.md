# Homography Playground

Interactive homography playground.

## Build for wasm

This is based on https://benw.is/posts/better-desktop-web-apps

In short, `trunk` should be used to serve the data. On first build / page hit - it will build the wasm code and store in `dist` directory. It will also make necessary changes to index.html.

Later, you can just use the files from `dist` directory.

* update egui to `0.19.0` to match the ```eframe_template``` sample

* steal the `index.html` from ```egui_template``` and remove references to unused assets. Key element in `index.html` is the canvas and canvas id.

* add additional sections to Cargo.toml like in ```eframe_template```

* reorganize your code into `app.rs`, `lib.rs` and `main.rs` (with two main functions)

* Change your platform specific code (eg. reading files from disk)

* run `cargo build --release` to build x86 app and generate `Cargo.lock`

* edit `Cargo.lock` to avoid later problems when compiling to wasm: `failed to verify the checksum of ndk-sys v0.4.0`. Make the change based on this diff: https://github.com/alacritty/alacritty/pull/6665/files

* `cargo install trunk`

* `trunk serve` and hit `127.0.0.1:8080` from private tab

## Run

`cd dist`

`python3 -m http.server 9000` and hit `127.0.0.1:9000`


# Roadmap

* Drag'n'Drop files in javascript


```rustup default nightly```

```rustup default stable```


bench - at the root of the project

test - point registration

test with 8x8 matrices - but with overlap (no too large shift)

test with 2x2 matrices - all pixels match

test with 4x4 matrices - all pixels match

test with 8x8 matrices - all pixels match

test with 16x16 matrices - gradients

# Test example




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



# Normalization

Normalization is not used in matlab code, only zero mean
G matrix has larger values than in Matlab code
Hessian matrix has larger values than in Matlab code
Hessian is not diagonal, but diagonal elements should be larger than side elements.
Hessian inverse then has small values, that result in small p increments

No zero mean for matrix G. It is not done in matlab.

Only using unique indices pairs, to get proper Hessain matrix.

Use gradient from Octave.

# Trans, Trz should match up to a point

Other uses of Hessain

# ECC Registration Performance
Rust - nalgebra performance ?

Maybe to use BLAS (https://docs.rs/blas/latest/blas/) ?

Looks like the diff is transposed

Octave reshape vs Matlab reshape

a b c
d e f

a b
c d
e f

a d
b e
c f
# TODO

* Add `ecc` lib to this project
