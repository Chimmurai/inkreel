# Store listing — English (en-us)

Ready to paste into Partner Center → your app → Store listings → English (United States).
Limits (Microsoft): description up to 10,000 characters of plain text, no HTML or links; short description up to 1,000 (the first 270 are what counts);
up to 20 product features of 200 characters each (each line below goes into its own field, without the leading dash); "What's new" stays empty for the first release.
Everything between the lines is the text itself; the headings are just labels.

## Product name

InkReel

## Short description

The world is made of two colors. InkReel turns videos, GIFs and photos into black and white with character: pure 1-bit ink, no grey. Pick a size and a quality level, InkReel does the rest. Offline, no ads.

## Description

The world is made of two colors. White is a pause. Black is a word. InkReel turns videos, GIFs and photos into pure black and white. Every pixel ends up either black or white, with no grey in between, yet shadows, outlines, text and texture stay readable: tones are drawn the way a person with a pen would draw them, with hatching along the shapes, dots and grain that flow into each other.

There is nothing to tweak. The look is chosen automatically for every frame. You only pick the size, the format and the quality.

Watch it while you decide. The built-in player works ahead into a buffer (like the grey stretch on a video site's seek bar) and plays from it in time with the original: pause, step frame by frame, loop, go full screen, and zoom with the mouse wheel in whole-pixel steps, so every pixel stays a crisp square.

What you can do
- Open almost anything: common video files, GIF and animated PNG/WebP, and photos including HEIC from iPhones, AVIF, TIFF and PSD.
- View the same frame as a positive and a negative at once: the line between them is a smooth S-curve that follows your cursor. Defocus blurs black and white into shades, for viewing only.
- Choose a size from 144p up to the size of the source, or type your own width and height.
- Pick one of five quality levels, from a quick "Sketch" to "Engraving", a mode in which strokes follow the shapes in the picture (Sketch, Outline, Halftone, Drawing, Engraving).
- Recolour the result with 20 two-colour palettes (ink and paper) and flip them with one click.
- Save as MP4, MKV, MOV, AVI, WebM, GIF, a PNG sequence or a compact JSON file, with the sound of the original if you want it. Save a single frame as PNG, BMP, TIFF or lossless WebP.
- Use your graphics card's video encoder (NVIDIA, AMD or Intel) when there is one, and all the cores of your processor for export, while the window stays responsive.
- Switch between Ukrainian and English at any time.

Private by design. InkReel works completely offline: your files never leave your computer. No account, no ads, no tracking.

The interface is black-and-white pixel art in the spirit of the classic media players, with keyboard shortcuts for everything you do often. There are a few small surprises hidden in it.

InkReel is free. Black and white with character.

## Product features

- Videos, GIFs and photos become pure black and white: every pixel is black or white, no grey.
- Tones are drawn like a pen drawing: hatching along shapes, dots and grain that blend into each other.
- No settings to fiddle with: the style is picked automatically for every frame.
- Five quality levels, from a fast "Sketch" to engraving-style strokes in "Engraving".
- Buffered player: pause, frame step, loop, full screen, mouse-wheel zoom in whole-pixel steps.
- "Two polarities" view and a defocus slider, for viewing only.
- Sizes from 144p to the source size, or your own width and height.
- 20 two-colour palettes with a one-click invert.
- Export to MP4, MKV, MOV, AVI, WebM, GIF, PNG sequence and JSON, with sound if you want it.
- Opens common video formats, GIF, APNG, animated WebP and photos including HEIC, AVIF, TIFF and PSD.
- Own .ikr format: "Render all" saves the computed video losslessly and lock mode opens it with no recomputation; "Normal mode" returns to the original with the already computed frames kept.
- Uses hardware video encoders (NVIDIA, AMD, Intel) and all processor cores for export.
- Ukrainian and English interface.
- Works offline. No account, no ads, no tracking.

## Search terms

black and white
binarize
1-bit
dither
halftone
video to gif
pixel art

## Notes for certification

InkReel is an offline desktop app (a classic Win32 program packaged with the Desktop Bridge). It needs no account, no sign-in and no network connection.

How to test:
1. Start InkReel.
2. Press Ctrl+O (or click the folder button) and open any video, GIF or photo, for example a short MP4 or a JPG.
3. The picture is converted to black and white and shown in the player. Change how it looks with the five quality dots, the size buttons under "Output size" and the arrows under "Palette". Space pauses a video.
4. Press Ctrl+S (or the Save button) and save the result as MP4 (video) or PNG (photo). The file appears in the folder you picked.

Please note: the first time a video is converted after installing (or after an update), InkReel prepares its image-processing code once. This can take up to about a minute on a slow PC, and the progress indicator may stay near the start meanwhile. The app is not frozen, and later conversions start at once.

## Restricted capability: runFullTrust

InkReel is a classic desktop program (Python with native libraries). It declares runFullTrust because it must start its own bundled helper processes (the FFmpeg encoder and worker processes that do the conversion) and read and write the files the user chooses anywhere on the disk, which the app container does not allow. It does not use the network and does not collect or transmit any data.

## Age rating (IARC questionnaire)

A utility / photo and video tool. Answer "No" to every question about violence, fear, sexual content, language, controlled substances, gambling, user-generated content, user interaction, sharing of location or personal information, and digital purchases. The expected rating is the lowest one (PEGI 3 / ESRB Everyone).
