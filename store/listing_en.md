# Store listing — English (en-us)

Ready to paste into Partner Center → your app → Store listings → English (United States).
Limits (Microsoft): description up to 10,000 characters of plain text, no HTML or links; short description up to 1,000 (the first 270 are what counts);
up to 20 product features of 200 characters each (each line below goes into its own field, without the leading dash); screenshot captions up to 200 characters;
"What's new" stays empty for the first release.
Everything between the lines is the text itself; the headings are just labels.

## Product name

InkReel

## Short description

The world is made of two colors. InkReel turns video, GIFs and photos into black and white with character: every pixel is ink or paper, and the tones are drawn with strokes, dots and grain. Free.

## Description

The world is made of two colors. White is a pause. Black is a word.

InkReel turns video, GIFs and photos into pure black and white. Every pixel becomes either ink or paper, with no grey in between, and yet shadows, outlines, text and texture stay readable. The tones are drawn the way a hand with a pen would draw them: strokes that follow the shapes, dots, grain that flows from one into the other.

Five levels, from a quick look to a finished print. Sketch is a fast local threshold, good for lines and text. Halftone makes the tones from noise, like a newspaper. Drawing chooses strokes, dots and grain by what is in the picture. Engraving lets the strokes follow the forms and keeps text crisp, with no flicker in video. Not sure? Try Engraving.

Next to the five levels stands IRNM1, InkReel's own neural network. It is not a sixth step but a separate branch, much faster, and it runs on your graphics card (DirectML) or, without one, on the processor.

Switch from Ink to Symbols and the frame is built from characters of a monospaced font, picked by tone, like text in a console. A photo or a frame can be saved as a .txt file.

Ink and paper do not have to be black and white. There are 20 two-colour palettes, among them Classic, Ink & paper, Carmine, Cyanotype, Amber and Terminal, and one click turns any of them inside out. It is still two colours and nothing in between.

Watch while you decide. The built-in player works ahead into a buffer and plays from it in time with the original: pause, frame step, loop, full screen. The mouse wheel zooms in whole steps up to 6400%, so every pixel stays a square, and the [ ] flags above the seek bar mark the fragment you want to save.

What goes in: common video files, GIF, animated PNG and WebP, and photos including HEIC from iPhones, AVIF, TIFF and PSD. Drop a file on the window or open it from the button.

What comes out: MP4 with the sound of the original, GIF, MKV, MOV, AVI, WebM, a PNG sequence or JSON. A single frame saves as PNG, BMP, TIFF or lossless WebP. "Render all" computes the whole video ahead and keeps it in InkReel's own .ikr file.

Pick a size from 144p up to the size of the source, or type your own width and height. "Auto" chooses one your PC can play in real time. The interface is black-and-white pixel art in the spirit of the classic media players, in Ukrainian or English.

InkReel is free. Black and white with character.

## Product features

- Video, GIFs and photos become pure black and white: every pixel is ink or paper, with no grey.
- Tones are drawn like a pen drawing: strokes along the shapes, dots and grain that flow into each other.
- Five levels, from a fast "Sketch" to "Engraving", where the strokes follow the forms.
- IRNM1: InkReel's own neural network on your graphics card (DirectML) or the processor, a separate branch next to the five levels.
- Symbols mode: the frame is built from monospaced font characters picked by tone; save it as .txt.
- 20 two-colour palettes and a one-click invert.
- A buffered player: pause, frame step, loop, full screen, and mouse-wheel zoom in whole steps up to 6400%.
- A [ ] fragment on the seek bar: save only the part you want.
- Export to MP4 with sound, GIF, MKV, MOV, AVI, WebM, PNG sequence and JSON.
- "Frame → photo": save a single frame as PNG, BMP, TIFF or lossless WebP.
- "Render all" saves the computed video in InkReel's own lossless .ikr file.
- Opens common video formats, GIF, APNG, animated WebP and photos including HEIC, AVIF, TIFF and PSD.
- Sizes from 144p to the source size, your own width and height, or "Auto" for real-time playback.
- Drag and drop a file onto the window. Ukrainian and English interface.

## Search terms

black and white
binarize
1-bit
dither
halftone
video to gif
pixel art

## What's new

(leave empty for the first release, 1.0.0)

## Screenshot captions

Files are in store/listing_assets/screenshots, the en_* set. One caption per screenshot, in this order.

1. Engraving: every pixel is ink or paper, and the strokes follow the shapes of the picture.
2. IRNM1, InkReel's own neural network, on a lake at dawn. Palette: Ink & paper.
3. The Drawing level with the Cyanotype palette: strokes, dots and grain chosen by what is in the picture.
4. Halftone with the Sepia palette: the tones are made of noise, like in a newspaper.
5. Symbols mode: the frame is built from monospaced font characters. Palette: Terminal.
6. The Outline level with the Carmine palette, on dunes seen from above.
7. IRNM1 with the Amber palette. 20 two-colour palettes, and one click turns any of them inside out.

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
