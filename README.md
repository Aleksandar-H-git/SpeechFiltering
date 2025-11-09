# SpeechFiltering — simple CLI

This small program provides a minimal `main` that accepts two arguments:

    SpeechFiltering.exe <input.(wav|mp3)>

- If the input file has extension `.wav` the program calls `processWAV` (simple validation + copy).
- If the input file has extension `.mp3` the program calls `processMP3`.

MP3 decoding uses [minimp3] if you compile the program with `-DUSE_MINIMP3` and provide the minimp3 sources/headers.

How to build (examples):

1) Using mingw / g++ (recommended for quick tests):

    g++ -std=c++11 main.cpp -o SpeechFiltering.exe

This builds without MP3 support (processMP3 will print an instruction). To enable MP3 decoding, download `minimp3.h` and `minimp3.c` (or `minimp3_ex.c`/`minimp3_ex.h`) and compile with:

    g++ -std=c++11 main.cpp minimp3.c -DUSE_MINIMP3 -o SpeechFiltering.exe

Or, if you use `minimp3_ex.c` helpers:

    g++ -std=c++11 main.cpp minimp3.c minimp3_ex.c -DUSE_MINIMP3 -o SpeechFiltering.exe

2) Using MSVC (Developer Command Prompt):

    cl /EHsc main.cpp

To enable minimp3, add the .c sources and define `USE_MINIMP3` in project settings or via `/D USE_MINIMP3`.

Usage examples:

    # Copy or validate WAV
    SpeechFiltering.exe input.wav out.wav

    # Decode MP3 to WAV (requires minimp3 enabled)
    SpeechFiltering.exe song.mp3 out.wav

Notes and limitations
- This is intentionally minimal. `processWAV` currently just validates the `RIFF` header and copies the file.
- The `processMP3` implementation expects `minimp3` APIs; the code uses `mp3dec_decode_frame` to decode into 16-bit PCM and writes a basic WAV header.
- For production use you'd want more robust WAV header handling, streaming decoding (avoid reading the whole file into memory for very large MP3s), and better error handling.

License: add your preferred license for this project.

[minimp3]: https://github.com/lieff/minimp3
