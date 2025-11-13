# Use MSVC compiler
CXX = cl
CFLAGS = /std:c++17 /EHsc /O2 /Wall /D USE_MINIMP3 /I external\minimp3 /I external\kissfft /I external\libtorch\include /I external\libtorch\include\torch\csrc\api\include
LDFLAGS = /link /LIBPATH:external\libtorch\lib torch_cpu.lib torch.lib c10.lib

# Directories
SRC_DIR = cpp
EXTERNAL_DIR = external

# Files
SRCS = $(SRC_DIR)\main.cpp $(SRC_DIR)\filters.cpp $(SRC_DIR)\spectral.cpp $(SRC_DIR)\speech_enhance.cpp $(EXTERNAL_DIR)\kissfft\kiss_fft.c
OBJS = main.obj filters.obj spectral.obj speech_enhance.obj kiss_fft.obj
TARGET = SpeechFiltering.exe

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CFLAGS) /Fe:$(TARGET) $(OBJS) $(LDFLAGS)

main.obj:
	$(CXX) /c $(CFLAGS) $(SRC_DIR)\main.cpp /Fo:main.obj >NUL 2>&1

filters.obj:
	$(CXX) /c $(CFLAGS) $(SRC_DIR)\filters.cpp /Fo:filters.obj >NUL 2>&1

spectral.obj:
	$(CXX) /c $(CFLAGS) $(SRC_DIR)\spectral.cpp /Fo:spectral.obj >NUL 2>&1

speech_enhance.obj:
	$(CXX) /c $(CFLAGS) $(SRC_DIR)\speech_enhance.cpp /Fo:speech_enhance.obj >NUL 2>&1

kiss_fft.obj:
	$(CXX) /c $(CFLAGS) $(EXTERNAL_DIR)\kissfft\kiss_fft.c /Fo:kiss_fft.obj >NUL 2>&1

clean:
	del /Q *.obj
	del /Q $(TARGET)
