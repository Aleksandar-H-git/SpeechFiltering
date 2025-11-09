# Compiler settings
CXX = g++
CXXFLAGS = -std=c++11 -Wall -O3 -DUSE_MINIMP3 -Iexternal/minimp3 -Iexternal/kissfft
LDFLAGS = -lm

# Directories
SRC_DIR = src
EXTERNAL_DIR = external

# Source files
SRCS = $(SRC_DIR)/main.cpp \
       $(SRC_DIR)/filters.cpp \
       $(SRC_DIR)/spectral.cpp \
       $(SRC_DIR)/speech_enhance.cpp \
       $(EXTERNAL_DIR)/kissfft/kiss_fft.c

# Object files
OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.c=.o)

# Target executable
TARGET = SpeechFiltering.exe

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Compile C++ files
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile C files (for kiss_fft.c)
$(EXTERNAL_DIR)/kissfft/%.o: $(EXTERNAL_DIR)/kissfft/%.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Force rebuild
FORCE:

# Clean build artifacts
clean:
	-powershell -Command "if (Test-Path $(TARGET)) { Remove-Item -Force $(TARGET) }"
	-powershell -Command "Get-ChildItem -Path $(SRC_DIR) -Filter *.o -Recurse | Remove-Item -Force"
	-powershell -Command "Get-ChildItem -Path $(EXTERNAL_DIR) -Filter *.o -Recurse | Remove-Item -Force"

# Help target
help:
	@echo Available targets:
	@echo   all     - Build the speech filtering application (default)
	@echo   clean   - Remove all build artifacts
	@echo   help    - Show this help message

.PHONY: all clean help FORCE