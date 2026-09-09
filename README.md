# SCU-CPP-Lab

A simple handwritten digit recognition app built with **Qt 6** and **C++20**.
It uses **LeNet-5** and **VGG16** convolutional neural networks, powered by
**OpenCV**, **Eigen** and **OpenMP**.

## Features

- Draw digits on the canvas or import a picture
- Train a model on the MNIST dataset (batch size, epochs, learning rate and
  target accuracy are configurable)
- Recognize digits and export them as PNG files
- Convert between PNG images and the MNIST format
- Merge multiple datasets
- Multi-language UI (11 languages, stored in `i18n/`)

## Build

Requirements: Qt 6.5+, CMake 3.19+, OpenCV, Eigen, OpenMP.

```bash
cmake -S . -B build
cmake --build build
```

> Note: adjust the OpenCV / Eigen paths at the top of `CMakeLists.txt`
> to match your machine.

## Project Layout

```
CMakeLists.txt      # Root build file
main.cpp            # Entry point
mainwindow.*        # UI layer
core/               # Model core (CNN layers, image/MNIST handling)
upper/              # LeNet-5, VGG16 and the language manager
i18n/               # Translation files (JSON)
```

## Install

```bash
cmake --install build
```

This copies the executable, the Qt/OpenCV runtime DLLs and the `i18n/`
translations next to it.
