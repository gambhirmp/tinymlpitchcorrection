#!/usr/bin/env python3
import pathlib
import runpy

if __name__ == "__main__":
    target = pathlib.Path(__file__).with_name("pipeline").joinpath("preprocess.py")
    runpy.run_path(str(target), run_name="__main__")


