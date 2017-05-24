#!/usr/bin/env sh
set -e

/home/core/Research/UnetCompetition/caffe-unet-src/build/tools/caffe train --solver=/home/core/Research/UnetCompetition/Unet_solver.prototxt $@
