#pragma once
#include"includes.h"
#include"Parmeters.h"
#include"preprocess.h"

std::vector<Detection> inference(Parameters params, const std::string engine_path, const std::string imgPath, std::vector<YOLOV8ScaleParams>& vetyolovtparams);