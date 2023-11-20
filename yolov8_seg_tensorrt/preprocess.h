#pragma once
#include"includes.h"
#include"Parmeters.h"
#include"postprocess.h"

bool load_image(std::string imagePath, std::vector<cv::Mat>& srcImgs);

void Resize(std::vector<cv::Mat> srcImgs, int target_h, int target_w, cv::Mat& dst, YOLOV8ScaleParams& pre_param);

bool normalization(cv::Mat src_resized, float* dst_data);

float* preprocess(std::string image_path, Parameters params, std::vector<YOLOV8ScaleParams>& vetyolovtparams);