#pragma once
#include"Parmeters.h"


std::vector<Anchor> nms(Detection outputs_arrange, float threshold);

//std::vector<float*> get_mask(std::vector<Anchor> nms_result, std::vector<float*> output1);

cv::Mat get_mask(std::vector<Anchor> nms_result, cv::Mat proto);

void crop_mask(std::vector<float*> &masks, std::vector<Anchor> nms_result);