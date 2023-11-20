#pragma once
#include"Parmeters.h"

cv::Mat get_inferMasks(std::vector<Anchor> nms_result);

std::vector<cv::Mat> get_Masks(cv::Mat inferMasks, cv::Mat proto);

std::vector<cv::Mat> cropUp_Masks(std::vector<cv::Mat> &Masks, std::vector<Anchor> nms_result);

cv::Mat drawMasks(std::vector<cv::Mat> upMasks);

cv::Mat drawMasks2(cv::Mat add_Masks, int n);

cv::Mat addMasks(std::vector<cv::Mat> upMasks);