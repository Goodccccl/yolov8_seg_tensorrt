#include"postprocess2.h"


cv::Mat get_inferMasks(std::vector<Anchor> nms_result)
{
	cv::Mat inferMasks;
	int num = nms_result.size();
	for (int i = 0; i < num; i++) {
		inferMasks.push_back(nms_result[i].mask);
	}
	inferMasks = inferMasks.reshape(1, { num, 32 });
	return inferMasks;
}

std::vector<cv::Mat> get_Masks(cv::Mat inferMasks, cv::Mat proto)
{
	cv::Mat matmulRes = (inferMasks * proto).t();
	for (int i = 0; i < 160 * 160; i++) {
		for (int j = 0; j < 4; j++) {
			float a = matmulRes.at<float>(i, j);
			float sig = 1 / (1 + exp(-a));
			matmulRes.at<float>(i, j) = sig;
		}
	}
	cv::Mat Masks = matmulRes.reshape(4, { 160, 160 });
	std::vector<cv::Mat> maskChannels;
	cv::split(Masks, maskChannels);
	return maskChannels£»
}

std::vector<cv::Mat> cropUp_Masks(std::vector<cv::Mat> &Masks, std::vector<Anchor> nms_result)
{
	int n = nms_result.size();
	std::vector<cv::Mat> upMasks;
	for (int i = 0; i < n; i++) {
		float* box = nms_result[i].box;
		float x1 = box[0] * 160 / 640;
		float y1 = box[1] * 160 / 640;
		float x2 = box[2] * 160 / 640;
		float y2 = box[3] * 160 / 640;
		for (int row = 0; row < 160; row++) {
			for (int col = 0; col < 160; col++) {
				if (row < x1 or row >= x2 or col < y1 or col >= y2) {
					Masks[i].at<float>(col, row) = 0;
				}
				else {
					if (Masks[i].at<float>(col, row) > 0.5) {
						Masks[i].at<float>(col, row) = i+1;
					}
					else
					{
						Masks[i].at<float>(col, row) = 0;
					}
				}
			}
		}
		cv::resize(Masks[i], Masks[i], cv::Size(1280, 1280), cv::INTER_NEAREST);
		int x_L = 0;
		int y_L = 280;
		int Xsize = 1280;
		int Ysize = 720;
		cv::Mat upMask = Masks[i](cv::Rect(x_L, y_L, Xsize, Ysize));
		upMasks.push_back(upMask);
	}
	return upMasks;
}

std::vector<cv::Scalar> get_colocMap(int n)
{
	std::vector<cv::Scalar> colorMap;
	if (n <= 3) {
		colorMap = {
			cv::Scalar(0,0,255),
			cv::Scalar(0,255,0),
			cv::Scalar(255,0,0),
		};
	}
	else
	{
		colorMap = {
			cv::Scalar(0,0,255),
			cv::Scalar(0,255,0),
			cv::Scalar(255,0,0),
		};
		int differ = n - 3;
		for (int i = 0; i < differ; i++) {
			int c1 = int(rand() % (255 - 0 + 1) + 0);
			int c2 = int(rand() % (255 - 0 + 1) + 0);
			int c3 = int(rand() % (255 - 0 + 1) + 0);
			cv::Scalar color = cv::Scalar(c1, c2, c3);
			colorMap.push_back(color);
		}
	}
	return colorMap;
}

int getMin(std::vector<cv::Mat> upMasks, int n, int row, int col)
{
	int min = INT_MAX;
	for (int i = 0; i < n; i++) {
		int val = int(upMasks[i].at<float>(row, col));
		if (val < min and val != 0) {
			min = val;
		}
	}
	if (min == INT_MAX)
	{
		min = 0;
	}
	return min;
}

cv::Mat addMasks(std::vector<cv::Mat> upMasks)
{
	int n = upMasks.size();
	int row = upMasks[0].rows;
	int col = upMasks[0].cols;
	cv::Mat add_Masks = cv::Mat::zeros(cv::Size(1280, 720), CV_8UC1);
	for (int row_ = 0; row_ < row; row_++) {
		for (int col_ = 0; col_ < col; col_++) {
			int val = getMin(upMasks, n, row_, col_);
			add_Masks.at<uchar>(row_, col_) = val;
		}
	}
	return add_Masks;
}


cv::Mat drawMasks(std::vector<cv::Mat> upMasks)
{
	int n = upMasks.size();
	cv::Mat Draw_Masks = cv::Mat::zeros(cv::Size(1280, 720), CV_8UC3);
	std::vector<cv::Scalar> colorMap = get_colocMap(n);
	for (int i = 0; i < n; i++)
	{
		int row = upMasks[i].rows;
		int col = upMasks[i].cols;
		for (int row_ = 0; row_ < row; row_++) {
			for (int col_ = 0; col_ < col; col_++) {
				if (int(upMasks[i].at<float>(row_, col_)) != 0)
				{
					Draw_Masks.at<cv::Vec3b>(row_, col_)[0] = colorMap[i][0];
					Draw_Masks.at<cv::Vec3b>(row_, col_)[1] = colorMap[i][1];
					Draw_Masks.at<cv::Vec3b>(row_, col_)[2] = colorMap[i][2];
				}
			}
		}
	}
	return Draw_Masks;
}


cv::Mat drawMasks2(cv::Mat add_Masks, int n)
{
	cv::Mat Draw_Masks = cv::Mat::zeros(cv::Size(1280, 720), CV_8UC3);
	std::vector<cv::Scalar> colorMap = get_colocMap(n);
	int row = add_Masks.rows;
	int col = add_Masks.cols;
	for (int row_ = 0; row_ < row; row_++) {
		for (int col_ = 0; col_ < col; col_++) {
			int v = int(add_Masks.at<int>(row_, col_));
			if (v != 0)
			{
				Draw_Masks.at<cv::Vec3b>(row_, col_)[0] = colorMap[v-1][0];
				Draw_Masks.at<cv::Vec3b>(row_, col_)[1] = colorMap[v-1][1];
				Draw_Masks.at<cv::Vec3b>(row_, col_)[2] = colorMap[v-1][2];
			}
		}
	}
	return Draw_Masks;
}