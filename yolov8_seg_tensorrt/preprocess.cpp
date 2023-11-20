#include"preprocess.h"

bool load_image(std::string imagePath, std::vector<cv::Mat>& srcImgs)
{
	if (imagePath.size() == 0)
	{
		return false;
	}
	else
	{
		cv::Mat srcimg = cv::imread(imagePath, 1);
		int srcChannel = srcimg.channels();
		switch (srcChannel)
		{
		case 1:
			cv::cvtColor(srcimg, srcimg, cv::COLOR_GRAY2BGR);
			break;
		case 3:
			cv::cvtColor(srcimg, srcimg, cv::COLOR_BGR2RGB);
			break;
		case 4:
			cv::cvtColor(srcimg, srcimg, cv::COLOR_BGRA2BGR);
			break;
		}
		srcImgs.emplace_back(srcimg);
	}
}

void Resize(std::vector<cv::Mat> srcImgs, int target_h, int target_w, cv::Mat& dst, YOLOV8ScaleParams& pre_param)
{
	cv::Mat img = srcImgs.at(0);
	int src_h = img.rows;
	int	src_w = img.cols;
	float ratio = std::min(float(target_h) / float(src_h), float(target_w) / float(src_w));
	int unpad_h = int(src_h * ratio);
	int dh = (target_h - unpad_h) / 2;
	int unpad_w = int(src_w * ratio);
	int dw = (target_w - unpad_w) / 2;
	dst = cv::Mat(target_w, target_h, CV_8UC3, cv::Scalar(114, 114, 114));
	cv::Mat unpad_dst;
	cv::resize(img, unpad_dst, cv::Size(unpad_w, unpad_h));
	unpad_dst.copyTo(dst(cv::Rect(dw, dh, unpad_w, unpad_h)));
	pre_param.ratio = ratio;
	pre_param.unpad_h = unpad_h;
	pre_param.unpad_w = unpad_w;
	pre_param.pad_h = dh;
	pre_param.pad_w = dw;
}

bool normalization(cv::Mat src_resized, float* dst_data)
{
	/* normalize and HWC2CHW */
	if (src_resized.empty()) return false;
	int i = 0;
	for (int row = 0; row < src_resized.rows; row++)
	{
		uchar* uc_pixel = src_resized.data + row * src_resized.step;
		for (int col = 0; col < src_resized.cols; col++)
		{
			dst_data[i] = (float)uc_pixel[0] / 255.0;
			dst_data[i + src_resized.rows * src_resized.cols] = (float)uc_pixel[1] / 255.0;
			dst_data[i + src_resized.rows * src_resized.cols * 2] = (float)uc_pixel[2] / 255.0;
			uc_pixel += 3;	// uc_pixel指针移动3个位置
			++i;
		}
	}
	return true;
}

float* preprocess(std::string image_path, Parameters params, std::vector<YOLOV8ScaleParams> &vetyolovtparams)
{
	float* data = (float*)malloc(sizeof(float) * params.input_channels * params.target_w * params.target_h);
	std::vector<cv::Mat> src_Imgs;
	load_image(image_path, src_Imgs);
	cv::Mat mat_rs;
	YOLOV8ScaleParams scale_params;
	Resize(src_Imgs, params.target_h, params.target_w, mat_rs, scale_params);
	vetyolovtparams.emplace_back(scale_params);
	normalization(mat_rs, data);
	return data;
}