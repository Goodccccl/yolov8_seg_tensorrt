#include"preprocess.h"
#include"infer.h"
#include "postprocess.h"
#include"postprocess2.h"

int main()
{
	std::string engine_path = "D:\\TensorRT-8.6.1.6\\bin\\yolov8s-seg.trt";
	Parameters params;
	params.batch_size = 1;
	params.input_channels = 3;
	params.input_width = 1280;
	params.input_height = 720;
	params.target_h = 640;
	params.target_w = 640;
	params.conf = 0.25;
	params.iou = 0.45;
	std::string imgPath = "D:/1/2.jpg";
	std::vector<YOLOV8ScaleParams> vetyolovtparams;
	cv::Mat src = cv::imread(imgPath, 1);
	
	std::vector<Detection> outputData = inference(params, engine_path, imgPath, vetyolovtparams);
	std::vector<Anchor> nms_result = nms(outputData[0], params.iou);
	cv::Mat inferMasks = get_inferMasks(nms_result);
	std::vector<cv::Mat> Masks = get_Masks(inferMasks, outputData[0].proto);
	std::vector<cv::Mat> upMasks = cropUp_Masks(Masks, nms_result);
	int n = upMasks.size();
	cv::Mat add_Masks = addMasks(upMasks);
	cv::Mat Draw_Masks = drawMasks2(add_Masks, n);
	cv::Mat result;
	cv::addWeighted(src, 0.5, Draw_Masks, 0.5, 0.0, result);
	return 1;
}