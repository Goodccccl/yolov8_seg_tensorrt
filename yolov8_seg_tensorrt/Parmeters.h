#pragma once
#include"includes.h"

 struct YOLOV8ScaleParams
{
	 float ratio;
	 int pad_w;
	 int pad_h;
	 int unpad_w;
	 int unpad_h;
};

 struct EngineParams
 {
	 std::vector<unsigned char> model_data;
	 nvinfer1::IRuntime* runtime;
	 nvinfer1::ICudaEngine* engine;
	 nvinfer1::IExecutionContext* context;
 };

 struct Anchor {
	 float box[4];
	 float conf;
	 int class_id;
	 std::vector<float> mask;
 };

 struct Detection {
	 std::vector<Anchor> anchors;
	 cv::Mat proto;
 };

 struct Parameters {
	 int batch_size;
	 int input_channels;
	 int input_width;
	 int input_height;
	 int target_h;
	 int target_w;
	 float conf;
	 float iou;
 };

 struct BindingsParams {
	 void* bindings[3];
	 std::vector<int> bindings_mem;
	 std::vector<float*> outputs;
 };

