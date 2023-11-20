#include"infer.h"
#include<fstream>

std::vector<unsigned char> load_modelData(const std::string enginePath)
{
	std::ifstream in(enginePath, std::ios::in | std::ios::binary);
	if (!in.is_open())
	{
		return {};
	}
	in.seekg(0, std::ios::end);
	int length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0)
	{
		in.seekg(0, std::ios::beg);
		data.resize(length);
		in.read((char*)&data[0], length);
	}
	in.close();
	return data;
}

EngineParams* load_engine(std::string enginePath)
{
	EngineParams* Engine = new EngineParams;
	Engine->model_data = load_modelData(enginePath);
	Engine->runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
	Engine->engine = Engine->runtime->deserializeCudaEngine(Engine->model_data.data(), Engine->model_data.size());
	if (Engine->engine == nullptr)
	{
		printf("Deserialize cuda engine failed!\n");
		Engine->runtime->destroy();
	}
	Engine->context = Engine->engine->createExecutionContext();
	return Engine;
}

void destory_engine(EngineParams* Engine)
{
	Engine->context->destroy();
	Engine->engine->destroy();
	Engine->runtime->destroy();
	Engine->model_data.clear();
	std::cout << "The Engine has been destroyed!" << std::endl;
}

int volume(nvinfer1::Dims dims)
{
	int nb_dims = dims.nbDims;
	int result = 1;
	for (int i = 0; i < nb_dims; i++)
	{
		result = result * dims.d[i];
	}
	return result;
}

std::vector<float> get_anchorOne(std::vector<float*> outputs, int start, int output_anchorsNb, int output_anchorsOne)
{
	int step = output_anchorsNb;
	std::vector<float> anchor_one;
	while (start < output_anchorsNb * output_anchorsOne)	// 修改模型需要改变
	{
		anchor_one.push_back(outputs[1][start]);
		start += step;
	}
	return anchor_one;
}

std::vector<float> get_anchorCls(std::vector<float> anchorOne, int output_anchorsOne)
{
	std::vector<float> anchorCls;
	for (int i = 4; i < output_anchorsOne - 32; i++) {
		float cls = anchorOne[i];
		anchorCls.push_back(cls);
	}
	return anchorCls;
}

std::vector<float> get_anchorMask(std::vector<float> anchorOne, int output_anchorsOne)
{
	std::vector<float> anchorMask;
	for (int i = 84; i < output_anchorsOne; i++) {
		float mask = anchorOne[i];
		anchorMask.push_back(mask);
	}
	return anchorMask;
}

//std::vector<float*> get_output1(std::vector<float*> outputs)
//{
//	std::vector<float*> temp;
//	for (int i = 0; i < 32; i++)
//	{
//		float* m = (float*)malloc(160 * 160 * sizeof(float));
//		int n = i;
//		for (int j = 0; j < 160 * 160; j++) {
//			m[j] = outputs[0][n];
//			n += 32;
//		}
//		temp.push_back(m);
//	}
//	return temp;
//}

std::vector<float*> get_output1(std::vector<float*> outputs)
{
	std::vector<float*> temp;
	for (int i = 0; i < 32; i++)
	{
		float* m = (float*)malloc(160 * 160 * sizeof(float));
		for (int j = 0; j < 160 * 160; j++) {
			m[j] = outputs[0][160 * 160 * i + j];
		}
		temp.push_back(m);
	}
	return temp;
}

cv::Mat get_proto(std::vector<float*> outputs)
{
	cv::Mat proto;
	proto = cv::Mat(32, 160 * 160, CV_32F, outputs[0]);
	return proto;
}

std::vector<Detection> Arrange_outputs(std::vector<float*> outputs, float conf, int output_anchorsNb, int output_anchorsOne)
{
	std::vector<Detection> outputs_arrange;
	Detection temp;
	for (int i = 0; i < output_anchorsNb; i++)
	{
		std::vector<float> anchorOne = get_anchorOne(outputs, i, output_anchorsNb, output_anchorsOne);
		Anchor temporary;
		std::vector<float> anchorCls = get_anchorCls(anchorOne, output_anchorsOne);
		temporary.conf = *std::max_element(anchorCls.begin(), anchorCls.end());
		if (temporary.conf > conf) {
			float x = anchorOne[0];
			float y = anchorOne[1];
			float w = anchorOne[2];
			float h = anchorOne[3];
			temporary.box[0] = x - w / 2;
			temporary.box[1] = y - h / 2;
			temporary.box[2] = x + w / 2;
			temporary.box[3] = y + h / 2;
			temporary.class_id = std::max_element(anchorCls.begin(), anchorCls.end()) - anchorCls.begin();
			temporary.mask = get_anchorMask(anchorOne, output_anchorsOne);
			temp.anchors.push_back(temporary);
		}
	}
	temp.proto = get_proto(outputs);
	outputs_arrange.push_back(temp);
	return outputs_arrange;
	
}


std::vector<Detection> inference(Parameters params, const std::string engine_path, const std::string imgPath, std::vector<YOLOV8ScaleParams>& vetyolovtparams)
{
	// load engine
	EngineParams* Engine = load_engine(engine_path);

	// set bindings
	BindingsParams Binding;
	int num_bindings = Engine->engine->getNbBindings();
	for (int i = 0; i < num_bindings; i++) {
		const char* name;
		int mode;
		nvinfer1::DataType dtype;
		nvinfer1::Dims dims;
		int totalSize;
		name = Engine->engine->getBindingName(i);
		mode = Engine->engine->bindingIsInput(i);
		dtype = Engine->engine->getBindingDataType(i);
		dims = Engine->engine->getBindingDimensions(i);
		totalSize = volume(dims) * sizeof(float);
		Binding.bindings_mem.push_back(totalSize);
		cudaMalloc(&Binding.bindings[i], totalSize);
		if (!mode) {
			int output_size = int(totalSize / sizeof(float));
			float* output = new float[output_size];
			Binding.outputs.emplace_back(output);
		}
	}

	float* src = preprocess(imgPath, params, vetyolovtparams);

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaMemcpy(Binding.bindings[0], src, Binding.bindings_mem[0], cudaMemcpyHostToDevice);
	Engine->context->enqueueV2(Binding.bindings, stream, nullptr);
	for (int i = 0; i < num_bindings - 1; i++) {
		cudaMemcpy(Binding.outputs[i], Binding.bindings[i + 1], Binding.bindings_mem[i + 1], cudaMemcpyDeviceToHost);
	}
	Engine->context->destroy();
	Engine->engine->destroy();
	Engine->runtime->destroy();
	std::vector<Detection> outputs_arrange;
	outputs_arrange = Arrange_outputs(Binding.outputs, params.conf, 8400, 116);
	return outputs_arrange;
}
