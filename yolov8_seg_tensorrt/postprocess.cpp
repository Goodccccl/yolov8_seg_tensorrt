#include"postprocess.h"


float iou(Anchor box1, Anchor box2)
{
	/* 两个anchor之间的iou得分 */
	float box1_w = box1.box[2] - box1.box[0];
	float box1_h = box1.box[3] - box1.box[1];
	float box1_area = box1_w * box1_h;
	float box2_w = box2.box[2] - box2.box[0];
	float box2_h = box2.box[3] - box2.box[1];
	float box2_area = box2_w * box2_h;
	float x1 = std::max(box1.box[0], box2.box[0]);
	float y1 = std::max(box1.box[1], box2.box[1]);
	float x2 = std::min(box1.box[2], box2.box[2]);
	float y2 = std::min(box1.box[3], box2.box[3]);
	float w = x2 - x1;
	float h = y2 - y1;
	float over_area = w * h;
	float iou_score = over_area / (box1_area + box2_area - over_area);
	return iou_score;
}

static bool sort_score(Anchor box1, Anchor box2)
{
	return box1.conf > box2.conf ? true : false;
}

std::vector<Anchor> nms(Detection outputs_arrange, float threshold)
{
	std::vector<Anchor> nms_results;
	std::sort(outputs_arrange.anchors.begin(), outputs_arrange.anchors.end(), sort_score);	// 用得分进行排序
	while (outputs_arrange.anchors.size() > 0)
	{
		nms_results.push_back(outputs_arrange.anchors[0]);
		int index = 1;
		while (index < outputs_arrange.anchors.size())
		{
			float iou_score = iou(outputs_arrange.anchors[0], outputs_arrange.anchors[index]);
			if (iou_score > threshold)
			{
				outputs_arrange.anchors.erase(outputs_arrange.anchors.begin() + index);		// 删除
			}
			else
			{
				index++;
			}
		}
		outputs_arrange.anchors.erase(outputs_arrange.anchors.begin());
	}
	return nms_results;
}


//std::vector<float*> get_mask(std::vector<Anchor> nms_result, std::vector<float*> output1)
//{
//	int n = nms_result.size();
//	std::vector<float*> masks;
//	//float* mask_ = (float*)malloc(4 * 160 * 160 * sizeof(float));
//	for (int i = 0; i < n; i++) {
//		float* mask_ = (float*)malloc(160 * 160 * sizeof(float));
//		std::vector<float> mask_1 = nms_result[i].mask;
//		for (int j = 0; j < 160 * 160; j++) {
//			int mul = 0;
//			for (int k = 0; k < 32; k++) {
//				float t = mask_1[k] * output1[k][j];
//				mul += t;
//			}
//			mask_[j] = 1 / (1 + exp(-mul));	// sigmoid
//		}
//		masks.push_back(mask_);
//		//free(mask_);
//	}
//	return masks;
//}


cv::Mat get_mask(std::vector<Anchor> nms_result, cv::Mat proto)
{
	int n = nms_result.size();
	cv::Mat nms_mask;
	for (int i = 0; i < n; i++) {
		nms_mask.push_back(nms_result[i].mask);
	}
	nms_mask = nms_mask.reshape(1, {32, n});
	cv::Mat masks = (nms_mask * proto).t();
	masks = masks.reshape(4, { 160, 160 });
	return masks;
}


//void crop_mask(std::vector<float*> &masks, std::vector<Anchor> nms_result)
//{
//	int n = masks.size();
//	for (int i = 0; i < n; i++)
//	{
//		//float *mask = masks[i];
//		float *box = nms_result[i].box;
//		//float a = box[0];
//		//float b = box[1];
//		//float c = box[2];
//		//float d = box[3];
//		float x1_ = box[0] * 160 / 640;
//		float y1_ = box[1] * 160 / 640;
//		float x2_ = box[2] * 160 / 640;
//		float y2_ = box[3] * 160 / 640;
//		for (int row = 0; row < 160; row++) {
//			for (int col = 0; col < 160; col++) {
//				if (row>=x1_ and row<x2_ and col>=y1_ and col<y2_) {
//					/*int a = int(masks[i][row * 160 + col] * 255);*/
//					masks[i][row * 160 + col] = int(masks[i][row * 160 + col] * 255);
//					//if (masks[i][row * 160 + col] != 0)
//					//{
//					//	masks[i][row * 160 + col] = 255;
//					//}
//					//else
//					//{
//					//	masks[i][row * 160 + col] = 0;
//					//}
//				}
//				else {
//					masks[i][row * 160 + col] = 0;
//				}
//			}
//		}
//	}
//}