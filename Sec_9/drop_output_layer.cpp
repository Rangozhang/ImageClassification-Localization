#include <vector>

#include "caffe/layers/drop_output_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DropOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
  top[0]->ReshapeLike(*bottom[0]);
  top[1]->ReshapeLike(*bottom[0]);

}


template <typename Dtype>
void DropOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label1 = bottom[1]->cpu_data();  // class labels
  const Dtype* bottom_label2 = bottom[2]->cpu_data();  // bbox coordinates 
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* top_label = top[1]->mutable_cpu_data();
  
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int reg_num = bottom[2]->count() / bottom[1]->count();
  CHECK(reg_num == 4);

  for (int i = 0; i < count; i ++) 
  {
    top_data[i] = 0;
    top_label[i] = 0;
  }

  for (int i = 0; i < num; i ++)
  {
	int ind = bottom_label1[i];
	for (int j = 0; j < reg_num; j++){
		top_data[i*channels+ind*reg_num+j] = bottom_data[i*channels+ind*reg_num+j];
		top_label[i*channels+ind*reg_num+j] = bottom_label2[i*reg_num+j];
	}	
  }

}


template <typename Dtype>
void DropOutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* bottom_label1 = bottom[1]->cpu_data();
  const Dtype* bottom_label2 = bottom[2]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int reg_num = bottom[2]->count() / bottom[1]->count();

  for (int i = 0; i < count; i ++) 
  {
    bottom_diff[i] = 0;
  }

  for (int i = 0; i < num; i ++)
  {
	int ind = bottom_label1[i];
	for (int j = 0; j < reg_num; j++){
		bottom_diff[i*channels+ind*reg_num+j] = top_diff[i*channels+ind*reg_num+j];
	}
  }

}


#ifdef CPU_ONLY
STUB_GPU(DropOutputLayer);
#endif

INSTANTIATE_CLASS(DropOutputLayer);
REGISTER_LAYER_CLASS(DropOutput);

}  // namespace caffe
