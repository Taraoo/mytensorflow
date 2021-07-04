#include <memory>
#include <vector>
#include <string>
#include <unordered_set>
#include <iostream>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::Status;
using tensorflow::Session;
using tensorflow::SavedModelBundle;
using tensorflow::SessionOptions;
using tensorflow::RunOptions;

int main(void)
{
    const std::string export_dir = "./model";
    SavedModelBundle bundle;
    SessionOptions session_options;
    RunOptions run_options;

    // Load model from SavedModel
    Status status = tensorflow::LoadSavedModel(session_options, run_options, export_dir, { tensorflow::kSavedModelTagServe }, &bundle);
    if (!status.ok()) {
        std::cout << "Failed to load saved model" << std::endl;
        std::cout << status.ToString() << std::endl;
        return -1;
    }

    // Create input data
    Tensor batch_size(tensorflow::DT_INT64, tensorflow::TensorShape());
    auto dst = batch_size.flat<long long>().data();
    long long bsize = 3L;
    memcpy(dst, &bsize, sizeof(bsize));

    Tensor input(tensorflow::DT_FLOAT, TensorShape({ 3, 1 }));
    auto input_dst = input.flat<float>().data();
    float arr[3] = { 2.0, 3.0, 4.0 };
    memcpy(input_dst, arr, sizeof(arr));

    // Initialize the iterator
    status = bundle.session->Run(
        { {"input", input}, {"target", input}, {"batch_size", batch_size} },
        {},
        { "dataset_init" },
        nullptr);
    if (!status.ok()) {
        std::cout << "Failed to run sesssion (dataset_init)" << std::endl;
        std::cout << status.ToString() << std::endl;
        return -1;
    }

    // Prediction
    std::vector<Tensor> outputs;
    status = bundle.session->Run({}, { "output:0" }, {}, &outputs);
    if (!status.ok()) {
        std::cout << "Failed to run sesssion (output:0)" << std::endl;
        std::cout << status.ToString() << std::endl;
        return -1;
    }

    Tensor a = outputs.at(0);
    const int out_dim = 3;
    for (int i = 0; i < out_dim; i++) {
        std::cout << a.flat<float>()(i) << std::endl;
    }

    return 0;
}