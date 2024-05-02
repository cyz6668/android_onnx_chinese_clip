//
// Created by chenyuezi on 2024/4/26.
//

#ifndef ANDROID_ONNX_SAM_HELPER_H
#define ANDROID_ONNX_SAM_HELPER_H
#include <vector>
#include <map>
#include <android/asset_manager_jni.h>
class helper {
public:
    helper();
private:
    std::vector<int> token2idx(std::string token);
    std::string idx2token(std::vector<int> idx);
    std::map<std::wstring, int> tokenizer_token2idx;
    std::map<int, std::wstring> tokenizer_idx2token;
    const int max_len = 52;
public:
    int load(std::string vocab);
    std::vector<float>converts(std::string text);
};


#endif //ANDROID_ONNX_SAM_HELPER_H
