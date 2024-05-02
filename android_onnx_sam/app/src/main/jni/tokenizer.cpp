#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <locale>
#include <codecvt>
#include <fstream>
#include <map>

//#include <platform.h>
//#include <benchmark.h>

#include "helper.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO , "read txt", __VA_ARGS__)

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON


static std::string UTF16StringToUTF8String(const char16_t* chars, size_t len) {
    std::u16string u16_string(chars, len);
    return std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t>{}
            .to_bytes(u16_string);
}

std::string JavaStringToString(JNIEnv* env, jstring str) {
    if (env == nullptr || str == nullptr) {
        return "";
    }
    const jchar* chars = env->GetStringChars(str, NULL);
    if (chars == nullptr) {
        return "";
    }
    std::string u8_string = UTF16StringToUTF8String(
            reinterpret_cast<const char16_t*>(chars), env->GetStringLength(str));
    env->ReleaseStringChars(str, chars);
    return u8_string;
}

static std::u16string UTF8StringToUTF16String(const std::string& string) {
    return std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t>{}
            .from_bytes(string);
}

jstring StringToJavaString(JNIEnv* env, const std::string& u8_string) {
    std::u16string u16_string = UTF8StringToUTF16String(u8_string);
    auto result =env->NewString(reinterpret_cast<const jchar*>(u16_string.data()),
                                u16_string.length());
    return result;
}

static helper*h=0;

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_demo_android_1onnx_1sam_tokenizer_tokenize(JNIEnv *env, jobject thiz, jstring in) {
    // TODO: implement tokenize()
    std::string tmp= JavaStringToString(env,in);
    std::vector<float>resp=h->converts(tmp);
    jfloatArray jres = env->NewFloatArray(resp.size());

    env->SetFloatArrayRegion(jres, 0, resp.size(), resp.data());
    return jres;
}
extern "C"
JNIEXPORT jboolean JNICALL
Java_demo_android_1onnx_1sam_tokenizer_load(JNIEnv *env, jobject thiz, jstring jvocab) {
    // TODO: implement load()
    std::string vocab = JavaStringToString(env,jvocab);
    if (!h) {
        h = new helper();
    }
    int res=h->load(vocab);
    if(res==0){return 1;}
    return 0;
}