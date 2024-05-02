package demo.android_onnx_sam;

import android.content.res.AssetManager;

public class tokenizer {
    public native float[] tokenize(String in);
    public native boolean load(String vocab);
    static {
        System.loadLibrary("tokenizerlib");
    }
}
