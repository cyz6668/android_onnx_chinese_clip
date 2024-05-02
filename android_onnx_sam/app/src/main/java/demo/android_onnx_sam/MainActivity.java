package demo.android_onnx_sam;

import ai.onnxruntime.*;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.*;
import android.net.Uri;
import android.provider.MediaStore;
import android.util.Log;
import android.view.MotionEvent;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

public class MainActivity extends AppCompatActivity {


    // onnxruntime 环境
    public static OrtEnvironment env1;
    public static OrtSession session1;
    public static OrtEnvironment env2;
    public static OrtSession session2;

    // 显示推理耗时
    private TextView timeuse;

    // 图片识别结果显示 1024,1024区域
    private ImageView imageView;


    // 当前显示的图片的备份
    private Bitmap copy;

    // 按钮用于选择本地图片
    private Button openpic;
    private Button koupic;

    private EditText inputtext;
    private String describetext;
    private tokenizer tk=new tokenizer();

    private float[] text_embedding;
    private float[] image_embedding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 显示推理耗时
        timeuse = findViewById(R.id.timeuse);

        // 加载onnx模型
        this.loadOnnx1("imgfp32-sim.onnx");
        this.loadOnnx2("txtfp32-sim.onnx");
        this.reload();
        // 按钮用于选择本地图片
        openpic = findViewById(R.id.openpic);
        openpic.setOnClickListener(v -> openGallery());

        // 按钮用于开始抠图
        koupic = findViewById(R.id.koupic);
        koupic.setOnClickListener(v-> task());

        inputtext=findViewById(R.id.editPictureDescribe);

        // 图片区域显示本地图片识别结果,默认黑色背景
        imageView = findViewById(R.id.image);
        ViewGroup.LayoutParams layoutParams = imageView.getLayoutParams();
        layoutParams.width =672;
        layoutParams.height = 672;
        imageView.setLayoutParams(layoutParams);
    }


    // 加载onnx模型
    public void loadOnnx1(String modulePath) {

        try {
            // 获取 AssetManager 对象来访问 src/main/assets 目录,读取模型的字节数组
            AssetManager assetManager = getAssets();
            InputStream inputStream = assetManager.open(modulePath);
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[1024];
            while ((nRead = inputStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }
            buffer.flush();
            byte[] module = buffer.toByteArray();
            System.out.println("开始加载模型");
            env1 = OrtEnvironment.getEnvironment();
            session1 = env1.createSession(module, new OrtSession.SessionOptions());
            // 打印模型信息,获取输入输出的shape以及类型：
            session1.getInputInfo().entrySet().stream().forEach(n -> {
                String inputName = n.getKey();
                NodeInfo inputInfo = n.getValue();
                long[] shape = ((TensorInfo) inputInfo.getInfo()).getShape();
                String javaType = ((TensorInfo) inputInfo.getInfo()).type.toString();
                System.out.println("模型输入:"+inputName + " -> " + Arrays.toString(shape) + " -> " + javaType);
            });
            session1.getOutputInfo().entrySet().stream().forEach(n -> {
                String outputName = n.getKey();
                NodeInfo outputInfo = n.getValue();
                long[] shape = ((TensorInfo) outputInfo.getInfo()).getShape();
                String javaType = ((TensorInfo) outputInfo.getInfo()).type.toString();
                System.out.println("模型输出:"+outputName + " -> " + Arrays.toString(shape) + " -> " + javaType);
            });
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public void loadOnnx2(String modulePath) {

        try {
            // 获取 AssetManager 对象来访问 src/main/assets 目录,读取模型的字节数组
            AssetManager assetManager = getAssets();
            InputStream inputStream = assetManager.open(modulePath);
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[1024];
            while ((nRead = inputStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }
            buffer.flush();
            byte[] module = buffer.toByteArray();
            System.out.println("开始加载模型");
            env2 = OrtEnvironment.getEnvironment();
            session2 = env2.createSession(module, new OrtSession.SessionOptions());
            // 打印模型信息,获取输入输出的shape以及类型：
            session2.getInputInfo().entrySet().stream().forEach(n -> {
                String inputName = n.getKey();
                NodeInfo inputInfo = n.getValue();
                long[] shape = ((TensorInfo) inputInfo.getInfo()).getShape();
                String javaType = ((TensorInfo) inputInfo.getInfo()).type.toString();
                System.out.println("模型输入:"+inputName + " -> " + Arrays.toString(shape) + " -> " + javaType);
            });
            session2.getOutputInfo().entrySet().stream().forEach(n -> {
                String outputName = n.getKey();
                NodeInfo outputInfo = n.getValue();
                long[] shape = ((TensorInfo) outputInfo.getInfo()).getShape();
                String javaType = ((TensorInfo) outputInfo.getInfo()).type.toString();
                System.out.println("模型输出:"+outputName + " -> " + Arrays.toString(shape) + " -> " + javaType);
            });
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    // 选择相册图片
    public void openGallery(){
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, 1);
    }


    //浏览相册图片信息
    public void readallimgs(){
        inference(describetext);
        String path="/storage/emulated/0/Pictures/Screenshots"; //相册存储地址
        File filepath = new File(path);
        File[] imgfiles=filepath.listFiles();
        float biggest=0;
        String save_img_path=",,";
        Bitmap save_img = null;
        for(File imgfile : imgfiles) {
            // 检查文件是否存在
            if (imgfile.exists()) {
                // 使用BitmapFactory.decodeFile()方法从文件路径加载图片为Bitmap
                Bitmap bm = BitmapFactory.decodeFile(imgfile.getAbsolutePath());
                Bitmap tmp=inferencr(bm);
                float tmp_res=get_similarity();
                if(tmp_res>biggest){
                    biggest=tmp_res;
                    save_img_path=imgfile.getAbsolutePath();
                    save_img=bm;
                }
            }
        }
//        System.out.println(save_img_path);
        Bitmap show = Bitmap.createScaledBitmap(save_img,672,672,false);
        imageView.setImageBitmap(show);
    }

    // 返回选择的图片
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == RESULT_OK && data != null) {
            Uri imageUri = data.getData();
            try {
                // 清空上次选择的图片和抠图点
                copy = null;
//                points.clear();
                // 显示选择的图片
                // 获取图片输入流,转为bitmap
                ContentResolver contentResolver = this.getContentResolver();
                InputStream inputStream = contentResolver.openInputStream(imageUri);
                Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                // 上面bitmap是本地文件不可变,这里创建一个副本来处理
                Bitmap show = Bitmap.createScaledBitmap(bitmap,672,672,false);
                imageView.setImageBitmap(show);
                // 保存备份
                copy = show.copy(Bitmap.Config.ARGB_8888,true);
            }
            catch (Exception e){
                e.printStackTrace();
            }
        }
    }

    // 对选择的图片进行处理
    private static AtomicBoolean doing = new AtomicBoolean(false);
    private void task(){
        describetext=inputtext.getText().toString();
        if (describetext.length()==0){
            System.out.println("请填写图片描述");
        }
        // 正在执行
        if(doing.get()){
            System.out.println("正在读图中");
            return;
        }
        else{
            doing.set(true);
            System.out.println("开始读图");
            try {
                // 推理
//                long t1 = System.currentTimeMillis();
//                Bitmap out = inferencr(copy);
//                long t2 = System.currentTimeMillis();
//                // 显示标注后的图片
////                imageView.setImageBitmap(out);
//                // 显示耗时
//                timeuse.setText("推理耗时1："+(t2-t1)+"ms");
//                inference(describetext);
//                float res=get_similarity();
//                if(res>=0.5){
//                    System.out.println(res);
//                    System.out.println("图片和文字叙述相似");
//                }else{
//                    System.out.println(res);
//                    System.out.println("图片和文字叙述不同");
//                }
                long t3 = System.currentTimeMillis();
                readallimgs();
                long t4 = System.currentTimeMillis();
                // 显示耗时
                timeuse.setText("推理耗时："+(t4-t3)+"ms");
            }
            catch (Exception e){
                e.printStackTrace();
            }
            doing.set(false);
            System.out.println("搜图完成");
        }
    }



    // 图片预处理推理和后处理
    private Bitmap inferencr(Bitmap copy){
        // 模型1参数
        int module_1_c = 3;
        int module_1_h = 224;
        int module_1_w = 224;
        // 图片备份
        Bitmap bitmap = Bitmap.createScaledBitmap(copy,224,224,false);
        // chw 排放
        float[] rgb = new float[ module_1_c * module_1_h * module_1_w ];
        int[] pixels = new int[ module_1_w * module_1_h ];
        bitmap.getPixels(pixels, 0, module_1_w, 0, 0, module_1_w, module_1_h);
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            // rgb分量归一化处理
            float r = ((pixel >> 16) & 0xFF) / 255f;
            float g = ((pixel >> 8) & 0xFF) / 255f;
            float b = (pixel & 0xFF) / 255f;
            // 存储到一维数组中
            rgb[i] = r;
            rgb[i+(module_1_w*module_1_h)] = g;
            rgb[i+(module_1_w*module_1_h)+(module_1_w*module_1_h)] = b;
        }
        // 计算均值和标准差
        float mean = 0.0f;
        float std = 0.0f;
        for (int i = 0; i < rgb.length; i++) {
            mean += rgb[i];
        }
        mean /= rgb.length;
        // 每个元素再进行归一化
        for (int i = 0; i < rgb.length; i++) {
            std += Math.pow(rgb[i] - mean, 2);
        }
        std = (float) Math.sqrt(std / rgb.length);

        // 再将所有元素减去均值并除标准差
        for (int i = 0; i < rgb.length; i++) {
            rgb[i] = (rgb[i] - mean) / std;
        }

        try {
            // 推理1
            OnnxTensor tensor = OnnxTensor.createTensor(env1, FloatBuffer.wrap(rgb), new long[]{1,module_1_c,module_1_h,module_1_w});
            OrtSession.Result res = session1.run(Collections.singletonMap("image", tensor));
            float[] image_embeddings = ((float[][])(res.get(0)).getValue())[0];
            image_embedding=image_embeddings;
            // 输出 256 * 64 * 64
            System.out.println(image_embeddings.length);
            // 推理成功
            System.out.println("图片推理成功!");
        }
        catch (Exception e){
            e.printStackTrace();
        }

        return bitmap;
    }


    //文字预处理推理和后处理
    public void inference(String input_text){
        float[]res_tmp=tk.tokenize(input_text);
        long[] res = new long[52];
        for(int i=0;i<res_tmp.length;i++){
            res[i]=(long)res_tmp[i];
        }
        System.out.println(res.length);
        try {
            // 推理1
            OnnxTensor tensor2 = OnnxTensor.createTensor(env2, LongBuffer.wrap(res), new long[]{1,52});
            OrtSession.Result res2 = session2.run(Collections.singletonMap("text", tensor2));
            float[] text_embeddings = ((float[][])(res2.get(0)).getValue())[0];
            text_embedding=text_embeddings;
            System.out.println(text_embeddings.length);
            // 推理成功
            System.out.println("文字推理成功!");
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    public float[] normalize(float[]v){
        double norm=0;
        for(float num:v){
            norm+=num*num;
        }
        norm=Math.sqrt(norm);
        for(int i=0;i<v.length;i++){
            v[i]= (float) (v[i]/norm);
        }
        return v;
    }

    public float get_similarity(){
        image_embedding=normalize(image_embedding);
        text_embedding=normalize(text_embedding);
        float similarity_num=0;
        for(int i=0;i<1024;i++){
            similarity_num+=image_embedding[i]*text_embedding[i];
        }
        return similarity_num;
    }

    private void reload() {
        String file="/storage/emulated/0/Documents/NewTextFile.txt";
        boolean ret_init = tk.load(file);
        if (!ret_init) {
            Log.e("MainActivity", "loadModel failed");
        }
    }

}