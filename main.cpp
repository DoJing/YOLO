#include <string>
#include "YOLO.h"
#define YOLOV4CFG      "/media/dojing/LINUXWIN/projects/YOLO/cfg/yolov4.cfg"
#define YOLOV4WEIGHTS  "/media/dojing/LINUXWIN/projects/YOLO/cfg/yolov4.weights"
#define YOLOV4tCFG     "/media/dojing/LINUXWIN/projects/YOLO/cfg/yolov4-tiny.cfg"
#define YOLOV4tWEIGHTS "/media/dojing/LINUXWIN/projects/YOLO/cfg/yolov4-tiny.weights"
using namespace std;

int main()
{
    // Give the configuration and weight files for the model
    string modelConfiguration =  YOLOV4CFG;
    string modelWeights =  YOLOV4WEIGHTS;
    string classesFile = "/media/dojing/LINUXWIN/projects/YOLO/data/coco.names";

    // Enter an image or video
    string image_path = "/media/dojing/LINUXWIN/projects/YOLO/data/dog.jpg";
    string video_path = "/home/dojing/视频/video/dancing11.mp4";

    // Output path settings
    std::string image_outputFile = "/media/dojing/LINUXWIN/projects/result/yolov4.jpg";
    std::string video_outputFile = "/media/dojing/LINUXWIN/projects/result/yolov4_out.avi";

    //Confidence threshold;Non-maximum suppression threshold;Width of network's input image;Height of network's input image
    YOLO yolo(0.5, 0.3, 384, 384);
    //测试图片
    //yolo.detect_image(image_path, modelWeights, modelConfiguration, classesFile, image_outputFile);
    //测试视频
    yolo.detect_video(video_path, modelWeights, modelConfiguration, classesFile, video_outputFile);

    cv::waitKey(0);
    return 0;
}

