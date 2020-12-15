//
// Created by dojing on 2020/9/16.
//

#ifndef PROJECTS_YOLO_H
#define PROJECTS_YOLO_H
#ifndef YOLO_H
#define YOLO_H

#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "dnn/dnn.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

class YOLO
{
public:
    YOLO(float confThreshold = 0.5, float nmsThreshold = 0.4, int inpWidth = 416, int inpHeight = 416);

    void detect_image(std::string image_path, std::string modelWeights, std::string modelConfiguration, std::string classesFile, std::string& outputFile);
    void detect_video(std::string video_path, std::string modelWeights, std::string modelConfiguration, std::string classesFile, std::string& outputFile);

    // Remove the bounding boxes with low confidence using non-maxima suppression
    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs);

    // Get the names of the output layers
    std::vector<String> getOutputsNames(const cv::dnn::Net& net);

    // Draw the predicted bounding box
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

private:

    // Initialize the parameters
    float mfConfThreshold;          // Confidence threshold
    float mfNmsThreshold;           // Non-maximum suppression threshold
    int mInpWidth;                  // Width of network's input image
    int mInpHeight;                 // Height of network's input image

    std::vector<int> vClassIds;     // The index corresponding to the category name
    std::vector<string> vClasses;   // Classification name of a category
    std::vector<float> vConfidences;// Maximum confidence greater than confidence threshold
    std::vector<cv::Rect> vBoxes;   // Various category boxes
    std::vector<int> vIndices;      // Candidate box index after non-maximum suppression
};


#endif // YOLO_H


#endif //PROJECTS_YOLO_H
