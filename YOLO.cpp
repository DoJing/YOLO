//
// Created by dojing on 2020/9/16.
//

#include "YOLO.h"

YOLO::YOLO(float confThreshold, float nmsThreshold, int inpWidth, int inpHeight)
{
    mfConfThreshold = confThreshold;
    mfNmsThreshold = nmsThreshold;

    mInpWidth = inpWidth;
    mInpHeight = inpHeight;
}

void YOLO::detect_image(string image_path, string modelWeights, string modelConfiguration, string classesFile, std::string& outputFile)
{
    // Load names of vClasses
    ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) vClasses.push_back(line);

    // Load the network
    dnn::Net net = dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);

    // Create a window
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_AUTOSIZE);

    // Create a 4D blob from a frame.
    cv::Mat blob;
    cv::Mat frame = cv::imread(image_path);

    // Scale transformation, scaling, subtracting mean, channel transformation
    dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(mInpWidth, mInpHeight), Scalar(0, 0, 0), true, false);

    // Sets the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Remove the bounding boxes with low confidence
    postprocess(frame, outs);

    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    std::vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = format("Inference time for a frame : %.2f ms", t);

    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

    // Write the frame with the detection boxes
    imshow(kWinName, frame);
    cv::imwrite(outputFile, frame);
}

void YOLO::detect_video(string video_path, string modelWeights, string modelConfiguration, string classesFile, std::string& outputFile)
{
    // Load names of vClasses
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) vClasses.push_back(line);

    // Load the network
    dnn::Net net = dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    VideoWriter video;

    Mat frame, blob;

    try {
        // Open the video file
        ifstream ifile(video_path);
        if (!ifile) throw("error");
        cap.open(video_path);
    }
    catch (...) {
        cout << "Could not open the input image/video stream" << endl;
        return;
    }

    // Get the video writer initialized to save the output video
    video.open(outputFile, VideoWriter::fourcc('M', 'J', 'P', 'G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));

    // Create a window
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    // Process frames.
    while (waitKey(1) < 0)
    {
        // get frame from the video
        cap >> frame;

        // Stop the program if reached end of video
        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            cout << "Output file is stored as " << outputFile << endl;
            waitKey(3000);
            break;
        }

        // Create a 4D blob from a frame.
        dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(mInpWidth, mInpHeight), Scalar(0, 0, 0), true, false);

        //Sets the input to the network
        net.setInput(blob);

        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));

        // Remove the bounding boxes with low confidence
        postprocess(frame, outs);

        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;

        string label = format("Inference time for a frame : %.2f ms", t);

        cv::putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

        // Write the frame with the detection boxes
        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        video.write(detectedFrame);

        imshow(kWinName, frame);
    }

    cap.release();
    video.release();
}

// Scan through all the bounding boxes output from the network and keep only the
// ones with high confidence scores. Assign the box's class label as the class
// with the highest score for the box.
void YOLO::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs)
{
    for (size_t i = 0; i < outs.size(); ++i)
    {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Point2i classIdPoint;
            double confidence;
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);

            // Get the maximum score value in a matrix or vector and locate it
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (confidence > mfConfThreshold)
            {
                // Get the parameters of the rectangular box.
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);

                int left = centerX - width / 2;
                int top = centerY - height / 2;

                if (left < 0) left = 0;
                if (top < 0) top = 0;

                vClassIds.push_back(classIdPoint.x);
                vConfidences.push_back((float)confidence);
                vBoxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
    dnn::NMSBoxes(vBoxes, vConfidences, mfConfThreshold, mfNmsThreshold, vIndices);

    for (size_t i = 0; i < vIndices.size(); ++i)
    {
        int idx = vIndices[i];
        Rect box = vBoxes[idx];

        int right = box.x + box.width;
        int bottom = box.y + box.height;

        if (right > frame.cols) right = frame.cols;
        if (bottom > frame.rows) bottom = frame.rows;


        drawPred(vClassIds[idx], vConfidences[idx], box.x, box.y, right, bottom, frame);
    }
    vIndices.clear();
    vBoxes.clear();
    //vClasses.clear();
    vConfidences.clear();
}

// Draw the predicted bounding box
void YOLO::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!vClasses.empty())
    {
        CV_Assert(classId < (int)vClasses.size());
        label = vClasses[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, NULL);
    top = max(top, labelSize.height);

    cv::rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + labelSize.height), Scalar(255, 255, 255), FILLED);
    cv::putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
std::vector<String> YOLO::getOutputsNames(const cv::dnn::Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

