/**
Copyright 2017 Rafael Muñoz Salinas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Rafael Muñoz Salinas.
*/
#include <stdio.h>
#include <iostream>
#include <netdb.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include<netinet/in.h>

#include "iomanip"
#define portnum 12345
#define FILE_SIZE 500
#define BUFFER_SIZE 1024

#include "aruco.h"
#include "timers.h"
#include "cameraparameters.h"
#include "markermap.h"
#include "markerdetector.h"
#include "posetracker.h"
#include "sglviewer.h"
#include "cvdrawingutils.h"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <string>
#include <stdexcept>
#include <Eigen/Geometry>
#include "sglviewer.h"
using namespace cv;
using namespace aruco;
using namespace std;
string TheMarkerMapConfigFile;
bool The3DInfoAvailable = false;
float TheMarkerSize = -1;
cv::VideoCapture TheVideoCapturer;
Mat TheInputImage, TheInputImageCopy;
cv::Mat left_half;
cv::Mat right_half;        
CameraParameters TheCameraParameters;
MarkerMap TheMarkerMapConfig;
MarkerDetector TheMarkerDetector;
MarkerMapPoseTracker TheMSPoseTracker;
int width;
int height;
 int waitTime = 0;
 int ref_id = 1;
 
 char camPositionStr[100];
 char camDirectionStr[100];
std::map<int, cv::Mat> frame_pose_map;  // set of poses and the frames they were detected
aruco::sgl_OpenCV_Viewer Viewer;

uint16_t transform2(float &data) {
    int MAX = 100/0.0054933198;
    bool sig = false;
    if (data > 0) {
        sig = true;
    }
    auto d = static_cast<uint16_t>(std::abs(data /0.0054933198));
    if (d > MAX) {
        return 0;
    }

    if (!sig) d = (d | (int16_t) 32768);
    cout<<d<<endl;
    cout<<hex<<d<<endl;
    cout<<hex<<htonl(d)<<endl;
    printf("%x", htonl(d));
//    return htonl(d);
    
    return d;
}

class CmdLineParser
{
    int argc;char** argv;public:
    CmdLineParser(int _argc, char** _argv): argc(_argc), argv(_argv){}
    //is the param?
    bool operator[](string param)
    {int idx = -1;   for (int i = 0; i < argc && idx == -1; i++)if (string(argv[i]) == param) idx = i;return (idx != -1); }
    //return the value of a param using a default value if it is not present
    string operator()(string param, string defvalue = "-1"){int idx = -1;for (int i = 0; i < argc && idx == -1; i++)if (string(argv[i]) == param) idx = i;if (idx == -1) return defvalue;else return (argv[idx + 1]);}
};
void savePCDFile(string fpath, const aruco::MarkerMap& ms,
                 const std::map<int, cv::Mat> frame_pose_map);
void savePosesToFile(string filename, const std::map<int, cv::Mat>& fmp);

uint16_t transform(float &data);

uint8_t crc8(uint8_t *data, int length);

union num{
    char* bits;
};


struct data {
    char m_head;
    char m_length;
    char m_obj_class;
    char m_id;
    uint16_t m_x;
    uint16_t m_y;
    uint16_t m_z;
    uint16_t m_roll;
    uint16_t m_yaw;
    uint16_t m_pitch;
    char m_crc;

    data(int head, int length, int obj_class, int id, uint16_t x, uint16_t y, uint16_t z, uint16_t roll, uint16_t yaw,
         int16_t pitch, int crc) {
        m_head = (char) head;
        m_length = (char) length;
        m_obj_class = (char) obj_class;
        m_id = (char) id;
        m_x = (x);
        m_y = y;
        m_z = z;
        m_roll = roll;
        m_yaw = yaw;
        m_pitch = pitch;
        m_crc = (char)crc;
    }
};


/************************************
 *
 *
 *
 *
 ************************************/
int main(int argc, char** argv)
{
    try
    {
    //-----------通信---------------------------------

        //char *host = "192.168.1.236";
        //char *host = "192.168.1.81";
         //char *host = "192.168.5.64";
        char *host= "0.0.0.0";
        char *port = "8888";
        int sockfd;
        char buf[BUFFER_SIZE] = "a";
        struct sockaddr_in servaddr;
        int flag =1;


        //创建socket
        while ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
            perror("socket");
            sleep(1000);
            // exit(-1);
        }
        //设置sockaddr_in结构体中相关参数
        bzero(&servaddr, sizeof(servaddr));
        servaddr.sin_family = AF_INET;
        servaddr.sin_port = htons(atoi(port));
        servaddr.sin_addr.s_addr = inet_addr(host);

        //调用connect()向服务器端建立TCP连接
        cout << "waitting connect......." << endl;

        while (connect(sockfd, (struct sockaddr *) &servaddr, sizeof(servaddr)) == -1) {
        sleep(1);
            perror("connect");
            //exit(-1);
        }
        cout << "连接成功";
        cout << host << endl;
        
        // unsigned char buffer[BUFFER_SIZE];
        //ssize_t bytes_received = recv(sockfd, buffer, BUFFER_SIZE, 0);
       // cout<<"waiting for start command"<<endl;
       // while(!(bytes_received == 4 &&
        //    buffer[0] == 0x55 &&
        //    buffer[1] == 0x02 &&
        //    buffer[2] == 0x20 &&
       //     buffer[3] == 0x2E)) {
       //     bytes_received = recv(sockfd, buffer, BUFFER_SIZE, 0);
        //    
            // 收到开始指令，退出循环
       // }
        cout<<"start!"<<endl;
        // -----------------------------------------------feizuse-----
        int flags = fcntl(sockfd, F_GETFL, 0);
        if (flags == -1) {
            perror("fcntl F_GETFL");
            // 错误处理...
        }

        // 添加 O_NONBLOCK 标志
        flags |= O_NONBLOCK;

        if (fcntl(sockfd, F_SETFL, flags) == -1) {
            perror("fcntl F_SETFL");
            // 错误处理...
        }

        
       
        CmdLineParser cml(argc, argv);
        if (argc < 4 || cml["-h"])
        {
            cerr << "Invalid number of arguments" << endl;
            cerr << "Usage: (in.avi|live[:camera_index(e.g 0 or 1)])) marksetconfig.yml camera_intrinsics.yml [optional_arguments]  "
                    "\n\t[-s marker_size] \n\t[-pcd out_pcd_file_with_camera_poses] \n\t[-poses out_file_with_poses] "
                    "\n\t[-mti value: minimum value in range (0,1) for the size of the detected markers. If 0, ] "
                    "\n\t[-config arucoConfig.yml: Loads the detector configuration from file ] "
                 << endl;
            return false;
        }
        TheMarkerMapConfig.readFromFile(argv[2]);

        TheMarkerMapConfigFile = argv[2];
        TheMarkerSize = stof(cml("-s", "1"));
        // read from camera or from  file
        string TheInputVideo=string(argv[1]);
         if ( TheInputVideo.find( "live")!=std::string::npos)
        {
            int vIdx = 0;
            // check if the :idx is here
            char cad[100];
            if (TheInputVideo.find(":") != string::npos)
            {
                std::replace(TheInputVideo.begin(), TheInputVideo.end(), ':', ' ');
                sscanf(TheInputVideo.c_str(), "%s %d", cad, &vIdx);
            }
            cout << "Opening camera index " << vIdx << endl;
            TheVideoCapturer.open(vIdx);
            waitTime = 10;
        }
        else
            TheVideoCapturer.open(argv[1]);        // check video is open
        if (!TheVideoCapturer.isOpened())
            throw std::runtime_error("Could not open video");
         TheVideoCapturer.set(cv::CAP_PROP_FPS, 30);
        TheVideoCapturer.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
           
   
        // read first image to get the dimensions
        TheVideoCapturer >> TheInputImage;
        width = TheInputImage.cols;
        height = TheInputImage.rows;
        // 左右切分
         left_half = TheInputImage(cv::Rect(0, 0, width / 2, height));
         right_half = TheInputImage(cv::Rect(width / 2, 0, width / 2, height));
        // read camera parameters if passed
        TheCameraParameters.readFromXMLFile(argv[3]);
        TheCameraParameters.resize(left_half.size());
        // prepare the detector

        TheMarkerDetector.setDictionary( TheMarkerMapConfig.getDictionary());

        if (cml["-config"])
            TheMarkerDetector.loadParamsFromFile(cml("-config"));
         // prepare the pose tracker if possible
        // if the camera parameers are avaiable, and the markerset can be expressed in meters, then go

        if (TheMarkerMapConfig.isExpressedInPixels() && TheMarkerSize > 0)
            TheMarkerMapConfig = TheMarkerMapConfig.convertToMeters(TheMarkerSize);

        cout << "TheCameraParameters.isValid()=" << TheCameraParameters.isValid() << " "<< TheMarkerMapConfig.isExpressedInMeters() << endl;

        if (TheCameraParameters.isValid() && TheMarkerMapConfig.isExpressedInMeters()){
            TheMSPoseTracker.setParams(TheCameraParameters, TheMarkerMapConfig);
            TheMarkerSize=cv::norm(TheMarkerMapConfig[0][0]- TheMarkerMapConfig[0][1]);
        }

        // Create gui

        Viewer.setParams(1.5,640,480,"map_viewer",TheMarkerSize);
        char key = 0;
        int index = 0;
        // capture until press ESC or until the end of the video
        cout << "Press 's' to start/stop video" << endl;
        aruco::Timer avrgTimer;
        do
        {
             TheVideoCapturer.retrieve(TheInputImage);
            double fps = TheVideoCapturer.get(cv::CAP_PROP_FPS);
            std::cout << fps << std::endl;
             // 获取图像宽度和高度
             width = TheInputImage.cols;
             height = TheInputImage.rows;
            // 左右切分
            left_half = TheInputImage(cv::Rect(0, 0, width / 2, height));
            right_half = TheInputImage(cv::Rect(width / 2, 0, width / 2, height));
            
            // 左右部分分别进行水平翻转
            // cv::Mat left_flipped, right_flipped;
            // cv::flip(left_half, left_flipped, -1);
            // cv::flip(right_half, right_flipped, -1);
            left_half.copyTo(TheInputImageCopy);
            //imshow("out",TheInputImage);
            index++;  // number of images captured
            if (index>1) avrgTimer.start();//do not consider first frame that requires initialization
            // Detection of the markers
            vector<aruco::Marker> detected_markers = TheMarkerDetector.detect(TheInputImageCopy,TheCameraParameters,TheMarkerSize);
            // estimate 3d camera pose if possible
            if (TheMSPoseTracker.isValid())
                if (TheMSPoseTracker.estimatePose(detected_markers)) {
                    frame_pose_map.insert(make_pair(index, TheMSPoseTracker.getRTMatrix()));
                    cout<<"---------------------------------------------------------------------"<<endl;
                    cout<<TheMSPoseTracker.getRvec()<<" "<< TheMSPoseTracker.getTvec()<<endl;
                    cout<<"---------------------------------------------------------------------"<<endl;
                    cv::Mat rMatrix,camPosMatrix,camVecMatrix;
                    //定义旋转矩阵，平移矩阵，相机位置矩阵和姿态矩阵
		    Rodrigues(TheMSPoseTracker.getRvec(),rMatrix);
		    cv::Mat rvec;
		     Rodrigues(rMatrix.inv(),rvec);
		    //获得相机坐标系与世界坐标系之间的旋转向量并转化为旋转矩阵
		    camPosMatrix=rMatrix.inv()*(-TheMSPoseTracker.getTvec().t());
		    
		    
		    
		   //计算出相机在世界坐标系下的三维坐标
		   cv::Mat vect=(cv::Mat_<float>(3,1)<<0,0,1);
		  //定义相机坐标系下的单位方向向量，将其转化到世界坐标系下便可求出相机在世界坐标系中的姿态
		 camVecMatrix=rMatrix.inv()*vect;//计算出相机在世界坐标系下的姿态
		    
		   // ------------------------------传输数据---------------------
                    float x, y, z, roll, yaw, pitch;
                    x = camPosMatrix.at<float>(0,0);
                  //x = TheMSPoseTracker.getRvec().at<float>(0,0);
                   y = camPosMatrix.at<float>(1,0);
 		//y = TheMSPoseTracker.getRvec().at<float>(1,0);
                    z = camPosMatrix.at<float>(2,0);
                    //z = TheMSPoseTracker.getRvec().at<float>(2,0);
                    //roll = camVecMatrix.at<float>(0,0);
                    roll =rvec.at<float>(0,0);
                   // roll =TheMSPoseTracker.getTvec().at<float>(0,0);
                    yaw =rvec.at<float>(1,0);
                  // yaw =TheMSPoseTracker.getTvec().at<float>(1,0);
                    //pitch = camVecMatrix.at<float>(2,0);
		pitch = rvec.at<float>(2,0);
                    cout<<"x ="<<x<<endl;
                    cout<<"y ="<<y<<endl;
                    cout<<"z ="<<z<<endl;
                    cout<<"roll ="<<roll<<endl;
                    cout<<"yaw ="<<yaw <<endl;
                    cout<<"pitch ="<<pitch  <<endl;
                    
                    if (!(transform(x) && transform(y) && transform(z)
                          && transform2(roll) && transform2(yaw) && transform2(pitch))) {
                        cout << "there 104" << endl;
                        std::cout << "输出超过阈值" << endl;
                    } else {
                        cout << "there 137" << endl;
                        uint16_t p_x = transform(x);
                        uint16_t p_y = transform(y);
                        uint16_t p_z = transform(z);
                        Eigen::Vector3d rot_vector(roll,yaw,pitch);
                        Eigen::AngleAxisd rotation_angle_axis(rot_vector.norm(),rot_vector.normalized());
                        Eigen::Matrix3d rotation_matrix=rotation_angle_axis.toRotationMatrix();
                        cout<<"旋转矩阵"<<rotation_matrix<<endl; 
                        uint16_t p_roll = transform2(roll);
                        uint16_t p_yaw = transform2(yaw);
                        uint16_t p_pitch = transform2(pitch);
                        cout<<"px ="<<x<<endl;
                    cout<<"py ="<<y<<endl;
                    cout<<"pz ="<<z<<endl;
                    cout<<"proll ="<<p_roll <<endl;
                    cout<<"pyaw ="<<p_yaw<<endl;
                    cout<<"ppitch ="<<p_pitch <<endl;

                        struct data mes(85, 15, 0, 1, p_x, p_y, p_z, p_roll, p_yaw, p_pitch, 0);
//        char data1[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
                        unsigned short c1;
//        c1 = crc8.crcCompute((char*) &mes, 9);
                        int length = sizeof(mes);
                        c1 = crc8((uint8_t*)&mes, length - 2);
                        
                        mes.m_crc = c1;
//        printf("")
                        //发送消息给服务器端
                       send(sockfd, &mes, sizeof(mes)-1, 0);
                       // unsigned char buffer[BUFFER_SIZE];
                        //ssize_t bytes_received = recv(sockfd, buffer, BUFFER_SIZE, 0);
                       // if (buffer[0] == 0){
                         //   
                         //   close(sockfd);
                         //   break;
                      //  }
                        

                    };
		    
		 sprintf(camPositionStr,"Camera Position: px=%f, py=%f, pz=%f", camPosMatrix.at<float>(0,0), 
                 camPosMatrix.at<float>(1,0), camPosMatrix.at<float>(2,0));
                 sprintf(camDirectionStr,"Camera Direction: dx=%f, dy=%f, dz=%f", camVecMatrix.at<float>(0,0), 
                camVecMatrix.at<float>(1,0), camVecMatrix.at<float>(2,0));
                }
            if (index>1) avrgTimer.end();
            cout<<"Average computing time "<<avrgTimer.getAverage()<<" ms"<<endl;
            
             if (TheMSPoseTracker.isValid())
                frame_pose_map.insert(make_pair(index,TheMSPoseTracker.getRTMatrix()));
                
            // print the markers detected that belongs to the markerset
            for (auto idx : TheMarkerMapConfig.getIndices(detected_markers))
                 {
                //detected_markers[idx].draw(TheInputImageCopy, Scalar(0, 0, 255), 1);
                detected_markers[idx].draw(TheInputImageCopy, cv::Scalar(0, 0, 255),1);
				if(ref_id == detected_markers[idx].id) aruco::CvDrawingUtils::draw3dAxis(TheInputImageCopy,detected_markers[idx],TheCameraParameters);
				}
		    putText(TheInputImageCopy,camPositionStr,cv::Point(10,13),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,255,255));
		    putText(TheInputImageCopy,camDirectionStr,cv::Point(10,30),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,255,255));
//imshow("out",TheInputImageCopy);
//***key = waitKey(waitTime);
            // show  informatino
            cout<<"flag=";
            cout<<(flag++)<<endl;
            key = Viewer.show(TheMarkerMapConfig,TheMSPoseTracker.getRTMatrix(),TheInputImageCopy,waitTime);
            if (key=='s') waitTime=waitTime?0:10;
            if(flag%1000==0){
           savePCDFile(cml("-pcd"), TheMarkerMapConfig, frame_pose_map);
        savePosesToFile(cml("-poses.txt"), frame_pose_map);
            }

        } while (key != 27 && TheVideoCapturer.grab());

        // save a beatiful pcd file (pcl library) showing the results (you can use pcl_viewer to see it)
        if (cml["-pcd"])
            savePCDFile(cml("-pcd"), TheMarkerMapConfig, frame_pose_map);
            savePosesToFile(cml("-poses.txt"), frame_pose_map);
    }
    catch (std::exception& ex)

    {
        cout << "Exception :" << ex.what() << endl;
    }
}



void  getQuaternionAndTranslationfromMatrix44(const cv::Mat &M_in ,double &qx,double &qy,double &qz,double &qw,double &tx,double &ty,double &tz){
    //get the 3d part of matrix and get quaternion
    assert(M_in.total()==16);
    cv::Mat M;
    M_in.convertTo(M,CV_64F);
    cv::Mat r33=cv::Mat ( M,cv::Rect ( 0,0,3,3 ) );
    //use now eigen
    Eigen::Matrix3f e_r33;
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            e_r33(i,j)=M.at<double>(i,j);

    //now, move to a angle axis
    Eigen::Quaternionf q(e_r33);
    qx=q.x();
    qy=q.y();
    qz=q.z();
    qw=q.w();


    tx=M.at<double>(0,3);
    ty=M.at<double>(1,3);
    tz=M.at<double>(2,3);
}

void savePosesToFile(string filename, const std::map<int, cv::Mat>& fmp)
{
    std::ofstream file(filename);
    double qx, qy, qz, qw, tx, ty, tz;
    for (auto frame : fmp)
    {
        if (!frame.second.empty())
        {
            cv::Mat minv=frame.second.inv();
            getQuaternionAndTranslationfromMatrix44(minv, qx, qy, qz, qw, tx, ty, tz);
            file << frame.first << " " << tx << " " << ty << " " << tz << " " << qx << " " << qy << " " << qz << " "
                 << qw << endl;
        }
    }
}

uint16_t l2b(uint16_t num)
{
    union{
        int16_t a;
        uint8_t b[2];
    }t;
    t.a=num;
    uint8_t temp = t.b[0];
    t.b[0] =t.b[1];
    t.b[1] = temp;

    return t.a;
}

uint16_t transform(float &data) {
    int MAX = 32767;
    bool sig = false;
    if (data > 0) {
        sig = true;
    }
    auto d = static_cast<uint16_t>(std::abs(data * 100));
    if (d > MAX) {
        return 0;
    }

    if (!sig) d = (d | (int16_t) 32768);
    //cout<<d<<endl;
    //cout<<hex<<d<<endl;
    //cout<<hex<<htonl(d)<<endl;
    //printf("%x", htonl(d));
//    return htonl(d);
    d = l2b(d);
    return d;
}

uint8_t crc8(uint8_t *data, int length) {
    uint8_t crc = 0;
    for (int i = 0; i < length; i++) {
        crc ^= data[i];

        for (int times = 8; times > 0; times--) {
            if (crc & 0x80) crc = (crc << 1) ^ 0x07;
            else crc <<= 1;
            crc &= 0xFF;
        }

    }
    return crc;


};
