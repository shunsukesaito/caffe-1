#include <iostream>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <memory>

#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>      /* for fprintf */
#include <string.h>     /* for memcpy */
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <pthread.h>
#include <time.h>

#include <time.h>

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

#define DEFAULT_DEPLOY_PATH "model/deloy.prototxt"
#define DEFAULT_WEIGHT_PATH "model/weight.caffemodel"
using namespace caffe;
using namespace cv;
using namespace std;

class TCPServer
{
public:
    TCPServer(const std::string &address, const short port = 0)
    {
        address_ = address;
        port_ = port;
        // create socket first
        std::cout << "Creating Socket..." << std::endl;
        if ((fd_ = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
            perror("cannot create socket");
        }
        int enable = 1;
        if (setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0)
            perror("setsockopt(SO_REUSEADDR) failed");
        int flags;
#if defined(O_NONBLOCK)
        /* Fixme: O_NONBLOCK is defined but broken on SunOS 4.1.x and AIX 3.2.5. */
        if (-1 == (flags = fcntl(fd, F_GETFL, 0)))
            flags = 0;
        fcntl(fd_, F_SETFL, flags | O_NONBLOCK);
#else
        /* Otherwise, use the old way of doing it */
        flags = 1;
        ioctl(fd_, FIONBIO, &flags);
#endif
        /* fill in the server's address and data */
        std::cout << "Setting Host..." << std::endl;
        memset((char*)&servaddr_, 0, sizeof(servaddr_));
        servaddr_.sin_family = AF_INET;
        servaddr_.sin_addr.s_addr = inet_addr(address.c_str());
        servaddr_.sin_port = htons(port);
        
        if (::bind(fd_, (struct sockaddr *)&servaddr_, sizeof(servaddr_)) < 0) {
            perror("bind failed");
        }
        
        int true_val = 1;
        setsockopt(fd_,SOL_SOCKET,SO_REUSEADDR,&true_val,sizeof(int));
        setsockopt(newfd_,SOL_SOCKET,SO_REUSEADDR,&true_val,sizeof(int));
    }

    ~TCPServer(){close(newfd_); close(fd_);}
    
    bool Wait(clock_t max_wait = 10)
    {
        listen(fd_,5);
        clilen_ = sizeof(cliaddr_);
        std::cout << "Wainting IP: " << address_ << " Port: " <<  port_ << "..." << std::endl;
        
        clock_t start_ = clock();
        do{
            newfd_ = accept(fd_, (struct sockaddr *) &cliaddr_, &clilen_);
            if(clock()-start_ > CLOCKS_PER_SEC*max_wait){
                std::cout << "Error: Client Session Timeout" << std::endl;
                return false;
            }
        }while(newfd_ < 0);
        
        std::cout << "Connection Established..." << std::endl;
        return true;
    }
    
    bool Send(void *data, size_t buffer_size)
    {
        // send data
        int n = write(newfd_,data,buffer_size);
        if (n < 0){
            perror("send failed");
            return false;
        }
        return true;
    }
    
    bool RcvImage(unsigned char* data, size_t buffer_size)
    {
        int rcvd_size = 0;
        
        start_idle_ = clock();
        while(rcvd_size < buffer_size){
            int n = read(newfd_,data+rcvd_size,buffer_size-rcvd_size);
            if (n < 0){
                perror("ERROR reading from socket");
                return false;
            }
            else if(clock() - start_idle_ > CLOCKS_PER_SEC*2){
                std::cout << "Disconnected..." << std::endl;
                return false;
            }
            
            rcvd_size += n;
        }
        return true;
    }
    
    void Close()
    {
        std::cout << "Socket is closing..." << std::endl; close(newfd_); close(fd_);
    }
    
private:
    int fd_, newfd_;
    clock_t start_idle_ = 0.0;
    struct sockaddr_in cliaddr_;
    struct sockaddr_in servaddr_;
    socklen_t clilen_;
    short port_;
    std::string address_;
};

class SegNet
{
public:
    SegNet()
    : net(DEFAULT_DEPLOY_PATH), deploy_path(DEFAULT_DEPLOY_PATH), weight_path(DEFAULT_WEIGHT_PATH), mode("cpu"), gpu_id(0){initialize();}
    SegNet(int device_id)
    : net(DEFAULT_DEPLOY_PATH), deploy_path(DEFAULT_DEPLOY_PATH), weight_path(DEFAULT_WEIGHT_PATH)
    {mode = (device_id<0)?"cpu":"gpu"; gpu_id = device_id; initialize();}
    SegNet( string deploy, string weight, string m, int gpu=0 )
    : net( deploy )
    {
        deploy_path = deploy;
        weight_path = weight;
        mode = m;
        gpu_id = gpu;
        
        initialize();
    }
    
    ~SegNet() {}
    
    void initialize()
    {
        Caffe::set_phase( Caffe::TEST );
        if ( mode.compare( "gpu" ) == 0 )
        {
            Caffe::SetDevice( gpu_id );
            Caffe::set_mode( Caffe::GPU );
            cout << "SegNet::initialize(): using GPU with device " << gpu_id << endl;
        }
        else if ( mode.compare( "cpu" ) == 0 )
        {
            Caffe::set_mode( Caffe::CPU );
            cout << "SegNet::initialize(): using CPU." << endl;
        }
        else
        {
            cerr << "SegNet::initialize(): WARNING: invalid mode, set to cpu mode for now." << endl;
            Caffe::set_mode( Caffe::CPU );
        }
        net.CopyTrainedLayersFrom( weight_path );
    }
    
    
    Mat infer( Mat& image )
    {
        // load the image into data layer
        vector<cv::Mat> imageVector;
        imageVector.push_back( image );
        
        vector<int> labelVector;
        labelVector.push_back(0); //push_back 0 for initialize purpose
        
        boost::shared_ptr<MemoryDataLayer<float> > memory_data_layer;
        memory_data_layer = boost::static_pointer_cast<MemoryDataLayer<float> >( net.layer_by_name( "data" ) );
        memory_data_layer->AddMatVector( imageVector, labelVector );
        
        // run ForwardPrefilled
        float loss = 0.0;
        vector<Blob<float>*> results = net.ForwardPrefilled( &loss );
        
        // return the result as Mat
        const vector< boost::shared_ptr< Blob< float > > > net_blobs = net.blobs();
        boost::shared_ptr< Blob< float > > softmax_blob = net_blobs[ net_blobs.size()-1 ];
        
        // TODO: check if this is truly the softmax blob
        
        // int num      = softmax_blob->num();
        // int channels = softmax_blob->channels();
        int height   = softmax_blob->height();
        int width    = softmax_blob->width();
        
        cv::Mat pmap( height, width, CV_32FC1, softmax_blob->mutable_cpu_data() );
        
        return pmap;
    }

private:
    // network
    Net<float>  net;
    
    // parameters
    string      deploy_path;
    string      weight_path;
    string      mode;
    int         gpu_id;
};

int main(int argc, const char **argv)
{
    std::vector<std::string> args(argv, argv + argc);
    int n = 1;
    std::string ip = "127.0.0.1";
    int port = 1153;
    std::string deploy_path = DEFAULT_DEPLOY_PATH;
    std::string weight_path = DEFAULT_WEIGHT_PATH;
    
    int gpu = -1;
    while ( n < argc )
    {
        if ( args[n] == "-ip" )
        {
            ip = args[n+1];
            n += 2;
        }
        else if ( args[n] == "-p" )
        {
            ip = atoi(args[n+1].c_str());
            n += 2;
        }
        else if( args[n] == "-d")
        {
            deploy_path = args[n+1];
            n += 2;
        }
        else if(args[n] == "-w")
        {
            weight_path = args[n+1];
            n += 2;
        }
        else if( args[n] == "-g" )
        {
            gpu = atoi(args[n+1].c_str());
            n += 2;
        }
        else
        {
            ++ n;
        }
    }
    
    SegNet* prob_net = new SegNet(deploy_path,weight_path,(gpu<0)?"cpu":"gpu",gpu);
    
    cv::Mat image(128,128,CV_8UC3);
    cv::Mat probmap(128,128,CV_32FC1);
    while(1)
    {
        TCPServer server(ip.c_str(),port);
        if(server.Wait(10)){
            if(server.RcvImage(image.ptr(),(int)image.total()*image.channels())){
                probmap = prob_net->infer(image);
                
                std::cout << "processed " << std::endl;
                
                server.Send(probmap.ptr(), (int)sizeof(float)*probmap.total()*probmap.channels());
                std::cout << "sent image" << std::endl;
            }
        }
        
        server.Close();
    }
}
