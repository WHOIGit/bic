#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "demosaic.hpp"
#include "Lightfield.hpp"
#include "ThreadsafeQueue.hpp"

using namespace std;
using namespace cv;

#define PATH_FILE "paths.txt"
#define MODEL_FILE "model.tiff"
#define OUT_DIR "out"
#define N_THREADS 8

class Job {
public:
  string inpath;
  string outpath;
  bool stop;
  Job(string inp, string outp) {
    inpath = inp;
    outpath = outp;
  }
  Job() {
    stop = true;
  }
};

void correct_worker(Lightfield *model, ThreadsafeQueue<Job>* queue) {
  while(true) { // FIXME need some way to stop
    Job job = queue->pop();
    if(job.stop) {
      cout << "DONE with one thread" << endl;
      return;
    }
    string inpath = job.inpath;
    string outpath = job.outpath;
    cout << "POPPED " << inpath << endl;
    Mat cfa_LR = model->correct(imread(inpath, CV_LOAD_IMAGE_ANYDEPTH));
    cout << "Demosaicing " << inpath << endl;
    Mat rgb_LR = demosaic_hqlinear(cfa_LR,"rggb");
    cout << "Saving RGB to " << outpath << endl;
    Mat rgb_LR_16u;
    rgb_LR.convertTo(rgb_LR_16u, CV_16U);
    imwrite(outpath, rgb_LR_16u);
  }
}

int main(int argc, char** argv) {
  ifstream inpaths(PATH_FILE);
  string inpath;
  Lightfield model;
  if(string(argv[1]) == "learn") {
    while(getline(inpaths,inpath)) {
      cout << "Learning " << inpath << endl;
      Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
      model.addImage(cfa_LR);
    }
    model.save("model.tiff");
  } else if(string(argv[1]) == "correct") {
    model.load("model.tiff");
    cout << "Loaded model" << endl;
    int count = 0;
    ifstream inpaths2(PATH_FILE);
    ThreadsafeQueue<Job> work;
    boost::thread_group workers;
    for(int i = 0; i < N_THREADS; i++) {
      boost::thread* worker = new boost::thread(correct_worker, &model, &work);
      workers.add_thread(worker);
    }
    while(getline(inpaths2,inpath)) {
      stringstream outpaths;
      string outpath;
      outpaths << "out/correct" << count << ".tiff";
      outpath = outpaths.str();
      work.push(Job(inpath,outpath));
      cout << "PUSHED " << inpath << endl;
      count++;
    }
    for(int i = 0; i < N_THREADS; i++) {
      work.push(Job()); // tell thread to stop
    }
    workers.join_all();
  }
  return 0;
}
