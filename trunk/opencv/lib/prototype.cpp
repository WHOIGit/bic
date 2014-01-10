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

class LearnJob {
public:
  string inpath;
  bool stop;
  LearnJob(String inp) {
    inpath = inp;
    stop = false;
  }
  LearnJob() {
    stop = true;
  }
};

void learn_worker(Lightfield *model, ThreadsafeQueue<LearnJob>* queue) {
  static boost::mutex mutex;
  while(true) {
    LearnJob job = queue->pop();
    if(job.stop) {
      cout << "DONE with one thread" << endl;
      return;
    } else {
      string inpath = job.inpath;
      cout << "POPPED " << inpath << endl;
      Mat cfa_LR = imread(inpath, CV_LOAD_IMAGE_ANYDEPTH);
      cout << "Read " << inpath << endl;
      {
	boost::lock_guard<boost::mutex> lock(mutex);
	model->addImage(cfa_LR);
      }
      cout << "Added " << inpath << endl;
    }
  }
}

class CorrectJob {
public:
  string inpath;
  string outpath;
  bool stop;
  CorrectJob(string inp, string outp) {
    inpath = inp;
    outpath = outp;
    stop = false;
  }
  CorrectJob() {
    stop = true;
  }
};

void correct_worker(Lightfield *model, ThreadsafeQueue<CorrectJob>* queue) {
  while(true) {
    CorrectJob job = queue->pop();
    if(job.stop) {
      cout << "DONE with one thread" << endl;
      return;
    }
    string inpath = job.inpath;
    string outpath = job.outpath;
    cout << "POPPED " << inpath << endl;
    Mat cfa_LR = model->correct(imread(inpath, CV_LOAD_IMAGE_ANYDEPTH));
    cout << "Demosaicing " << inpath << endl;
    Mat rgb_LR = demosaic(cfa_LR,"rgGb");
    cout << "Saving RGB to " << outpath << endl;
    Mat rgb_LR_16u;
    rgb_LR.convertTo(rgb_LR_16u, CV_16U);
    imwrite(outpath, rgb_LR_16u);
  }
}

void learn_prototype() {
  ifstream inpaths(PATH_FILE);
  string inpath;
  Lightfield model;
  ThreadsafeQueue<LearnJob> work;
  boost::thread_group workers;
  while(getline(inpaths,inpath)) {
    work.push(LearnJob(inpath));
    cout << "PUSHED " << inpath << endl;
  }
  for(int i = 0; i < N_THREADS; i++) {
    boost::thread* worker = new boost::thread(learn_worker, &model, &work);
    workers.add_thread(worker);
  }
  for(int i = 0; i < N_THREADS; i++) {
    work.push(LearnJob()); // tell thread to stop
  }
  workers.join_all();
  cout << "Saving lightmap" << endl;
  model.save("model.tiff");
  cout << "SUCCESS" << endl;
}

void correct_prototype() {
  ifstream inpaths(PATH_FILE);
  string inpath;
  Lightfield model;
  cout << "Loading lightmap" << endl;
  model.load("model.tiff");
  int count = 0;
  ifstream inpaths2(PATH_FILE);
  ThreadsafeQueue<CorrectJob> work;
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
    work.push(CorrectJob(inpath,outpath));
    cout << "PUSHED " << inpath << endl;
    count++;
  }
  for(int i = 0; i < N_THREADS; i++) {
    work.push(CorrectJob()); // tell thread to stop
  }
  workers.join_all();
  cout << "SUCCESS" << endl;
}
