#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "prototype.hpp"

using namespace std;

int main(int argc, char **argv) {
  if(string(argv[1])=="learn") {
    learn_prototype();
  } else {
    correct_prototype();
  }
}
