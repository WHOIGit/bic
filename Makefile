# apt packages required include libopencv-dev libboost-dev libboost-thread-dev
CXXFLAGS=-I/usr/include/opencv -g
LDLIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann -lboost_thread-mt -lboost_regex -lboost_system

SRCS=main.cpp demosaic.cpp prototype.cpp illumination.cpp stereo.cpp interpolation.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

includes = $(wildcard *.hpp)

all: app

app: $(OBJS)
	g++ $(LDFLAGS) -o app $(OBJS) $(LDLIBS) 

%.o: %.cpp ${includes}
	g++ -c -g $< -o $@

clean:
	$(RM) $(OBJS)
	$(RM) app
