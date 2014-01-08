CXXFLAGS=-I/usr/include/opencv -g
LDLIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann -lpthread

SRCS=demosaic.cpp main.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: app

app: $(OBJS)
	g++ $(LDFLAGS) -o app $(OBJS) $(LDLIBS) 

%.o: %.cpp %.hpp
	g++ -c $< -o $@

clean:
	$(RM) $(OBJS)
	$(RM) app
