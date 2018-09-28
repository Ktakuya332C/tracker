CXX=g++
CXXFLAGS=-g -std=c++11 -Wall -I../../third_party/eigen3/

.PHONY: all
all: tracker_test tracker_helper_test

tracker_test: tracker_test.o tracker.o tracker_helper.o math_helper.o
	$(CXX) $(CXXFLAGS) $^ -o $@

tracker_helper_test: tracker_helper_test.o tracker_helper.o math_helper.o
	$(CXX) $(CXXFLAGS) $^ -o $@

tracker_helper_test.o: tracker_helper_test.cc
	$(CXX) $(CXXFLAGS) -c $^ -o $@

tracker_helper.o: tracker_helper.cc
	$(CXX) $(CXXFLAGS) -c $^ -o $@

tracker_test.o: tracker_test.cc
	$(CXX) $(CXXFLAGS) -c $^ -o $@

tracker.o: tracker.cc
	$(CXX) $(CXXFLAGS) -c $^ -o $@

math_helper.o: math_helper.cc
	$(CXX) $(CXXFLAGS) -c $^ -o $@

.PHONY: clean
clean:
	rm *.o tracker_test tracker_helper_test
