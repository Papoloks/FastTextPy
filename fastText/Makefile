CXX = c++
CXXFLAGS = -pthread -std=c++0x
OBJS = args.o dictionary.o matrix.o vector.o model.o utils.o
INCLUDES = -I.

opt: CXXFLAGS += -O3 -funroll-loops -fPIC
opt: fasttextPy


args.o: src/args.cc src/args.h
	$(CXX) $(CXXFLAGS) -c src/args.cc

dictionary.o: src/dictionary.cc src/dictionary.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/dictionary.cc

matrix.o: src/matrix.cc src/matrix.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/matrix.cc

vector.o: src/vector.cc src/vector.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/vector.cc

model.o: src/model.cc src/model.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/model.cc

utils.o: src/utils.cc src/utils.h
	$(CXX) $(CXXFLAGS) -c src/utils.cc

fasttextPy : $(OBJS) src/py.cpp
	$(CXX) $(CXXFLAGS) $(OBJS) src/py.cpp -shared -o libfasttext.so -fPIC

clean:
	rm -rf *.o libfasttext.so