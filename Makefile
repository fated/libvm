CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 2
OS = $(shell uname)

all: vm-offline vm-online vm-cv

vm-offline: vm-offline.cpp utilities.o knn.o svm.o vm.o
	$(CXX) $(CFLAGS) vm-offline.cpp utilities.o knn.o svm.o vm.o -o vm-offline -lm

vm-online: vm-online.cpp utilities.o knn.o svm.o vm.o
	$(CXX) $(CFLAGS) vm-online.cpp utilities.o knn.o svm.o vm.o -o vm-online -lm

vm-cv: vm-cv.cpp utilities.o knn.o svm.o vm.o
	$(CXX) $(CFLAGS) vm-cv.cpp utilities.o knn.o svm.o vm.o -o vm-cv -lm

utilities.o: utilities.cpp utilities.h
	$(CXX) $(CFLAGS) -c utilities.cpp

knn.o: knn.cpp knn.h
	$(CXX) $(CFLAGS) -c knn.cpp

svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp

vm.o: vm.cpp vm.h
	$(CXX) $(CFLAGS) -c vm.cpp

clean:
	rm -f utilities.o knn.o svm.o vm.o vm-offline vm-online vm-cv
