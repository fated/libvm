CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 2
OS = $(shell uname)

all: vm-offline

# lib: utilities.o
# 	if [ "$(OS)" = "Darwin" ]; then \
# 		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
# 	else \
# 		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so.$(SHVER)"; \
# 	fi; \
# 	$(CXX) $${SHARED_LIB_FLAG} svm.o -o libsvm.so.$(SHVER)

vm-offline: vm-offline.cpp utilities.o
	$(CXX) $(CFLAGS) vm-offline.cpp utilities.o -o vm-offline -lm

# objects = foo.o bar.o
# all: $(objects)
# $(objects): %.o: %.cpp %.h
# $(CXX) $(CFLAGS) -c $< -o $@

utilities.o: utilities.cpp utilities.h
	$(CXX) $(CFLAGS) -c utilities.cpp

clean:
	rm -f utilities.o vm-offline
